"""RPC server wrapping a frozen VLA-Adapter policy (replaces RLinf Pi0.5)."""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from rpent.utils.config import get_adapter_checkpoint_path, get_vla_adapter_root
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

logger = get_logger("vla_server")

# Make the parent VLA-Adapter package importable (experiments / prismatic).
_ADAPTER_ROOT = str(get_vla_adapter_root())
if _ADAPTER_ROOT not in sys.path:
    sys.path.insert(0, _ADAPTER_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from experiments.robot.robot_utils import (  # noqa: E402
    invert_gripper_action,
    normalize_gripper_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AdapterInferConfig:
    """Minimal config surface matching VLA-Adapter eval scripts."""

    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_minivlm: bool = True
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = NUM_ACTIONS_CHUNK
    unnorm_key: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_pro_version: bool = True
    save_version: str = "vla-adapter-harness"


def _decode_image_block(block: dict[str, Any]) -> np.ndarray:
    import imageio.v2 as imageio

    fmt = (block.get("format") or "png").lower()
    if fmt != "png":
        raise ValueError(f"unsupported image format: {fmt!r} (only 'png')")
    data = block.get("data")
    if not isinstance(data, str) or not data:
        raise ValueError("image block missing base64 'data'")
    raw = base64.b64decode(data)
    img = np.asarray(imageio.imread(io.BytesIO(raw)))
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3 RGB; got {img.shape}")
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _resolve_unnorm_key(model, unnorm_key: str) -> str:
    key = unnorm_key
    # Strip LIBERO-Pro perturbation suffixes (e.g. libero_spatial_swap → libero_spatial).
    for base in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
        if key == base or key.startswith(base + "_"):
            key = base
            break
    if key not in model.norm_stats and f"{key}_no_noops" in model.norm_stats:
        key = f"{key}_no_noops"
    if key not in model.norm_stats:
        raise KeyError(
            f"Action un-norm key {key!r} not found in VLA norm_stats "
            f"(available: {sorted(model.norm_stats.keys())})"
        )
    return key


def _process_actions_for_env(actions: List[np.ndarray]) -> np.ndarray:
    """Stack chunk actions and apply LIBERO gripper conventions."""
    arr = np.stack([np.asarray(a, dtype=np.float32) for a in actions], axis=0)
    arr = normalize_gripper_action(arr, binarize=True)
    arr = invert_gripper_action(arr)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Facade implementing the rpent.utils.vla_client protocol
# ---------------------------------------------------------------------------


class VLAFacade(RpcFacade):
    """Implements :class:`rpent.utils.vla_client.VLAClient` over VLA-Adapter."""

    def __init__(self, cfg: AdapterInferConfig):
        super().__init__()
        t0 = time.time()
        logger.info(
            "loading VLA-Adapter (checkpoint=%s, unnorm_key=%s, pro=%s) ...",
            cfg.pretrained_checkpoint,
            cfg.unnorm_key,
            cfg.use_pro_version,
        )
        self.cfg = cfg
        self.vla = get_vla(cfg)
        if hasattr(self.vla, "set_version"):
            self.vla.set_version(cfg.save_version)

        self.cfg.unnorm_key = _resolve_unnorm_key(self.vla, str(cfg.unnorm_key))

        self.proprio_projector = None
        if cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(
                cfg, self.vla.llm_dim, proprio_dim=PROPRIO_DIM
            )

        self.action_head = None
        if cfg.use_l1_regression:
            self.action_head = get_action_head(cfg, self.vla.llm_dim)

        self.processor = get_processor(cfg)
        logger.info("model ready in %.1fs", time.time() - t0)

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method == "predict":
            return self.predict(*args, **kwargs)
        raise ValueError(f"unknown RPC method: {method!r}")

    def predict(
        self,
        instruction: str,
        images: dict[str, Any],
        state: list,
        mode: str = "eval",
    ) -> dict[str, Any]:
        del mode  # VLA-Adapter eval path has no train/eval split here.
        if "main" not in images:
            raise ValueError("'images.main' is required")
        main = _decode_image_block(images["main"])
        obs: dict[str, Any] = {"full_image": main}

        if isinstance(images.get("wrist"), dict):
            obs["wrist_image"] = _decode_image_block(images["wrist"])

        states = np.asarray(state, dtype=np.float32)
        if states.ndim == 2:
            states = states[0]
        if states.ndim != 1:
            raise ValueError(f"state must be [state_dim] or [B,state_dim]; got {states.shape}")
        obs["state"] = states

        actions = get_vla_action(
            self.cfg,
            self.vla,
            self.processor,
            obs,
            instruction,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
            use_minivlm=self.cfg.use_minivlm,
        )
        actions_np = _process_actions_for_env(actions)
        # Wire schema: [B=1, chunk, action_dim]
        batched = actions_np[None]
        return {
            "actions": batched.tolist(),
            "shape": list(batched.shape),
            "dtype": "float32",
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="VLA-Adapter RPC server for sde_harness")
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument(
        "--model-path",
        default=None,
        help="VLA-Adapter checkpoint (defaults to ADAPTER_CHECKPOINT_PATH)",
    )
    p.add_argument(
        "--unnorm-key",
        default=None,
        help="Action unnorm key / base suite name (e.g. libero_spatial)",
    )
    p.add_argument("--num-images-in-input", type=int, default=2)
    p.add_argument("--use-proprio", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-film", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-minivlm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--use-pro-version",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load Pro action-head / projector checkpoints",
    )
    args = p.parse_args()

    model_path = args.model_path or get_adapter_checkpoint_path()
    if not model_path:
        raise RuntimeError(
            "ADAPTER_CHECKPOINT_PATH is not set; provide the VLA-Adapter "
            "checkpoint via --model-path or the environment."
        )
    unnorm_key = args.unnorm_key or os.environ.get("ADAPTER_UNNORM_KEY", "")
    if not unnorm_key:
        raise RuntimeError(
            "unnorm key missing; pass --unnorm-key (e.g. libero_spatial) "
            "or set ADAPTER_UNNORM_KEY."
        )

    cfg = AdapterInferConfig(
        pretrained_checkpoint=model_path,
        unnorm_key=unnorm_key,
        num_images_in_input=args.num_images_in_input,
        use_proprio=args.use_proprio,
        use_film=args.use_film,
        use_minivlm=args.use_minivlm,
        use_pro_version=args.use_pro_version,
    )
    facade = VLAFacade(cfg)
    facade.serve(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
