"""
run_libero_pro_eval.py

Evaluate a trained policy on the LIBERO-Pro generalization benchmark
(perturbed LIBERO suites). This is distinct from VLA-Adapter Pro weights
(`--use_pro_version` / checkpoints named `*-Pro`).
"""

# NOTE: Do not enable `from __future__ import annotations` here.
# draccus.wrap() needs runtime dataclass type hints (same as run_libero_eval.py).

import json
import logging
import os
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import yaml

# ---------------------------------------------------------------------------
# Point imports at third_party/LIBERO-PRO before importing `libero`.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_PRO_ROOT = REPO_ROOT / "third_party" / "LIBERO-PRO"
LIBERO_PRO_PKG = LIBERO_PRO_ROOT / "libero" / "libero"

if not LIBERO_PRO_ROOT.is_dir():
    raise FileNotFoundError(
        f"LIBERO-Pro package not found at {LIBERO_PRO_ROOT}. "
        "Clone https://github.com/Zxy-MLlab/LIBERO-PRO into third_party/LIBERO-PRO"
    )

_LIBERO_PRO_CONFIG_DIR = REPO_ROOT / "experiments" / "robot" / "libero" / ".libero_pro_config"
_LIBERO_PRO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_LIBERO_PRO_CONFIG_FILE = _LIBERO_PRO_CONFIG_DIR / "config.yaml"
if not _LIBERO_PRO_CONFIG_FILE.exists():
    with open(_LIBERO_PRO_CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "benchmark_root": str(LIBERO_PRO_PKG),
                "bddl_files": str(LIBERO_PRO_PKG / "bddl_files"),
                "init_states": str(LIBERO_PRO_PKG / "init_files"),
                "datasets": str(LIBERO_PRO_ROOT / "libero" / "datasets"),
                "assets": str(LIBERO_PRO_PKG / "assets"),
            },
            f,
        )

os.environ["LIBERO_CONFIG_PATH"] = str(_LIBERO_PRO_CONFIG_DIR)
# Prefer LIBERO-PRO over the original ./LIBERO package.
sys.path = [p for p in sys.path if Path(p).resolve() != (REPO_ROOT / "LIBERO").resolve()]
sys.path.insert(0, str(LIBERO_PRO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import perturbation  # noqa: E402  # from third_party/LIBERO-PRO
from libero.libero import benchmark  # noqa: E402

import wandb  # noqa: E402

sys.path.append("../..")
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (  # noqa: E402
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK  # noqa: E402


BASE_TASK_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)

class TaskSuite(str, Enum):
    # Base suites (input to perturbation setup)
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"

    # LIBERO-Pro perturbed suites
    LIBERO_GOAL_TEMP = "libero_goal_temp"
    LIBERO_SPATIAL_TEMP = "libero_spatial_temp"
    LIBERO_10_TEMP = "libero_10_temp"
    LIBERO_OBJECT_TEMP = "libero_object_temp"
    LIBERO_GOAL_LAN = "libero_goal_lan"
    LIBERO_SPATIAL_LAN = "libero_spatial_lan"
    LIBERO_10_LAN = "libero_10_lan"
    LIBERO_OBJECT_LAN = "libero_object_lan"
    LIBERO_GOAL_OBJECT = "libero_goal_object"
    LIBERO_SPATIAL_OBJECT = "libero_spatial_object"
    LIBERO_10_OBJECT = "libero_10_object"
    LIBERO_OBJECT_OBJECT = "libero_object_object"
    LIBERO_GOAL_SWAP = "libero_goal_swap"
    LIBERO_SPATIAL_SWAP = "libero_spatial_swap"
    LIBERO_10_SWAP = "libero_10_swap"
    LIBERO_OBJECT_SWAP = "libero_object_swap"
    LIBERO_GOAL_TASK = "libero_goal_task"
    LIBERO_SPATIAL_TASK = "libero_spatial_task"
    LIBERO_10_TASK = "libero_10_task"
    LIBERO_OBJECT_TASK = "libero_object_task"
    LIBERO_GOAL_ENV = "libero_goal_env"
    LIBERO_SPATIAL_ENV = "libero_spatial_env"
    LIBERO_10_ENV = "libero_10_env"
    LIBERO_OBJECT_ENV = "libero_object_env"


_BASE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

TASK_MAX_STEPS = {}
for suite in TaskSuite:
    base = next((b for b in BASE_TASK_SUITES if suite.value == b or suite.value.startswith(b + "_")), None)
    if base is not None:
        TASK_MAX_STEPS[suite.value] = _BASE_MAX_STEPS[base]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    use_l1_regression: bool = True
    use_minivlm: bool = True
    num_diffusion_steps: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True

    center_crop: bool = True
    num_open_loop_steps: int = 8
    unnorm_key: Union[str, Path] = ""

    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Base suite before perturbation (e.g. libero_spatial). Setup rewrites to *_env / *_swap / ...
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256
    # Optional smoke / subset eval (0 = all tasks).
    max_tasks: int = 0
    start_task_id: int = 0

    evaluation_config_path: str = str(
        REPO_ROOT / "experiments" / "robot" / "libero" / "libero_pro_evaluation_config.yaml"
    )

    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    seed: int = 7

    save_version: str = "vla-adapter-libero-pro"
    save_videos: bool = False  # rollout MP4s off by default; logs still written
    use_pro_version: bool = True  # VLA-Adapter Pro *model*; unrelated to LIBERO-Pro benchmark
    phase: str = "Inference"
    # fmt: on


def get_base_task_suite(task_suite_name: str) -> str:
    for base in BASE_TASK_SUITES:
        if task_suite_name == base or task_suite_name.startswith(base + "_"):
            return base
    return task_suite_name


def resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def load_evaluation_config(cfg: GenerateConfig) -> dict:
    config_path = Path(resolve_path(cfg.evaluation_config_path))
    with open(config_path, "r", encoding="utf-8") as f:
        evaluation_cfg = yaml.safe_load(f)

    evaluation_cfg["bddl_files_path"] = resolve_path(evaluation_cfg["bddl_files_path"])
    evaluation_cfg["init_file_dir"] = resolve_path(evaluation_cfg["init_file_dir"])
    evaluation_cfg["script_path"] = resolve_path(evaluation_cfg["script_path"])

    ood = evaluation_cfg.get("ood_task_configs", {})
    evaluation_cfg["ood_task_configs"] = {k: resolve_path(v) for k, v in ood.items()}
    return evaluation_cfg


def _suite_ready(bddl_dir: Path, init_dir: Path) -> bool:
    if not bddl_dir.is_dir() or not init_dir.is_dir():
        return False
    return any(bddl_dir.glob("*.bddl")) and any(init_dir.glob("*.pruned_init"))


def _replace_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))


def setup_libero_pro_suite(cfg: GenerateConfig, evaluation_cfg: dict) -> str:
    """Materialize perturbed suite dirs and return the suite name to evaluate."""
    base_suite = get_base_task_suite(cfg.task_suite_name)
    if base_suite not in BASE_TASK_SUITES:
        raise ValueError(f"Unsupported base suite for LIBERO-Pro: {cfg.task_suite_name}")

    bddl_root = Path(evaluation_cfg["bddl_files_path"])
    init_root = Path(evaluation_cfg["init_file_dir"])
    mapping = evaluation_cfg.get("perturbation_mapping", {})

    use_swap = bool(evaluation_cfg.get("use_swap", False))
    use_object = bool(evaluation_cfg.get("use_object", False))
    use_language = bool(evaluation_cfg.get("use_language", False))
    use_task = bool(evaluation_cfg.get("use_task", False))
    use_environment = bool(evaluation_cfg.get("use_environment", False))
    flags = [use_swap, use_object, use_language, use_task, use_environment]
    n_enabled = sum(flags)
    if n_enabled == 0:
        raise ValueError(
            "LIBERO-Pro evaluation_config must enable at least one of: "
            "use_environment / use_swap / use_object / use_language / use_task"
        )

    create_cfg = dict(evaluation_cfg)
    create_cfg["bddl_files_path"] = str(bddl_root / base_suite)
    create_cfg["task_suite_name"] = base_suite
    create_cfg["seed"] = int(evaluation_cfg.get("seed", cfg.seed))

    if n_enabled > 1:
        target_suite = f"{base_suite}_temp"
        bddl_dir = bddl_root / target_suite
        init_dir = init_root / target_suite
        expected_log = f"{use_swap},{use_object},{use_language},{use_task},{use_environment}"
        log_path = bddl_dir / "log.txt"

        need_create = not _suite_ready(bddl_dir, init_dir)
        if not need_create and log_path.exists():
            need_create = log_path.read_text(encoding="utf-8").strip() != expected_log
        elif not log_path.exists():
            need_create = True

        if need_create:
            if bddl_dir.exists():
                shutil.rmtree(bddl_dir)
            if init_dir.exists():
                shutil.rmtree(init_dir)
            bddl_dir.mkdir(parents=True, exist_ok=True)
            init_dir.mkdir(parents=True, exist_ok=True)
            log_path.write_text(expected_log, encoding="utf-8")
            logger.info("Creating multi-perturbation LIBERO-Pro suite: %s", target_suite)
            perturbation.create_env(configs=create_cfg)
        return target_suite

    # Single perturbation
    if use_swap:
        perturb_key = "use_swap"
    elif use_object:
        perturb_key = "use_object"
    elif use_language:
        perturb_key = "use_language"
    elif use_task:
        perturb_key = "use_task"
    else:
        perturb_key = "use_environment"

    suffix = mapping.get(perturb_key, "temp")
    target_suite = f"{base_suite}_{suffix}"
    bddl_dir = bddl_root / target_suite
    init_dir = init_root / target_suite

    if _suite_ready(bddl_dir, init_dir):
        logger.info("Using existing LIBERO-Pro suite: %s", target_suite)
        return target_suite

    logger.info(
        "Suite %s missing; generating via perturbation (may take a while)...",
        target_suite,
    )
    # create_env always writes to `{base}_temp`; rename to the official suffix.
    temp_bddl = bddl_root / f"{base_suite}_temp"
    temp_init = init_root / f"{base_suite}_temp"
    if temp_bddl.exists():
        shutil.rmtree(temp_bddl)
    if temp_init.exists():
        shutil.rmtree(temp_init)

    perturbation.create_env(configs=create_cfg)

    if not temp_bddl.exists():
        raise RuntimeError(f"perturbation.create_env did not create {temp_bddl}")

    _replace_dir(temp_bddl, bddl_dir)
    if temp_init.exists():
        _replace_dir(temp_init, init_dir)
    else:
        init_dir.mkdir(parents=True, exist_ok=True)

    if not _suite_ready(bddl_dir, init_dir):
        raise RuntimeError(
            f"Failed to materialize LIBERO-Pro suite '{target_suite}'. "
            "Prefer downloading prebuilt files: "
            "hf download zhouxueyang/LIBERO-Pro --repo-type dataset"
        )
    return target_suite


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    base = get_base_task_suite(cfg.task_suite_name)
    assert base in BASE_TASK_SUITES, f"Invalid task suite: {cfg.task_suite_name}"


def initialize_model(cfg: GenerateConfig):
    model = get_model(cfg)
    model.set_version(cfg.save_version)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    # Action stats come from the original LIBERO suite, not the perturbed suite name.
    if cfg.unnorm_key:
        unnorm_key = get_base_task_suite(str(cfg.unnorm_key))
    else:
        unnorm_key = get_base_task_suite(cfg.task_suite_name)

    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-LIBERO-PRO-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    initial_states = task_suite.get_task_init_states(task_id)
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    log_message("Using default initial states", log_file)
    return initial_states, None


def prepare_observation(obs, resize_size):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }
    return observation, img


def process_action(action, model_family):
    action = normalize_gripper_action(action, binarize=True)
    if model_family == "openvla":
        action = invert_gripper_action(action)
    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
):
    env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    if cfg.num_open_loop_steps != NUM_ACTIONS_CHUNK:
        print(
            f"WARNING: cfg.num_open_loop_steps ({cfg.num_open_loop_steps}) does not match "
            f"NUM_ACTIONS_CHUNK {NUM_ACTIONS_CHUNK}."
        )
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            if len(action_queue) == 0:
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                    use_minivlm=cfg.use_minivlm,
                )
                action_queue.extend(actions)

            action = process_action(action_queue.popleft(), cfg.model_family)
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    save_version=None,
):
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)

        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        if cfg.save_videos:
            save_rollout_video(
                replay_images,
                total_episodes,
                success=success,
                task_description=task_description,
                log_file=log_file,
                save_version=save_version,
            )

        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    env.close()
    del env

    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero_pro(cfg: GenerateConfig) -> float:
    """Evaluate on LIBERO-Pro (perturbed) suites."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    # Resolve perturbed suite before model init so logs/unnorm stay consistent.
    evaluation_cfg = load_evaluation_config(cfg)
    base_suite = get_base_task_suite(cfg.task_suite_name)
    cfg.unnorm_key = base_suite
    perturbed_suite = setup_libero_pro_suite(cfg, evaluation_cfg)
    cfg.task_suite_name = perturbed_suite

    if perturbed_suite not in TASK_MAX_STEPS:
        TASK_MAX_STEPS[perturbed_suite] = _BASE_MAX_STEPS[base_suite]

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    if cfg.task_suite_name not in benchmark_dict:
        raise KeyError(
            f"Suite '{cfg.task_suite_name}' not registered in LIBERO-Pro benchmark. "
            f"Available (sample): {list(benchmark_dict.keys())[:20]} ..."
        )
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    task_ids = list(range(cfg.start_task_id, num_tasks))
    if cfg.max_tasks > 0:
        task_ids = task_ids[: cfg.max_tasks]
    if not task_ids:
        raise RuntimeError("No tasks selected. Check --max_tasks / --start_task_id.")

    log_message(f"Benchmark: LIBERO-Pro", log_file)
    log_message(f"Base suite: {base_suite}", log_file)
    log_message(f"Perturbed suite: {cfg.task_suite_name}", log_file)
    log_message(f"unnorm_key: {cfg.unnorm_key}", log_file)
    log_message(f"Adapter checkpoint: {cfg.pretrained_checkpoint}", log_file)
    log_message(f"Evaluating {len(task_ids)} task(s) (suite has {num_tasks})", log_file)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            cfg.save_version,
        )

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    if cfg.use_wandb:
        wandb.log({"success_rate/total": final_success_rate, "num_episodes/total": total_episodes})
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()
    return final_success_rate


if __name__ == "__main__":
    eval_libero_pro()
