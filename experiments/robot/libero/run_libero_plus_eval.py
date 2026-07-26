"""
run_libero_plus_eval.py

Evaluate a trained policy on the LIBERO-plus robustness benchmark
(perturbed LIBERO suites with ~10k tasks across 7 dimensions).

Official protocol: num_trials_per_task=1 (each task is itself a perturbation).
Uses existing VLA-Adapter / LIBERO finetuned weights; no Plus mix-SFT required
for evaluation.
"""

# NOTE: Do not enable `from __future__ import annotations` here.
# draccus.wrap() needs runtime dataclass type hints (same as run_libero_eval.py).

import json
import logging
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import yaml

# ---------------------------------------------------------------------------
# Point imports at third_party/LIBERO-plus before importing `libero`.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
LIBERO_PLUS_ROOT = REPO_ROOT / "third_party" / "LIBERO-plus"
LIBERO_PLUS_PKG = LIBERO_PLUS_ROOT / "libero" / "libero"

if not LIBERO_PLUS_ROOT.is_dir():
    raise FileNotFoundError(
        f"LIBERO-plus package not found at {LIBERO_PLUS_ROOT}. "
        "Clone https://github.com/sylvestf/LIBERO-plus into third_party/LIBERO-plus"
    )

_LIBERO_PLUS_CONFIG_DIR = REPO_ROOT / "experiments" / "robot" / "libero" / ".libero_plus_config"
_LIBERO_PLUS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
_LIBERO_PLUS_CONFIG_FILE = _LIBERO_PLUS_CONFIG_DIR / "config.yaml"
_DEFAULT_PLUS_PATHS = {
    "benchmark_root": str(LIBERO_PLUS_PKG),
    "bddl_files": str(LIBERO_PLUS_PKG / "bddl_files"),
    "init_states": str(LIBERO_PLUS_PKG / "init_files"),
    "datasets": str(LIBERO_PLUS_ROOT / "libero" / "datasets"),
    "assets": str(LIBERO_PLUS_PKG / "assets"),
}
# Always rewrite so a previous LIBERO / LIBERO-Pro ~/.libero config cannot leak in.
with open(_LIBERO_PLUS_CONFIG_FILE, "w", encoding="utf-8") as f:
    yaml.dump(_DEFAULT_PLUS_PATHS, f)

os.environ["LIBERO_CONFIG_PATH"] = str(_LIBERO_PLUS_CONFIG_DIR)
# Prefer LIBERO-plus over the original ./LIBERO package.
sys.path = [p for p in sys.path if Path(p).resolve() != (REPO_ROOT / "LIBERO").resolve()]
sys.path.insert(0, str(LIBERO_PLUS_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from libero.libero import benchmark  # noqa: E402

import wandb  # noqa: E402

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


class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}

TASK_CLASSIFICATION_PATH = LIBERO_PLUS_PKG / "benchmark" / "task_classification.json"

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

    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_steps_wait: int = 10
    # Official LIBERO-plus protocol uses 1 trial per (already-perturbed) task.
    num_trials_per_task: int = 1
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    # Optional filters for smoke / subset eval (empty = full suite).
    # category examples: "Camera Viewpoints", "Robot Initial States", "Language Instructions",
    # "Light Conditions", "Background Textures", "Sensor Noise", "Objects Layout"
    perturbation_category: str = ""
    max_tasks: int = 0  # 0 = all matching tasks
    start_task_id: int = 0

    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_entity: str = "your-wandb-entity"
    wandb_project: str = "your-wandb-project"
    seed: int = 7

    save_version: str = "vla-adapter-libero-plus"
    save_videos: bool = False  # rollout MP4s off by default; logs still written
    use_pro_version: bool = True  # VLA-Adapter Pro *model*; unrelated to LIBERO-plus benchmark
    phase: str = "Inference"
    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"
    assets_dir = Path(_DEFAULT_PLUS_PATHS["assets"])
    if not assets_dir.is_dir():
        raise FileNotFoundError(
            f"LIBERO-plus assets not found at {assets_dir}.\n"
            "Download via mirror (4 workers):\n"
            "  bash vla-scripts/download_libero_plus.sh"
        )


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
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    run_id = f"EVAL-LIBERO-PLUS-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.perturbation_category:
        run_id += f"--{cfg.perturbation_category.replace(' ', '_')}"
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


def load_task_classification(suite_name: str) -> dict:
    """Map task name -> {category, difficulty_level, id}."""
    if not TASK_CLASSIFICATION_PATH.is_file():
        return {}
    with open(TASK_CLASSIFICATION_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for item in data.get(suite_name, []):
        mapping[item["name"]] = item
    return mapping


def select_task_ids(cfg: GenerateConfig, task_suite, classification: dict) -> list:
    """Select task indices to evaluate (optionally filtered by category / max_tasks)."""
    n_tasks = task_suite.n_tasks
    selected = []
    for task_id in range(cfg.start_task_id, n_tasks):
        task = task_suite.get_task(task_id)
        meta = classification.get(task.name, {})
        category = meta.get("category", "Unknown")
        if cfg.perturbation_category and category != cfg.perturbation_category:
            continue
        selected.append(task_id)
        if cfg.max_tasks > 0 and len(selected) >= cfg.max_tasks:
            break
    return selected


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
            f"NUM_ACTIONS_CHUNK ({NUM_ACTIONS_CHUNK})."
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
    category: str = "Unknown",
):
    task = task_suite.get_task(task_id)
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask[{task_id}] ({category}): {task_description}", log_file)
        log_message(f"Task name: {task.name}", log_file)

        if cfg.initial_states_path == "DEFAULT":
            # Plus tasks typically have a single init state; clamp for safety.
            initial_state = initial_states[min(episode_idx, len(initial_states) - 1)]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(
                    f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!",
                    log_file,
                )
                continue
            initial_state = np.array(
                all_initial_states[initial_states_task_key][episode_key]["initial_state"]
            )

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
        log_message(
            f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)",
            log_file,
        )

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

    return total_episodes, total_successes, task_successes, task_episodes


@draccus.wrap()
def eval_libero_plus(cfg: GenerateConfig) -> float:
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    if cfg.task_suite_name not in benchmark_dict:
        raise KeyError(
            f"Suite '{cfg.task_suite_name}' not registered in LIBERO-plus benchmark. "
            f"Available: {sorted(benchmark_dict.keys())}"
        )
    task_suite = benchmark_dict[cfg.task_suite_name]()
    classification = load_task_classification(cfg.task_suite_name)
    task_ids = select_task_ids(cfg, task_suite, classification)

    log_message("Benchmark: LIBERO-plus", log_file)
    log_message(f"Task suite: {cfg.task_suite_name} ({task_suite.n_tasks} total tasks)", log_file)
    if cfg.perturbation_category:
        log_message(f"Category filter: {cfg.perturbation_category}", log_file)
    log_message(f"Evaluating {len(task_ids)} task(s)", log_file)
    if not task_ids:
        raise RuntimeError("No tasks selected. Check --perturbation_category / --max_tasks / --start_task_id.")

    total_episodes, total_successes = 0, 0
    category_stats = defaultdict(lambda: {"successes": 0, "episodes": 0})

    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        category = classification.get(task.name, {}).get("category", "Unknown")
        total_episodes, total_successes, task_successes, task_episodes = run_task(
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
            category=category,
        )
        category_stats[category]["successes"] += task_successes
        category_stats[category]["episodes"] += task_episodes

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(
        f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)",
        log_file,
    )
    log_message("Per-category success rates:", log_file)
    for category, stats in sorted(category_stats.items()):
        rate = float(stats["successes"]) / float(stats["episodes"]) if stats["episodes"] else 0.0
        log_message(
            f"  {category}: {rate:.4f} ({stats['successes']}/{stats['episodes']})",
            log_file,
        )

    if cfg.use_wandb:
        wandb.log({"success_rate/total": final_success_rate, "num_episodes/total": total_episodes})
        for category, stats in category_stats.items():
            rate = float(stats["successes"]) / float(stats["episodes"]) if stats["episodes"] else 0.0
            wandb.log({f"success_rate/{category}": rate, f"num_episodes/{category}": stats["episodes"]})
        wandb.save(local_log_filepath)

    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero_plus()
