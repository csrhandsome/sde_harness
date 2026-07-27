# VLA-Adapter — LIBERO 评测脚本

本仓库通过 `vla-scripts/` 下的 shell 脚本完成 **权重下载**、**标准 LIBERO / LIBERO-plus / LIBERO-Pro 评测**，以及 **Harness VLA（sde_harness）** 评测。

> **命名注意**
> - `*-Pro` checkpoint（如 `LIBERO-Spatial-Pro`）= **VLA-Adapter Pro 模型权重**，与 LIBERO-Pro 评测集无关。
> - **LIBERO-Pro** = 泛化评测基准（env / object / language / task / swap 扰动）。
> - **LIBERO-plus** = 鲁棒性评测基准（camera / robot / language / light / bg / noise / layout）。

## 脚本一览

| 脚本 | 作用 |
|------|------|
| `download_libero_pro_ckpts.sh` | 下载 4 个 suite 对应的 Pro 权重到 `outputs/` |
| `download_libero_plus.sh` | 下载并安装 LIBERO-plus assets |
| `download_libero_pro.sh` | 下载并安装 LIBERO-Pro 预构建 BDDL/init suites |
| `run_libero_eval.sh` | 标准 LIBERO 评测 |
| `run_libero_plus_eval.sh` | LIBERO-plus 评测 |
| `run_libero_pro_eval.sh` | LIBERO-Pro 评测 |
| `run_libero_eval_harness.sh` | 标准 LIBERO + Harness VLA |
| `run_libero_plus_eval_harness.sh` | LIBERO-plus + Harness VLA |
| `run_libero_pro_eval_harness.sh` | LIBERO-Pro + Harness VLA |

## Suite ↔ Checkpoint 映射

所有评测脚本默认按 suite 自动选择权重（可用 `ADAPTER_CKPT` 覆盖）：

| `TASK_SUITE` | Checkpoint |
|--------------|------------|
| `libero_spatial` | `outputs/LIBERO-Spatial-Pro` |
| `libero_object` | `outputs/LIBERO-Object-Pro` |
| `libero_goal` | `outputs/LIBERO-Goal-Pro` |
| `libero_10` | `outputs/LIBERO-Long-Pro` |

Harness / Pro 脚本也支持带扰动后缀的 suite（如 `libero_spatial_swap`），会按 base suite 解析 checkpoint。

---

## 1. 下载权重

```bash
bash vla-scripts/download_libero_pro_ckpts.sh

# 只下某一个 suite
ONLY=libero_object bash vla-scripts/download_libero_pro_ckpts.sh

# 强制重下
FORCE=1 bash vla-scripts/download_libero_pro_ckpts.sh

# 不用镜像时
HF_ENDPOINT=https://huggingface.co bash vla-scripts/download_libero_pro_ckpts.sh
```

默认从 `https://hf-mirror.com` 拉取，写入 `outputs/LIBERO-*-Pro/`。

---

## 2. 标准 LIBERO 评测

依赖仓库内 `./LIBERO`，默认每个任务 20 trials。

```bash
# 默认：libero_spatial
bash vla-scripts/run_libero_eval.sh

TASK_SUITE=libero_object bash vla-scripts/run_libero_eval.sh
TASK_SUITE=all bash vla-scripts/run_libero_eval.sh
ADAPTER_CKPT=outputs/my-ckpt TASK_SUITE=libero_spatial bash vla-scripts/run_libero_eval.sh
```

常用环境变量：`TASK_SUITE`、`NUM_TRIALS`、`CUDA_VISIBLE_DEVICES`、`ADAPTER_CKPT`。

`TASK_SUITE=all` 时不能设置 `ADAPTER_CKPT`（每个 suite 需要各自权重）。

---

## 3. LIBERO-plus 评测

先准备 assets（需已有 `third_party/LIBERO-plus`）：

```bash
bash vla-scripts/download_libero_plus.sh
```

再评测（官方 Plus 协议默认 `NUM_TRIALS=1`）：

```bash
# 冒烟
MAX_TASKS=2 bash vla-scripts/run_libero_plus_eval.sh

# 指定扰动维度
PERTURBATION_CATEGORY="Camera Viewpoints" MAX_TASKS=10 bash vla-scripts/run_libero_plus_eval.sh

TASK_SUITE=libero_object bash vla-scripts/run_libero_plus_eval.sh
TASK_SUITE=all bash vla-scripts/run_libero_plus_eval.sh
```

常用环境变量：`TASK_SUITE`、`NUM_TRIALS`、`PERTURBATION_CATEGORY`、`MAX_TASKS`、`START_TASK_ID`、`SAVE_VIDEOS`、`ADAPTER_CKPT`、`CUDA_VISIBLE_DEVICES`。

---

## 4. LIBERO-Pro 评测

先下载预构建 suites（需已有 `third_party/LIBERO-PRO`）：

```bash
bash vla-scripts/download_libero_pro.sh
```

再评测：

```bash
# 冒烟
MAX_TASKS=1 NUM_TRIALS=1 bash vla-scripts/run_libero_pro_eval.sh

TASK_SUITE=libero_object bash vla-scripts/run_libero_pro_eval.sh
TASK_SUITE=all bash vla-scripts/run_libero_pro_eval.sh
```

扰动类型在配置文件中切换：

```text
experiments/robot/libero/libero_pro_evaluation_config.yaml
```

（`use_environment` / `use_swap` / `use_object` / `use_language` / `use_task`）

可用 `EVAL_CFG` 指定其它配置路径。常用环境变量：`TASK_SUITE`、`NUM_TRIALS`、`MAX_TASKS`、`START_TASK_ID`、`SAVE_VIDEOS`、`ADAPTER_CKPT`、`CUDA_VISIBLE_DEVICES`。

---

## 5. Harness VLA（sde_harness）

在对应 benchmark 上跑 agent harness。入口与上面三套 eval 对齐，默认 `PLANNER=cursor`、`MODEL=composer-2.5`（需 `sde_harness/.env.local` 或 `CURSOR_API_KEY`）。

Harness 已并入仓库根目录 uv workspace，与 VLA-Adapter 共用同一个 `.venv`：

```bash
# 在仓库根目录一次即可（同时装 vla-adapter + sde-harness）
uv sync
```

### 标准 LIBERO

```bash
bash vla-scripts/run_libero_eval_harness.sh
TASK_SUITE=libero_object TASK=0 bash vla-scripts/run_libero_eval_harness.sh
ADAPTER_CKPT=outputs/my-ckpt TASK_SUITE=libero_spatial bash vla-scripts/run_libero_eval_harness.sh
```

### LIBERO-plus

需先 `download_libero_plus.sh`：

```bash
bash vla-scripts/run_libero_plus_eval_harness.sh
TASK_SUITE=libero_object TASK=0 bash vla-scripts/run_libero_plus_eval_harness.sh
```

### LIBERO-Pro

需先 `download_libero_pro.sh`。默认 suite 为 `libero_spatial_swap`：

```bash
bash vla-scripts/run_libero_pro_eval_harness.sh
TASK_SUITE=libero_object_swap TASK=0 bash vla-scripts/run_libero_pro_eval_harness.sh
TASK_SUITE=libero_spatial_env bash vla-scripts/run_libero_pro_eval_harness.sh
```

Harness 常用环境变量：`TASK_SUITE`、`TASK`、`SEED`、`PLANNER`、`MODEL`、`ADAPTER_CKPT`、`CUDA_VISIBLE_DEVICES`。额外 CLI 参数可直接追加到脚本后（透传给 `python -m rpent.cli.main`）。

更多 Harness 安装与 API key 配置见 [`sde_harness/README.md`](sde_harness/README.md)。

---

## 推荐流程

```bash
# 1) 权重
bash vla-scripts/download_libero_pro_ckpts.sh

# 2a) 标准 LIBERO
bash vla-scripts/run_libero_eval.sh

# 2b) LIBERO-plus
bash vla-scripts/download_libero_plus.sh
bash vla-scripts/run_libero_plus_eval.sh

# 2c) LIBERO-Pro
bash vla-scripts/download_libero_pro.sh
bash vla-scripts/run_libero_pro_eval.sh

# 3) 需要 agent harness 时，换用对应 *_harness.sh
```
