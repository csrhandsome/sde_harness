# sde_harness

从 [RPent](https://github.com/RLinf/RPent)（Harness VLA）抽出的 agent harness，接到本仓库的 **VLA-Adapter** 权重与本地 **LIBERO-PRO / LIBERO-plus** 环境上。

原项目用 RLinf 的 Pi0.5 checkpoint + `rlinf.envs.libero.LiberoEnv`。这里已去掉 RLinf 依赖：

| 组件 | 原 RPent | 本目录 |
|------|----------|--------|
| 规划器 / toolkit / RPC / dashboard | `rpent/` | 保留 |
| 冻结 VLA | RLinf Pi0.5 | `robots/libero/vla_server.py` → VLA-Adapter |
| 仿真 env | `rlinf.envs.libero.LiberoEnv` | `robots/libero/env_server.py` → 本地 LIBERO-* |

## 目录

```
sde_harness/
  rpent/                 # harness 核心（planner / tools / RPC / dashboard）
  robots/libero/         # LIBERO 侧 env/vla/sam3 + toolkit/prompts
  scripts/codex_proxy/   # Codex 代理（可选）
```

入口脚本在仓库根目录 `vla-scripts/`（与现有 eval 脚本对齐）：

- `vla-scripts/run_libero_eval_harness.sh`
- `vla-scripts/run_libero_pro_eval_harness.sh`
- `vla-scripts/run_libero_plus_eval_harness.sh`

## 安装

与仓库根目录共用同一个 uv 环境（workspace member）：

```bash
# 在仓库根目录
uv sync
```

会可编辑安装 `vla-adapter` + `sde-harness`（planner / robosuite 等）到根目录 `.venv`。

SAM3（`sam3`）与 VLA-Adapter 钉死的 `timm==0.9.10` 冲突，未放进默认依赖；需要 segment 时另开环境或自行处理版本。

默认 planner 为 Cursor（`CURSOR_API_KEY` / `.env.local`）。Claude Code / API / Codex / CodeBuddy 仍可用，按需覆盖 `PLANNER`/`MODEL`。

## 环境变量

```bash
# 必需：VLA-Adapter checkpoint（替代原来的 PI05_CHECKPOINT_PATH）
export ADAPTER_CHECKPOINT_PATH=/path/to/outputs/LIBERO-Spatial-Pro

# LIBERO 变体：pro | plus | standard（默认 pro）
export LIBERO_TYPE=pro

# 可选：SAM3（未设置则跳过 segment；8GB 卡建议不加）
# export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt

# Planner（按后端选其一）
export ANTHROPIC_API_KEY=sk-xxx          # api / claude_code
# export ANTHROPIC_BASE_URL=https://...
# 或写入 gitignored 的本地文件（CLI / cursor / codebuddy planner 会自动加载）：
#   cp .env.local.example .env.local
#   # 编辑 CURSOR_API_KEY=crsr_... 或 CODEBUDDY_API_KEY=...
export CURSOR_API_KEY=crsr_...           # --planner cursor（也可用 .env.local）
# export CURSOR_MODEL=auto               # 或 composer-2.5
export CODEBUDDY_API_KEY=...             # --planner codebuddy（也可用 .env.local）
# 国内版需额外：export CODEBUDDY_INTERNET_ENVIRONMENT=internal
# export CODEBUDDY_MODEL=deepseek-v3.1
```

`unnorm_key` 默认从 `--suite` 推导（如 `libero_spatial_swap` → `libero_spatial`）。

## 运行

```bash
# Pro：libero_spatial_swap，task 0，seed 0
bash vla-scripts/run_libero_pro_eval_harness.sh

# 或覆盖 suite / planner：
TASK_SUITE=libero_spatial_swap TASK=0 SEED=0 \
PLANNER=claude_code MODEL=claude-opus-4-8 \
  bash vla-scripts/run_libero_pro_eval_harness.sh

# Cursor SDK（local Agent + CustomTool；需 CURSOR_API_KEY 或 .env.local）：
# 默认 composer-2.5 + fast=true（少思考、更快）；勿用 auto（可能挑到会 thinking 的模型）
PLANNER=cursor MODEL=composer-2.5 \
  bash vla-scripts/run_libero_pro_eval_harness.sh

# CodeBuddy Agent SDK（需 CODEBUDDY_API_KEY 或已登录 codebuddy CLI）：
PLANNER=codebuddy MODEL=deepseek-v3.1 \
  bash vla-scripts/run_libero_pro_eval_harness.sh

# Plus / standard：
bash vla-scripts/run_libero_plus_eval_harness.sh
bash vla-scripts/run_libero_eval_harness.sh

# 或直接（需已在仓库根目录 uv sync）：
cd /path/to/VLA-Adapter
export PYTHONPATH="$(pwd)/sde_harness:${PYTHONPATH:-}"
export ADAPTER_CHECKPOINT_PATH=outputs/LIBERO-Spatial-Pro
export LIBERO_TYPE=pro
uv run sde-harness --env libero --suite libero_spatial_swap --task 0 --seed 0 \
  --adapter-checkpoint "$ADAPTER_CHECKPOINT_PATH" \
  --planner cursor --model composer-2.5
```

常用 flag：

- `--libero-type {pro,plus,standard}`
- `--adapter-checkpoint` / `--unnorm-key` / `--use-pro-version` / `--no-use-pro-version`
- `--planner {api,claude_code,codex,cursor,codebuddy}` / `--model`
- `--vla-endpoint` / `--env-endpoint` / `--sam3-endpoint`：复用已启动的服务
- `--dashboard` / `--interactive`

Cursor planner 说明：

- 包：`cursor-sdk`（已写入 `pyproject.toml`）
- 鉴权：`CURSOR_API_KEY`，或 `sde_harness/.env.local`（见 `.env.local.example`；已 gitignore）
- 申请：https://cursor.com/dashboard/api → New API Key
- 模型：默认 `composer-2.5` 并自动加 `fast=true`；也可用
  `MODEL='composer-2.5,fast=true'` / `CURSOR_MODEL_PARAMS=fast=true`
  （`auto` 可跑但可能更慢，因为会自选带 thinking 的模型）
- 工具桥：`LocalAgentOptions.custom_tools`（in-process，不经 HTTP MCP）

CodeBuddy planner 说明：

- 包：`codebuddy-agent-sdk`（已写入 `pyproject.toml`）
- 鉴权：`CODEBUDDY_API_KEY`，或已有 `codebuddy` CLI 登录态；也可写在 `.env.local`
- 国际版 Key：https://www.codebuddy.ai/profile/keys
- 国内版 Key：https://copilot.tencent.com/profile/ ，并设 `CODEBUDDY_INTERNET_ENVIRONMENT=internal`
- 模型：默认 `deepseek-v3.1`（可用 `CODEBUDDY_MODEL` / `--model` 覆盖）
- 工具桥：`create_sdk_mcp_server`（in-process MCP，与 Claude Code planner 同构）
- 本地 OpenAI compatible（如 `serving/` 的 vLLM）：设 `CODEBUDDY_OPENAI_BASE_URL`
  （planner 会写入 gitignored 的 `.codebuddy/models.json` 并加载 project settings）

```bash
# 终端 1：启动 serving（需 GPU / 权重）
bash serving/scripts/serve.sh --model hy-embodied-vlm-1.0

# 终端 2：CodeBuddy → 本地 /v1/chat/completions（served_name=hy_a3b）
export CODEBUDDY_OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export CODEBUDDY_API_KEY=EMPTY
PLANNER=codebuddy MODEL=hy_a3b \
  bash vla-scripts/run_libero_pro_eval_harness.sh
```

注意：`hy_a3b` 的 tool-call 解析器（`hy_v3`）与 CodeBuddy 标准 OpenAI tool-call 是否完全兼容需上线后验证；该接线主要用于把 Agent SDK 指到自建端点。

## 流程（与 RPent 相同）

1. Agent（LLM planner）调工具（`pi0_pick` / `move_to` / `finish` …）
2. Toolkit → `vla_server.predict`（VLA-Adapter 推理出 action chunk）
3. `env_server.chunk_step` 在 LIBERO 里执行
4. 渲染图像 + 状态写回 LLM，直到 `finish` 或步数上限

工具名里仍保留 `pi0_*`（与原 harness prompt 对齐）；底层策略已是 VLA-Adapter。
