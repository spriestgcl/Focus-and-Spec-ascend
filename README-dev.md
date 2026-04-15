# Focus-and-Spec-ascend 开发说明

本仓库采用 **monorepo** 方式管理两个代码目录：

- `vllm-ascend/`：基于官方 `vllm-ascend v0.9.1` 快照纳入
- `verl/`：基于当前可用的 Ascend/NPU 版 VERL 代码

目标是：在同一个仓库里协同修改推理框架（vLLM Ascend）与训练框架（VERL），支持 Ascend/NPU 上的 RL 训练加速研发。

## 1) 新机器初始化（推荐流程）

下面这套流程是当前机器已经验证通过的一套可用组合。

```bash
git clone <你的仓库地址> Focus-and-Spec-ascend
cd Focus-and-Spec-ascend

# 先把 vllm-ascend 固定到兼容版本
cd vllm-ascend
git checkout v0.9.1
cd ..

# 建议使用独立 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n verl-ascend python=3.10 -y
conda activate verl-ascend

# 加载 Ascend 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -m pip install -U pip setuptools wheel

# 兼容版本组合
python -m pip install \
  "torch==2.5.1" \
  "torch-npu==2.5.1.post1" \
  "torchvision==0.20.1" \
  "transformers==4.52.4" \
  "numpy==1.26.4" \
  "vllm==0.11.0"

# 先安装 vllm-ascend（会编译扩展）
export COMPILE_CUSTOM_KERNELS=1
python -m pip install --no-build-isolation -e ./vllm-ascend

# 再安装 verl，避免覆盖前面的依赖组合
python -m pip install tensorboard
python -m pip install --no-deps -e ./verl
```

## 2) 快速自检

先确认设备可见：

```bash
npu-smi info
```

再确认 Python 导入和 NPU 可见性：

```bash
python - <<'PY'
import torch
import torch_npu
import vllm
import vllm_ascend
import verl

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("vllm:", getattr(vllm, "__version__", "unknown"))
print("vllm_ascend:", getattr(vllm_ascend, "__version__", "unknown"))
print("verl:", getattr(verl, "__version__", "unknown"))
print("npu available:", torch_npu.npu.is_available())
print("npu count:", torch_npu.npu.device_count())
PY
```

若 `import` 正常且 `npu available: True`，说明本地可开发。

## 3) 训练脚本

当前仓库内的最小测试脚本：

```bash
cd /root/Focus-and-Spec-ascend/verl
bash wyj_test/run_grpo.sh
```

如需覆盖路径，可这样运行：

```bash
TRAIN_FILE=/home/deepmath/train.parquet \
TEST_FILE=/home/deepmath/test.parquet \
MODEL_PATH=/mnt/model/gcl/Qwen2.5-7B-Instruct \
LOG_FILE=grpo_npu.log \
bash wyj_test/run_grpo.sh trainer.total_epochs=1
```

## 4) 日常开发规范

- 修改 `vllm-ascend` 相关功能时，仅改 `vllm-ascend/` 下文件。
- 修改 `verl` 相关功能时，仅改 `verl/` 下文件。
- 功能分支建议命名：`feat/vllm-ascend-xxx`、`feat/verl-xxx`、`feat/rl-accel-xxx`。
- 提交信息建议带作用域，例如：`vllm-ascend: optimize xxx`。

## 5) 上游同步建议（保持可维护）

当前仓库不是 submodule，而是“快照式”纳入上游代码。建议每次同步都走独立 PR：

1. 新建分支：`chore/sync-vllm-ascend-<tag>` 或 `chore/sync-verl-<tag>`
2. 用目标上游 tag 的代码覆盖对应目录
3. 运行安装与最小自检
4. 合并后打标记，例如：`sync-vllm-ascend-0.9.1`

这样可以把“上游同步”和“你自己的加速改动”分开，后续定位回归会更清晰。

## 6) 常见问题

- 不要直接执行 `pip install -r ./verl/requirements-npu.txt`，这会覆盖已经对齐好的 `vllm-ascend` 依赖组合。
- `vllm-ascend/` 目录请固定在 `v0.9.1`，不要直接用当前上游较新的 `0.13.x` 分支搭这份 `verl`。
- `pip install -e ./vllm-ascend` 需要 Ascend 环境与编译依赖；安装前建议先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。
- 如果 `python import` 正常但训练时报设备错误，先检查 `npu-smi info`、`ASCEND_HOME_PATH`、`LD_LIBRARY_PATH`、以及容器内 `/dev/davinci*` 是否可见。
- `vllm_ascend/_version.py` 是构建生成文件，不需要手动提交。

## 7) NPU 可复用最终配置快照（2026-04-15）

下面这套是今天在本机反复调试后，已经验证能进入稳定训练循环的配置（非 Docker，本地 conda 环境）。

### 7.1 运行环境

- conda 环境：`verl-ascend-local`（建议与你当前验证一致）
- Ascend 环境：`source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- 工作目录：`/root/Focus-and-Spec-ascend/verl`
- 设备绑定：`ASCEND_RT_VISIBLE_DEVICES=12,13`（按你机器空闲卡位替换）

### 7.2 一键启动命令

```bash
bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh
conda activate verl-ascend-local
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/Focus-and-Spec-ascend/verl
ray stop --force || true
LOG_FILE=codex_grpo_retry36.log bash wyj_test/run_grpo.sh
'
```

### 7.3 当前稳定参数（run_grpo.sh 默认值）

- `TORCHDYNAMO_DISABLE=1`
- `VERL_ENABLE_VLLM_LORA_HIJACK=1`
- `FAST_SMOKE_RUN=1`（调试模式，默认采样 64/64）
- `USE_LORA=1`，`LORA_RANK=16`，`LORA_ALPHA=32`
- `ROLLOUT_LOAD_FORMAT=safetensors`
- `ROLLOUT_LAYERED_SUMMON=True`
- `ROLLOUT_GPU_MEMORY_UTILIZATION=0.45`
- `MAX_PROMPT_LENGTH=1024`
- `MAX_RESPONSE_LENGTH=4096`
- `TRAIN_BATCH_SIZE=2`，`PPO_MINI_BATCH_SIZE=2`，`PPO_MICRO_BATCH_SIZE_PER_GPU=1`

说明：这是一份“优先稳定跑通”的配置，不是吞吐最优配置。

### 7.4 已验证结果

- 训练已跨过初始化阶段，进入 `Training Progress` 主循环。
- 已连续通过 `step 1 -> step 4`，并进入 `step 5`。
- 关键日志文件：`/root/Focus-and-Spec-ascend/verl/codex_grpo_retry36.log`

### 7.5 今天完成的关键修复（总结）

1. 修复/绕过 vllm-ascend sleep/camem 兼容问题，避免初始化即崩溃。  
2. 在 VERL rollout 的 `release()` 路径加保护，绕过非 sleep 模式下断言。  
3. 在 `run_grpo.sh` 增加 preflight 快速失败检查，避免长时间等待后才报 import 兼容错误。  
4. 在 Ray runtime_env 显式透传 NPU 设备与关键环境变量（尤其 `ASCEND_RT_VISIBLE_DEVICES`），修复 worker 设备错绑导致的 OOM。  
5. 引入 LoRA 调试路径并修复适配器配置缺失，绕开全参优化器 OOM。  
6. 固定 rollout 相关加载策略（`safetensors + layered_summon`）提升 NPU 侧稳定性。  

### 7.6 已知现象（目前不阻塞）

- 日志中仍会出现 `No tokenizer found in /simon-stub-path` 警告（已回退到 base tokenizer，不阻塞训练）。  
- `Failed to register custom ops` 警告仍存在，可能影响性能，但当前可跑通。  

### 7.7 从“调试模式”切到“正式训练”的建议

先只改这几项，逐步放大，避免回归难定位：

1. `FAST_SMOKE_RUN=0`  
2. 逐步增大 `TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE`  
3. 将 `MAX_RESPONSE_LENGTH` 从 `4096` 下调到业务需要值（如 `1024` 或 `2048`）以提升迭代速度  
