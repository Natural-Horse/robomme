# SmolVLA-style TokenDrop / FrameSamp Rebuild

更新时间：2026-06-09 10:10 CST

## 1. RoboMME 处理方式

RoboMME / MME-VLA 的 perceptual memory 不是直接拼历史 RGB 帧，而是：

```text
history RGB
  -> resize / normalize
  -> SigLIP visual tokens
  -> pooling to 8x8 or 4x4
  -> time-space position embedding
  -> fixed budget memory tokens
  -> context / modulation / expert integration
```

FrameSamp:

- `budget=512`
- `token_per_image=16`，即每帧 4x4 token
- 最多保留 `512 / (16 * num_views)` 帧，按全历史均匀采样

TokenDrop:

- `budget=512`
- `token_per_image=64`，即每帧 8x8 token
- 初始帧强制高分保留
- 后续按像素 patch 变化量打分
- 候选池典型值 `token_drop_keptsize=2048`
- stride 典型为 `streaming_obs_horizon / 2 = 8`

## 2. LIBERO 与 SmolVLA 后训练数据差异

官方 SmolVLA 文档指向的后训练/示例数据是 `lerobot/svla_so100_pickplace`：

- 50 episodes
- 5 个 cube position，每个 position 10 episodes
- Hugging Face dataset viewer 显示约 19.6k rows
- SO100 real/sim style pick-place 数据，state/action 维度为 6

当前 LIBERO IPEC v30:

- Goal / Spatial / Object 三个 suite
- 共约 1314 episodes、171996 effective frames
- simulation manipulation tasks
- action dim 7
- 当前训练用 `*-mem-k4`：4 个历史帧 + 1 当前帧

关键差异：

- SmolVLA 数据更像单域小规模后训练；LIBERO 是多 suite 大规模仿真数据。
- RoboMME/SmolVLA 风格的 512-token memory 在 LIBERO `mem-k4` 上不一定填满。
- FrameSamp 在 LIBERO `mem-k4` 下实际 token 数为 `4 history * 2 cameras * 16 = 128`。
- TokenDrop 在 LIBERO `mem-k4` 下实际 token 数为 `4 history * 2 cameras * 64 = 512`。

## 3. VLA-scratch 改造

本地镜像修改：

```text
remote_edit/bridge.py
remote_edit/policy.py
remote_edit/config.py
```

远端同步到：

```text
/root/autodl-tmp/res/VLA-scratch/vla_scratch/policies/modules/vlm_bridge/smolvlm/bridge.py
/root/autodl-tmp/res/VLA-scratch/vla_scratch/policies/modules/vlm_bridge/base.py
/root/autodl-tmp/res/VLA-scratch/vla_scratch/policies/pi/policy.py
/root/autodl-tmp/res/VLA-scratch/vla_scratch/policies/pi/config.py
```

新增能力：

- `vision_memory_token_per_image`
- `vision_memory_add_pos_emb`
- `vlm_layer_selection = first | last`
- SmolVLA-style policy:
  - `pi-smol-vismem-layerwise-k5`
  - `pi-smol-tokendrop-layerwise-k5`

新配置默认：

```text
vlm_layer_selection=first
policy.action_expert_cfg.only_attend_to_final_layer=false
```

这使 DiT cross-attention 不再所有层读最终 VLM 层，而是按 action expert cross-attention 层逐层读取 VLM hidden state。

## 4. 训练启动状态

启动脚本：

```text
/root/autodl-tmp/res/install_logs/scripts/launch_smolvla_memory_per_suite_20260609.sh
```

监控脚本：

```text
/root/autodl-tmp/res/install_logs/scripts/monitor_smolvla_memory_0609.sh
```

运行 ID：

```text
smolvla_memory_layerwise_0609
```

训练顺序：

```text
framesamp: goal -> spatial -> object
tokendrop: goal -> spatial -> object
```

当前已启动：

```text
framesamp/goal
```

启动参数要点：

```text
batch_size=16
epochs=50
save_interval=50
action_horizon=30
freeze_vlm=true
num_obs_registers=0
expert_only_use_register=false
only_attend_to_final_layer=false
```

初始运行状态：

- 远端 py_compile 与 Hydra config import 通过。
- `framesamp/goal` 已过模型加载和前 200 step。
- GPU 使用约 5.6GB。
- 训练日志显示 `memory/train.tokens=128`，符合 LIBERO `mem-k4` + FrameSamp 4x4 token 的预期。
