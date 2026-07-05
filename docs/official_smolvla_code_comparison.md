# 官方 SmolVLA 代码学习与本地实现对比

更新时间：2026-06-08

## 1. 官方仓库位置

已在 seetacloud 服务器 clone 官方 LeRobot 仓库：

```text
/root/autodl-tmp/res/official_repos/lerobot
```

当前远程仓库：

```text
https://github.com/huggingface/lerobot.git
```

当前 commit：

```text
49755a3
```

SmolVLA 相关代码位置：

```text
src/lerobot/policies/smolvla/
  configuration_smolvla.py
  modeling_smolvla.py
  processor_smolvla.py
  smolvlm_with_expert.py
```

## 2. 官方 SmolVLA 的核心结构

官方 SmolVLA 由两部分组成：

1. SmolVLM backbone：负责图像和语言 prefix。
2. Action expert：负责 noisy action suffix 的 flow matching denoising。

配置在 `configuration_smolvla.py` 中，关键默认值如下：

```text
vlm_model_name = HuggingFaceTB/SmolVLM2-500M-Video-Instruct
chunk_size = 50
n_action_steps = 50
num_steps = 10
use_cache = True
freeze_vision_encoder = True
train_expert_only = True
attention_mode = cross_attn
prefix_length = -1
num_vlm_layers = 16
num_expert_layers = -1
self_attn_every_n_layers = 2
```

其中 `num_expert_layers=-1` 表示 action expert 默认和 VLM 使用相同层数；`num_vlm_layers=16` 表示只使用 SmolVLM text model 的前 16 层。

## 3. 官方逐层融合方式

官方 SmolVLA 的逐层融合在 `smolvlm_with_expert.py` 里实现。

核心点：

1. `SmolVLMWithExpertModel` 同时持有：
   - `self.vlm`
   - `self.lm_expert`
2. `get_model_layers()` 会把 VLM text layers 和 expert layers 对齐成两个 layer list。
3. `forward()` 对 `layer_idx in range(num_vlm_layers)` 逐层循环。
4. 每一层根据配置选择：
   - `forward_attn_layer()`：VLM 和 expert 都走各自 self-attention。
   - `forward_cross_attn_layer()`：expert query attend VLM prefix 的 key/value。
5. `self_attn_every_n_layers=2` 时，部分层保留 expert self-attention，其余层执行 expert-to-VLM cross-attention。

也就是说，官方 SmolVLA 的“逐层融合”不是把 VLM 所有 hidden states 先收集出来，再交给一个外部 DiT；而是 VLM layer 和 action expert layer 在同一个 forward loop 中同步推进。expert 在多层内部直接 cross-attend 到当前/缓存的 VLM prefix 表征。

## 4. 官方训练/推理流程

训练入口在 `modeling_smolvla.py`：

1. `embed_prefix()`：
   - 图像经 SmolVLM vision tower 编码。
   - 语言 token 经 SmolVLM text embedding。
   - prefix 包含 image tokens、language tokens、state token。
2. `embed_suffix()`：
   - noisy action 经 `action_in_proj`。
   - timestep 经过 sinusoidal embedding 和 MLP。
   - action embedding 与 time embedding 拼接后生成 expert suffix tokens。
3. `forward()`：
   - 构造 flow matching 目标：`x_t = t * noise + (1 - t) * actions`，`u_t = noise - actions`。
   - 调用 `vlm_with_expert.forward(inputs_embeds=[prefix_embs, suffix_embs])`。
   - 取 suffix 输出，经 `action_out_proj` 预测 `v_t`。
4. `sample_actions()`：
   - 先用 prefix 计算 KV cache。
   - 每个 denoising step 只送 suffix tokens，并复用 prefix KV cache。

这和我们之前的 VLA-scratch 流程不同：官方 SmolVLA 的 prefix cache 和 expert cross-attention 是 action expert 内部的一等公民。

## 5. 和我们之前 VLA-scratch 写法的差异

我们之前的 VLA-scratch 代码主要在这些文件里：

```text
worklog/remote_edit/bridge.py
worklog/remote_edit/policy.py
worklog/remote_edit/config.py
```

### 5.1 融合结构不同

我们的 VLA-scratch 写法：

1. SmolVLM bridge 先完整跑 VLM。
2. 收集 `hidden_state_list`。
3. `PiPolicy.construct_suffix_input()` 取最后 `action_expert_cfg.num_hidden_layers` 层。
4. 默认 `expert_only_use_register=True`，所以只保留最后几个 observation/register tokens。
5. DiT action expert 通过 `encoder_hidden_states` 做 cross-attention。

官方 SmolVLA 写法：

1. VLM prefix 和 action expert suffix 在同一个 `SmolVLMWithExpertModel.forward()` 里逐层推进。
2. expert 层在内部直接 self-attend 或 cross-attend VLM prefix。
3. 推理时 prefix KV cache 被明确复用。

结论：我们之前的 VLA-scratch 是“VLM 先编码，再把 selected hidden states 给外部 DiT”；官方 SmolVLA 是“VLM 与 expert 逐层耦合运行”。

### 5.2 默认逐层行为不同

我们的 VLA-scratch 默认配置：

```text
only_attend_to_final_layer = True
expert_only_use_register = True
num_obs_registers = 4
```

所以即使 bridge 收集了多层 hidden states，默认 action expert 仍主要看最终 VLM 层的 register 表征。

官方 SmolVLA 默认配置：

```text
attention_mode = cross_attn
num_vlm_layers = 16
self_attn_every_n_layers = 2
```

因此官方代码默认就是逐层 action expert/VLM 交互，并且每隔若干层插入 expert self-attention。

### 5.3 Action expert 类型不同

我们的 VLA-scratch：

```text
DiT action expert
12 layers
hidden size 1024
cross_attention_every = 2
action_horizon = 10
```

官方 SmolVLA：

```text
SmolVLM text model + lm_expert
chunk_size = 50
n_action_steps = 50
num_steps = 10
expert hidden size = VLM hidden size * expert_width_multiplier
```

官方 action expert 更接近“SmolVLM text stack 的并行 expert 分支”，不是独立 DiT。

## 6. 和我们 TokenDrop / FrameSamp 写法的关系

我们写的 TokenDrop / FrameSamp 是 history memory selection 方法，主要在 VLA-scratch 的 SmolVLM bridge 里实现：

```text
_encode_history_as_memory_tokens()
_pixel_patch_change_scores()
_even_indices()
```

FrameSamp：

```text
从历史帧中等间隔选帧，然后保留这些帧的视觉 tokens，再按 max_tokens 截断。
```

TokenDrop：

```text
把历史帧划分为 patch，按 RGB patch 变化量打分；首帧给高优先级；后续帧按 stride 与前一帧比较；最后 top-k 选择 memory tokens。
```

这些方法解决的是“哪些历史视觉 token 进入 memory”。它们没有改变 action expert 和 VLM 的根本耦合方式。

官方 SmolVLA 当前代码重点是 action expert 与 VLM 的逐层交互，并不直接提供我们这套 TokenDrop / FrameSamp memory selection。若要把 TokenDrop / FrameSamp 迁移到官方 SmolVLA，更合理的位置不是外部 DiT，而是：

1. 在 `embed_prefix()` 阶段构造额外 history memory tokens。
2. 把 memory tokens 作为 prefix 的一部分，或作为 expert cross-attention 的额外 KV。
3. 保持 `SmolVLMWithExpertModel.forward()` 的逐层交互结构。

## 7. 和我之前 StarVLA SmolVLM adapter 的差异

我之前本地写过一个未完成的 StarVLA adapter：

```text
starVLA_stage/starVLA/model/modules/vlm/SmolVLM.py
```

这个 adapter 目前只是把 SmolVLM 包成 StarVLA 可调用的 VLM 接口：

1. 加载 `SmolVLMForConditionalGeneration`。
2. 用 `SmolVLMProcessor` 构造 image/text inputs。
3. 返回 HuggingFace 模型输出 hidden states。

它的问题是：

1. 没有注册到 StarVLA VLM registry。
2. 没有实现官方 SmolVLA 的 action expert。
3. 没有实现逐层 expert/VLM cross-attention。
4. 没有处理 prefix KV cache 与 suffix denoising 的官方结构。

结论：这个 StarVLA adapter 只能算“SmolVLM 作为普通 VLM backbone 接入 StarVLA”的开始，不是官方 SmolVLA 复现。

## 8. 对后续重写的建议

如果我们要严肃复现官方 SmolVLA，而不是继续修补 VLA-scratch：

1. 不应再用“VLM 最后一层 hidden state + 外部 DiT”作为主结构。
2. 应该以 `SmolVLMWithExpertModel` 为核心，保留官方逐层 VLM/expert loop。
3. TokenDrop / FrameSamp 应作为 prefix memory token 构造模块接入，而不是替代官方 action expert。
4. 推理时必须复用 prefix KV cache，否则速度和结构都偏离官方实现。
5. 若只想在 VLA-scratch 里做对照实验，至少要打开 `only_attend_to_final_layer=False`，否则仍然不是严格逐层融合。

## 9. 简短结论

官方 SmolVLA 的关键不是“用 SmolVLM 做最后一层特征提取”，而是“SmolVLM prefix 与 action expert suffix 在 transformer 层内逐层交互”。我们之前的 VLA-scratch 写法和这个结构有本质差异；TokenDrop / FrameSamp 只是 history token selection，不能弥补 action expert 融合结构上的差别。

