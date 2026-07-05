# LIBERO 实验结果汇总

更新时间：2026-06-15

说明：表中 SR 为 LIBERO 官方评测成功率。FrameSamp 和 TokenDrop 按相同训练设置下各 suite 的最佳评测值汇总；MEM 目前只有 Spatial 结果。

| 方法 | 训练配置 | Backbone 与主要参数 | Goal SR | Spatial SR | Object SR | 平均 SR | 备注 |
|---|---|---|---:|---:|---:|---:|---|
| baseline | 50 epochs | SmolVLM2-256M-Video-Instruct；DiT action expert：12 layers，hidden 1024，cross-attn every 2 layers；action horizon 10，state history 1；使用最终 VLM 层 register tokens。 | 46.0% | 36.0% | 56.0% | 46.0% | 已更新为 full100 no-video 评测。不使用历史记忆输入，基于当前观测和语言指令预测动作。 |
| framesamp | 50 epochs | SmolVLA-style layerwise 配置：SmolVLM2-256M；输入图像 256；action horizon 30；action expert 8 layers，hidden 384；使用前段 VLM 层信息，`only_attend_to_final_layer=false`；vision token memory，selection=even，max tokens 512，token_per_image=16，带时空位置编码。 | 49.0% | 31.0% | 64.0% | 48.0% | 已更新为 full100 no-video 评测。SmolVLM connector 后每张图 16 tokens，4 历史帧 x 2 相机共 128 个 tokens。 |
| tokendrop | 50 epochs | SmolVLA-style layerwise 配置；vision token memory，selection=tokendrop，max tokens 512，candidate tokens 2048，token_per_image=64，token_drop_stride=8，带时空位置编码；`only_attend_to_final_layer=false`。 | 48.0% | 48.0% | 60.0% | 52.0% | Goal 使用当前 corrected 架构 `checkpoint_30` 的 full100 no-video 评测；Spatial 最佳为当前 corrected 架构 `checkpoint_30` 的 full100 no-video 评测，`checkpoint_50` 的最终 full100 no-video 测评为 30.0%。本轮 Spatial 50 epochs 结束后已停止后续 Object 训练。旧 tokendrop 结果来自错误架构，已不纳入此行。 |
| MEM | 50 epochs | SmolVLM2-256M-Video-Instruct；MEM video encoder：5-frame history，2 cameras，every 4 vision layers；DiT 参数同 baseline。 | N/A | 8.0% | N/A | 8.0% | Spatial 已更新为 full100 no-video 评测；使用固定长度历史视觉记忆作为额外输入，训练读取 memory 数据，推理从在线历史缓存构造。 |
