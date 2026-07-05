from __future__ import annotations

import importlib
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from copy import copy

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from vla_scratch.policies.modules.vlm_bridge.smolvlm.utils import (
    replace_context,
    resolve_local_smolvlm_model_source,
)
from vla_scratch.policies.utils.training import (
    apply_checkpoint_when_training,
    fully_shard_layers,
)
from vla_scratch.policies.modules.vlm_bridge.base import (
    VLMBridge,
    VLMOutputs,
    TARGET_IGNORE_ID,
)
from vla_scratch.policies.modules.vlm_bridge.smolvlm.processor import (
    SmolVLMPolicyInput,
)
from vla_scratch.policies.utils.transformers import make_att_2d_masks

if TYPE_CHECKING:
    from transformers.models.smolvlm.modeling_smolvlm import (
        SmolVLMForConditionalGeneration,
    )
    from vla_scratch.transforms.data_types import Observation


class SmolVLMBridge(VLMBridge):
    def __init__(self, *, model_id: str, vlm_type: str):
        super().__init__()
        resolved_model_id, local_files_only = resolve_local_smolvlm_model_source(
            model_id
        )
        tfm = importlib.import_module("transformers")
        try:
            vlm_cls = getattr(tfm, vlm_type)
        except AttributeError as e:
            raise ImportError(
                f"transformers has no class named '{vlm_type}'."
            ) from e

        load_kwargs = {
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        default_device = torch.get_default_device()
        if default_device.type == "cuda":
            load_kwargs["device_map"] = torch.cuda.current_device()
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.causal_model: "SmolVLMForConditionalGeneration" = (
            vlm_cls.from_pretrained(resolved_model_id, **load_kwargs)
        )

        SmolVLMProcessor = getattr(tfm, "SmolVLMProcessor")
        self.processor = SmolVLMProcessor.from_pretrained(
            resolved_model_id, local_files_only=local_files_only
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        # MEM support
        self.mem_config = None
        self.use_mem = False
        self.use_vision_token_memory = False
        self.vision_memory_max_tokens = 128
        self.vision_memory_selection = "even"
        self.vision_memory_integration = "context"
        self.vision_memory_candidate_tokens = 512
        self.vision_memory_token_drop_stride = 1
        self.vision_memory_token_per_image = 0
        self.vision_memory_add_pos_emb = False

    def configure_vision_token_memory(
        self,
        *,
        enabled: bool = False,
        max_tokens: int = 128,
        selection: str = "even",
        integration: str = "context",
        candidate_tokens: int = 512,
        token_drop_stride: int = 1,
        token_per_image: int = 0,
        add_pos_emb: bool = False,
    ):
        self.use_vision_token_memory = bool(enabled)
        self.vision_memory_max_tokens = int(max_tokens)
        if selection not in {"even", "tokendrop"}:
            raise ValueError(
                "vision token memory selection must be 'even' or 'tokendrop', "
                f"got {selection!r}"
            )
        self.vision_memory_selection = selection
        if integration not in {"context", "modulator", "expert"}:
            raise ValueError(
                "vision token memory integration must be 'context', "
                f"'modulator', or 'expert', got {integration!r}"
            )
        self.vision_memory_integration = integration
        self.vision_memory_candidate_tokens = int(candidate_tokens)
        self.vision_memory_token_drop_stride = max(1, int(token_drop_stride))
        self.vision_memory_token_per_image = int(token_per_image)
        self.vision_memory_add_pos_emb = bool(add_pos_emb)

    @staticmethod
    def _even_indices(length: int, count: int, device: torch.device) -> torch.Tensor:
        if length <= 0 or count <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if length <= count:
            return torch.arange(length, dtype=torch.long, device=device)
        return torch.linspace(0, length - 1, count, device=device).round().long()

    @staticmethod
    def _limit_tokens_even(tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
        if max_tokens <= 0 or tokens.shape[1] <= max_tokens:
            return tokens
        indices = SmolVLMBridge._even_indices(
            tokens.shape[1], max_tokens, tokens.device
        )
        return tokens.index_select(1, indices)

    @staticmethod
    def _pool_token_grid(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if target_tokens <= 0 or tokens.shape[-2] == target_tokens:
            return tokens
        source_tokens = tokens.shape[-2]
        source_grid = int(source_tokens ** 0.5)
        target_grid = int(target_tokens ** 0.5)
        if source_grid * source_grid != source_tokens:
            raise ValueError(f"Expected square source token grid, got {source_tokens}")
        if target_grid * target_grid != target_tokens:
            raise ValueError(f"Expected square target token grid, got {target_tokens}")
        if source_grid % target_grid != 0:
            raise ValueError(
                f"Cannot mean-pool {source_grid}x{source_grid} tokens to "
                f"{target_grid}x{target_grid}"
            )
        pool = source_grid // target_grid
        prefix_shape = tokens.shape[:-2]
        hidden = tokens.shape[-1]

        # Block mean pooling:
        #   z'_{i,j} = mean_{a,b < pool} z_{i*pool+a, j*pool+b}.
        # The square-grid checks above keep the spatial partition exact.
        grid = tokens.reshape(*prefix_shape, source_grid, source_grid, hidden)
        grid = grid.reshape(*prefix_shape, target_grid, pool, target_grid, pool, hidden)
        grid = grid.mean(dim=(-4, -2))
        return grid.reshape(*prefix_shape, target_tokens, hidden)

    @staticmethod
    def _sinusoidal_scalar(values: torch.Tensor, dim: int) -> torch.Tensor:
        if dim <= 0:
            return values.new_empty(*values.shape, 0)
        half = max(1, dim // 2)
        freq = torch.linspace(0.0, 1.0, half, device=values.device, dtype=torch.float32)
        period = 1.0 * (10000.0 ** freq)
        angles = values.to(torch.float32).unsqueeze(-1) / period
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.shape[-1] < dim:
            emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
        return emb[..., :dim]

    @staticmethod
    def _memory_position_embedding(
        *,
        batch_size: int,
        hist_frames: int,
        num_images: int,
        tokens_per_image: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        grid = int(tokens_per_image ** 0.5)
        if grid * grid != tokens_per_image:
            raise ValueError(
                f"Position embedding requires square token grids, got {tokens_per_image}"
            )
        t_dim = hidden_size // 3
        y_dim = hidden_size // 3
        x_dim = hidden_size - t_dim - y_dim
        time = torch.arange(hist_frames, device=device, dtype=torch.float32)
        view = torch.zeros(num_images, device=device, dtype=torch.float32)
        y = torch.arange(grid, device=device, dtype=torch.float32)
        x = torch.arange(grid, device=device, dtype=torch.float32)
        if hist_frames > 1:
            time = time / (hist_frames - 1)
        if num_images > 1:
            view = (
                torch.arange(num_images, device=device, dtype=torch.float32)
                / (num_images - 1)
            )
        if grid > 1:
            y = y / (grid - 1)
            x = x / (grid - 1)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        tt = time[:, None, None, None].expand(hist_frames, num_images, grid, grid)
        vv = view[None, :, None, None].expand(hist_frames, num_images, grid, grid)
        yy = yy[None, None].expand(hist_frames, num_images, grid, grid)
        xx = xx[None, None].expand(hist_frames, num_images, grid, grid)
        # Fold camera id into time so two camera streams do not share identical positions.
        pos = torch.cat(
            [
                SmolVLMBridge._sinusoidal_scalar((tt + vv) * 0.5, t_dim),
                SmolVLMBridge._sinusoidal_scalar(yy, y_dim),
                SmolVLMBridge._sinusoidal_scalar(xx, x_dim),
            ],
            dim=-1,
        )
        pos = pos.reshape(1, hist_frames, num_images, tokens_per_image, hidden_size)
        return pos.expand(batch_size, -1, -1, -1, -1).to(dtype=dtype)

    @staticmethod
    def _pixel_patch_change_scores(
        history_values: torch.Tensor,
        *,
        tokens_per_image: int,
        stride: int,
    ) -> torch.Tensor:
        batch_size, num_views, hist_frames, channels, height, width = (
            history_values.shape
        )
        grid = int(tokens_per_image ** 0.5)
        if grid * grid != tokens_per_image:
            raise ValueError(
                f"TokenDrop requires square token grids, got {tokens_per_image}"
            )
        patch_h = height // grid
        patch_w = width // grid
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(
                f"Image size {(height, width)} is too small for {grid}x{grid} tokens"
            )

        pixels = history_values[:, :, :, :, : patch_h * grid, : patch_w * grid]
        pixels = pixels.permute(0, 2, 1, 3, 4, 5).contiguous()
        pixels = pixels.reshape(
            batch_size,
            hist_frames,
            num_views,
            channels,
            grid,
            patch_h,
            grid,
            patch_w,
        )
        patch_pixels = pixels.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(
            batch_size,
            hist_frames,
            num_views,
            tokens_per_image,
            -1,
        )

        scores = torch.full(
            (batch_size, hist_frames, num_views, tokens_per_image),
            torch.finfo(history_values.dtype).min,
            dtype=history_values.dtype,
            device=history_values.device,
        )

        # RoboMME-style RGB-difference priority:
        #   s(t, v, p) = mean(|patch(t, v, p) - patch(t-stride, v, p)|).
        # The first history frame is pinned in memory; unchanged patches receive
        # -inf so a tight token budget is spent on visible motion.
        scores[:, 0] = 1000.0
        for frame_idx in range(stride, hist_frames, stride):
            prev_idx = max(0, frame_idx - stride)
            diff = (
                patch_pixels[:, frame_idx] - patch_pixels[:, prev_idx]
            ).abs().mean(dim=-1)
            scores[:, frame_idx] = torch.where(
                diff >= 1e-4,
                diff,
                torch.full_like(diff, torch.finfo(diff.dtype).min),
            )
        return scores

    def configure_mem_video_encoder(self, **kwargs):
        self.mem_config = kwargs
        self.use_mem = kwargs.get(
            "enabled",
            kwargs.get("use_mem_video_encoder", False),
        )
        self.mem_encoder = None
        if self.use_mem:
            from .video_mem_encoder import MEMVideoEncoder

            every_n = kwargs.get(
                "every_n_layers",
                kwargs.get("mem_video_every_n_layers", 4),
            )
            vision_encoder = self.causal_model.model.vision_model.encoder
            self.mem_encoder = MEMVideoEncoder(
                vision_encoder,
                mem_every_n_layers=every_n,
            )

    def _encode_images_with_mem(
        self, pixel_values: torch.Tensor
    ) -> torch.Tensor:
        if pixel_values.ndim != 6:
            raise ValueError(
                f"Expected 6D pixel_values for MEM, got {pixel_values.shape}"
            )
        if self.mem_encoder is None:
            raise RuntimeError("MEM video encoder is not configured.")

        batch_size, num_images, num_frames, channels, height, width = (
            pixel_values.shape
        )
        pixel_values_flat = pixel_values.reshape(
            batch_size * num_images * num_frames,
            channels,
            height,
            width,
        )

        vision_model = self.causal_model.model.vision_model
        patch_size = vision_model.config.patch_size
        patch_attention_mask = torch.ones(
            (
                pixel_values_flat.shape[0],
                height // patch_size,
                width // patch_size,
            ),
            dtype=torch.bool,
            device=pixel_values_flat.device,
        )
        hidden_states = vision_model.embeddings(
            pixel_values=pixel_values_flat.to(
                vision_model.embeddings.patch_embedding.weight.dtype
            ),
            patch_attention_mask=patch_attention_mask,
        )

        spatial_attention_mask = patch_attention_mask.view(
            pixel_values_flat.shape[0], -1
        )
        if vision_model.config._attn_implementation != "flash_attention_2":
            spatial_attention_mask = _prepare_4d_attention_mask(
                spatial_attention_mask,
                hidden_states.dtype,
            )
        elif not torch.any(~spatial_attention_mask):
            spatial_attention_mask = None

        hidden_states = self.mem_encoder(
            inputs_embeds=hidden_states,
            attention_mask=spatial_attention_mask,
            num_frames=num_frames,
        )
        last_hidden_state = vision_model.post_layernorm(hidden_states)
        return self.causal_model.model.connector(last_hidden_state)

    def _encode_history_as_memory_tokens(
        self,
        pixel_values: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.use_vision_token_memory or pixel_values.ndim != 6:
            return None

        batch_size, num_images, num_frames, channels, height, width = (
            pixel_values.shape
        )
        if num_frames <= 1:
            return None

        history_values = pixel_values[:, :, :-1]
        hist_frames = num_frames - 1
        encoder_values = history_values.permute(0, 2, 1, 3, 4, 5)
        encoder_values = encoder_values.reshape(
            batch_size,
            hist_frames * num_images,
            channels,
            height,
            width,
        )

        memory_tokens = self.causal_model.model.get_image_features(
            encoder_values
        )

        max_tokens = self.vision_memory_max_tokens
        if memory_tokens.ndim == 4:
            target_tokens = self.vision_memory_token_per_image
            if target_tokens > 0:
                memory_tokens = self._pool_token_grid(memory_tokens, target_tokens)
            tokens_per_image = memory_tokens.shape[2]
            token_grid = memory_tokens.reshape(
                batch_size,
                hist_frames,
                num_images,
                tokens_per_image,
                memory_tokens.shape[-1],
            )
            if self.vision_memory_add_pos_emb:
                token_grid = token_grid + self._memory_position_embedding(
                    batch_size=batch_size,
                    hist_frames=hist_frames,
                    num_images=num_images,
                    tokens_per_image=tokens_per_image,
                    hidden_size=token_grid.shape[-1],
                    device=token_grid.device,
                    dtype=token_grid.dtype,
                )

            if self.vision_memory_selection == "tokendrop":
                scores = self._pixel_patch_change_scores(
                    history_values,
                    tokens_per_image=tokens_per_image,
                    stride=self.vision_memory_token_drop_stride,
                )
                flat_tokens = token_grid.reshape(
                    batch_size, -1, token_grid.shape[-1]
                )
                flat_scores = scores.reshape(batch_size, -1)
                candidate_tokens = self.vision_memory_candidate_tokens
                if candidate_tokens > 0 and flat_tokens.shape[1] > candidate_tokens:
                    candidate = flat_scores.topk(candidate_tokens, dim=1).indices
                    candidate_scores = flat_scores.gather(1, candidate)
                    keep = (
                        min(max_tokens, candidate.shape[1])
                        if max_tokens > 0
                        else candidate.shape[1]
                    )
                    selected = candidate.gather(
                        1, candidate_scores.topk(keep, dim=1).indices
                    )
                elif max_tokens > 0 and flat_tokens.shape[1] > max_tokens:
                    selected = flat_scores.topk(max_tokens, dim=1).indices
                else:
                    selected = torch.arange(
                        flat_tokens.shape[1],
                        device=flat_tokens.device,
                        dtype=torch.long,
                    ).unsqueeze(0).expand(batch_size, -1)
                selected = selected.sort(dim=1).values
                gather_index = selected.unsqueeze(-1).expand(
                    -1, -1, flat_tokens.shape[-1]
                )
                memory_tokens = flat_tokens.gather(1, gather_index)
            else:
                tokens_per_frame = num_images * tokens_per_image
                if max_tokens > 0:
                    max_frames = max(1, max_tokens // max(1, tokens_per_frame))
                else:
                    max_frames = hist_frames
                frame_indices = self._even_indices(
                    hist_frames, max_frames, token_grid.device
                )
                memory_tokens = token_grid.index_select(1, frame_indices)
                memory_tokens = memory_tokens.reshape(
                    batch_size, -1, token_grid.shape[-1]
                )
                memory_tokens = self._limit_tokens_even(memory_tokens, max_tokens)
        else:
            memory_tokens = memory_tokens.reshape(batch_size, -1, memory_tokens.shape[-1])
            memory_tokens = self._limit_tokens_even(memory_tokens, max_tokens)

        return memory_tokens.to(device=device, dtype=dtype)

    def apply_fsdp(self, mp_policy, mesh):
        fully_shard_layers(
            self.causal_model.model.vision_model.encoder.layers,
            mesh,
            mp_policy,
            num_to_prefetch=6,
        )
        fully_shard_layers(
            self.causal_model.model.text_model.layers,
            mesh,
            mp_policy,
            num_to_prefetch=6,
        )

    def get_text_dims(self) -> Tuple[int, int, int]:
        cfg = self.causal_model.config.text_config
        head_dim = getattr(
            cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
        )
        return (
            cfg.num_hidden_layers,
            head_dim,
            cfg.num_key_value_heads,
            cfg.hidden_size,
        )

    @replace_context()
    def encode(
        self,
        observation: "Observation",
        *,
        extra_embs: Optional[torch.Tensor] = None,
        extra_pad_masks: Optional[torch.Tensor] = None,
        extra_att_masks: Optional[torch.Tensor] = None,
        zero_pos_id_for_extra: bool = False,
        extra_attention_mask: bool = False,
    ) -> Tuple[torch.Tensor, VLMOutputs, Dict]:
        policy_td: "SmolVLMPolicyInput" = observation.policy_input
        if not isinstance(policy_td, SmolVLMPolicyInput):
            raise TypeError(
                "Observation policy_input must be SmolVLMPolicyInput"
            )

        input_ids = policy_td.input_ids
        input_pad_masks = policy_td.attention_mask
        target_ids = policy_td.target_ids
        pixel_values = policy_td.pixel_values
        bsz = input_ids.shape[0]

        torch.cuda.nvtx.range_push("embed_text")
        text_model = self.causal_model.model.text_model
        inputs_embeds = text_model.get_input_embeddings()(input_ids)
        torch.cuda.nvtx.range_pop()

        memory_tokens = self._encode_history_as_memory_tokens(
            pixel_values,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        torch.cuda.nvtx.range_push("embed_image")
        if self.use_mem and pixel_values.ndim == 6:
            image_hidden_states = self._encode_images_with_mem(pixel_values)
        else:
            if pixel_values.ndim == 6:
                pixel_values = pixel_values[:, :, -1]

            image_hidden_states = self.causal_model.model.get_image_features(
                pixel_values
            )
        image_hidden_states = image_hidden_states.to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        torch.cuda.nvtx.range_push("merge_inputs")
        merged_embeds = self.causal_model.model.inputs_merger(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_hidden_states=image_hidden_states,
        )
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

        embs = [merged_embeds]
        pad_masks = [input_pad_masks]
        att_masks = [
            torch.ones(
                merged_embeds.shape[1],
                dtype=torch.bool,
                device=merged_embeds.device,
            ),
        ]

        loss_exclude_len = 0
        memory_pad_masks = None
        if memory_tokens is not None:
            memory_pad_masks = torch.ones(
                (bsz, memory_tokens.shape[1]),
                dtype=torch.bool,
                device=memory_tokens.device,
            )

        if (
            memory_tokens is not None
            and self.vision_memory_integration == "context"
        ):
            embs.append(memory_tokens)
            pad_masks.append(memory_pad_masks)
            memory_att_mask = torch.zeros(
                memory_tokens.shape[1],
                dtype=torch.bool,
                device=memory_tokens.device,
            )
            if memory_att_mask.numel() > 0:
                memory_att_mask[0] = True
            att_masks.append(memory_att_mask)
            loss_exclude_len = memory_tokens.shape[1]

        extra_len = 0
        if extra_embs is not None:
            embs.append(extra_embs)
            pad_masks.append(extra_pad_masks)
            att_masks.append(extra_att_masks)
            extra_len = extra_embs.shape[1]
            loss_exclude_len += extra_len

        embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_att_masks_1d = torch.cat(att_masks, dim=0).expand(bsz, -1)

        # position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # position_ids.masked_fill_(~prefix_pad_masks.bool(), 0)
        position_ids = torch.arange(embs.shape[1], device=embs.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        if extra_len > 0 and zero_pos_id_for_extra:
            position_ids[:, -extra_len:] = 0

        attention_mask = prefix_pad_masks
        if extra_embs is not None or extra_attention_mask:
            prefix_att_2d = make_att_2d_masks(
                prefix_pad_masks, prefix_att_masks_1d
            )
            if extra_embs is not None and extra_attention_mask:
                obs_reg_att_mask = policy_td.obs_register_att_mask
                prefix_len = input_pad_masks.shape[1]
                obs_reg_att_mask = obs_reg_att_mask[:, None, :].expand(
                    bsz, extra_len, prefix_len
                )
                prefix_att_2d[:, -extra_len:, :prefix_len] = obs_reg_att_mask

            attn_mask = torch.zeros(
                prefix_att_2d.shape, device=embs.device, dtype=embs.dtype
            )
            attn_mask.masked_fill_(
                ~prefix_att_2d, torch.finfo(attn_mask.dtype).min
            )
            attention_mask = attn_mask[:, None, :, :]

        # from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask

        cache_position = torch.arange(embs.shape[1], device=embs.device)
        # past_key_values = DynamicCache()
        causal_mask = create_causal_mask(
            config=text_model.config,
            input_embeds=embs,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        position_embeddings = text_model.rotary_emb(embs, position_ids)

        hidden_states = embs
        kv_cache_list = []
        encoder_hidden_states_list = []
        for layer_idx, decoder_layer in enumerate(text_model.layers):
            torch.cuda.nvtx.range_push(f"layer_{layer_idx}")
            # past_key_values_this_layer = copy(past_key_values)
            outputs = apply_checkpoint_when_training(
                self,
                decoder_layer,
                hidden_states,
                attention_mask=causal_mask,
                # past_key_values=past_key_values_this_layer,
                position_embeddings=position_embeddings,
                disable=True,
            )
            torch.cuda.nvtx.range_pop()
            # layer_cache = past_key_values_this_layer.layers.pop(-1)
            # kv = (layer_cache.keys, layer_cache.values)
            hidden_states, kv = outputs

            kv_cache_list.append(kv)
            encoder_hidden_states_list.append(hidden_states)

        hidden_states = text_model.norm(hidden_states)
        hidden_state_list = torch.stack(encoder_hidden_states_list, dim=1)

        pred_logits = self.causal_model.lm_head(
            hidden_states[:, : hidden_states.shape[1] - loss_exclude_len]
        )
        pred_logits = pred_logits[:, :-1]
        pred_logits = pred_logits.reshape(-1, pred_logits.shape[-1])
        target_ids = target_ids[:, 1:].reshape(-1)
        ce_loss_sum = torch.nn.functional.cross_entropy(
            pred_logits,
            target_ids,
            ignore_index=TARGET_IGNORE_ID,
            reduction="sum",
        )
        num_correct_tokens = (pred_logits.argmax(dim=-1) == target_ids).sum()
        total = (target_ids != TARGET_IGNORE_ID).sum().clamp(min=1)
        ce_loss = ce_loss_sum / total
        accuracy = num_correct_tokens.float() / total

        key_states = torch.stack([k for k, _v in kv_cache_list], dim=1)
        value_states = torch.stack([v for _k, v in kv_cache_list], dim=1)

        vlm_outputs = VLMOutputs(
            last_hidden_state=hidden_states,
            prefix_pad_masks=prefix_pad_masks,
            key_states=key_states,
            value_states=value_states,
            hidden_state_list=hidden_state_list,
            memory_tokens=memory_tokens,
            memory_pad_masks=memory_pad_masks,
            batch_size=[bsz],
        )
        padding_ratio = policy_td.attention_mask.float().mean(dim=-1)
        log_dict = {
            "padding_ratio/mean": padding_ratio.mean(),
            "padding_ratio/std": padding_ratio.std(unbiased=False),
            "padding_ratio/min": padding_ratio.min(),
            "padding_ratio/max": padding_ratio.max(),
            "loss/ce_loss": ce_loss.detach(),
            "loss/accuracy": accuracy.detach(),
        }
        if memory_tokens is not None:
            log_dict["memory/tokens"] = torch.as_tensor(
                memory_tokens.shape[1],
                device=memory_tokens.device,
                dtype=torch.float32,
            )
        return ce_loss, vlm_outputs, log_dict
