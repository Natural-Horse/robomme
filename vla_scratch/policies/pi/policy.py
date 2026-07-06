import time
from typing import Tuple, TYPE_CHECKING, Dict
import einops
import jaxtyping as at

import torch
import torch.nn as nn
from torch.distributed.fsdp._fully_shard import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)

from tensordict import TensorClass

from vla_scratch.policies.base import BasePolicy
from vla_scratch.policies.modules.action_expert import DiTModel
from vla_scratch.policies.modules.vlm_bridge import (
    Qwen3VLBridge,
    PaligemmaBridge,
    SmolVLMBridge,
)

from vla_scratch.policies.utils.training import (
    apply_checkpoint_when_training,
    fully_shard_layers,
)
from vla_scratch.policies.utils.diffusion import (
    build_beta_time_dist,
    sample_clamped_time,
    repeat_batch,
    sample_noise,
)
from vla_scratch.policies.utils.transformers import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)

if TYPE_CHECKING:
    from vla_scratch.policies.modules.vlm_bridge.base import VLMOutputs
    from vla_scratch.policies.pi.config import PiConfig
    from vla_scratch.transforms.data_types import Observation, DataSample


class SuffixInput(TensorClass):
    prefix_pad_masks: at.Bool[torch.Tensor, " batch prefix_len"]  # noqa: F722
    hidden_state_list: at.Float[
        torch.Tensor, " batch n_layer prefix_len hidden"  # noqa: F722
    ]
    memory_cond: torch.Tensor


class PiPolicy(BasePolicy):
    suffix_pad_mask: at.Bool[torch.Tensor, " action_horizon"]  # noqa: F722
    suffix_att_mask: at.Bool[torch.Tensor, " action_horizon"]  # noqa: F722

    def __init__(self, config: "PiConfig"):
        super().__init__()
        self.config = config
        
        # Debug print
        print(f"DEBUG IN PiPolicy.__init__: config type={type(config)}")
        print(f"DEBUG IN PiPolicy.__init__: config.action_dim={config.action_dim}")
        print(f"DEBUG IN PiPolicy.__init__: config.state_dim={config.state_dim}")

        if config.action_dim is None or config.state_dim is None:
            print("ERROR: action_dim or state_dim is None!")
            raise ValueError(
                "PiConfig.action_dim and PiConfig.state_dim must be set before "
                "initializing PiPolicy."
            )

        start_time = time.time()
        if config.vlm_type == "PaliGemmaForConditionalGeneration":
            self.vlm_bridge = PaligemmaBridge(
                model_id=config.model_id,
                vlm_type=config.vlm_type,
            )
        elif config.vlm_type == "Qwen3VLForConditionalGeneration":
            self.vlm_bridge = Qwen3VLBridge(
                model_id=config.model_id,
                vlm_type=config.vlm_type,
            )
        elif config.vlm_type == "SmolVLMForConditionalGeneration":
            self.vlm_bridge = SmolVLMBridge(
                model_id=config.model_id,
                vlm_type=config.vlm_type,
            )
        else:
            raise NotImplementedError(
                f"Unsupported VLM type for PiPolicy: {config.vlm_type}"
            )

        if hasattr(self.vlm_bridge, "configure_mem_video_encoder"):
            self.vlm_bridge.configure_mem_video_encoder(
                enabled=config.use_mem_video_encoder,
                every_n_layers=config.mem_video_every_n_layers,
                frame_history=config.mem_video_frame_history,
                num_cameras=config.mem_video_num_cameras,
                temporal_min_period=config.mem_video_temporal_min_period,
                temporal_max_period=config.mem_video_temporal_max_period,
                drop_past=config.mem_video_drop_past,
            )
        if hasattr(self.vlm_bridge, "configure_vision_token_memory"):
            self.vlm_bridge.configure_vision_token_memory(
                enabled=config.use_vision_token_memory,
                max_tokens=config.vision_memory_max_tokens,
                selection=config.vision_memory_selection,
                integration=config.vision_memory_integration,
                candidate_tokens=config.vision_memory_candidate_tokens,
                token_drop_stride=config.vision_memory_token_drop_stride,
                token_per_image=config.vision_memory_token_per_image,
                add_pos_emb=config.vision_memory_add_pos_emb,
            )

        end_time = time.time()
        print(
            f"VLM model initialized in {end_time - start_time:.2f} seconds: {config.vlm_type}"
        )
        if config.freeze_vlm:
            for param in self.vlm_bridge.parameters():
                param.requires_grad = False

        # number of hidden layers and head dim must match to do cross-attention at each layer
        text_layers, text_head_dim, text_num_kv_heads, vlm_hidden_size = (
            self.vlm_bridge.get_text_dims()
        )
        self.vlm_hidden_size = vlm_hidden_size

        self.use_obs_register = config.num_obs_registers > 0
        if self.use_obs_register:
            # add a learnable token to the VLM for observation register
            self.obs_registers = nn.Parameter(
                torch.zeros(config.num_obs_registers, vlm_hidden_size)
            )
            self.register_buffer(
                "obs_registers_pad_masks",
                torch.ones(config.num_obs_registers, dtype=torch.bool),
                persistent=False,
            )
            self.register_buffer(
                "obs_registers_att_masks",
                torch.zeros(config.num_obs_registers, dtype=torch.bool),
                persistent=False,
            )
            self.obs_registers_att_masks[0] = 1
            # prevent prefix from attending to registers
        else:
            assert not config.expert_only_use_register, (
                "expert_only_use_register must be False when num_obs_registers is 0."
            )
        action_expert_config = config.action_expert_cfg
        start_time = time.time()
        self.action_expert = DiTModel(config=action_expert_config)
        end_time = time.time()
        print(
            f"Action expert initialized in {end_time - start_time:.2f} seconds."
        )

        action_expert_width = action_expert_config.hidden_size
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_width)
        self.action_out_proj = nn.Linear(action_expert_width, config.action_dim)
        self.state_in_proj = nn.Linear(config.state_dim, action_expert_width)
        self.memory_mod_proj = nn.Linear(vlm_hidden_size, action_expert_width)

        self.time_mlp = nn.Sequential(
            nn.Linear(action_expert_width, action_expert_width),
            nn.SiLU(),
            nn.Linear(action_expert_width, action_expert_width),
            nn.SiLU(),
        )

        # register buffers
        if config.use_state:
            suffix_len = config.action_horizon + config.state_history
        else:
            suffix_len = config.action_horizon
        self.suffix_len = suffix_len
        suffix_pad_mask = torch.ones(suffix_len, dtype=torch.bool)
        suffix_att_mask = torch.zeros(suffix_len, dtype=torch.bool)
        # create a new attention block for the suffix, prefix should not attend to suffix
        suffix_att_mask[0] = 1
        self.register_buffer(
            "suffix_pad_mask", suffix_pad_mask, persistent=False
        )
        self.register_buffer(
            "suffix_att_mask", suffix_att_mask, persistent=False
        )

        if config.suffix_add_pos_emb:
            pos_emb_state = torch.zeros(
                config.state_history, action_expert_width
            )
            self.position_embedding_state = nn.Parameter(pos_emb_state)
            pos_emb_action = torch.zeros(
                config.action_horizon, action_expert_width
            )
            self.position_embedding_action = nn.Parameter(pos_emb_action)

        param_device = next(self.parameters()).device
        self.time_dist = build_beta_time_dist(
            self.config.time_dist_alpha,
            self.config.time_dist_beta,
            device=param_device,
        )

    def initialize_weights(self):
        if self.config.suffix_add_pos_emb:
            nn.init.normal_(
                self.position_embedding_state,
                mean=0.0,
                std=self.config.suffix_pos_emb_init_gain,
            )
            nn.init.normal_(
                self.position_embedding_action,
                mean=0.0,
                std=self.config.suffix_pos_emb_init_gain,
            )
        if self.use_obs_register:
            nn.init.normal_(
                self.obs_registers,
                mean=0.0,
                std=self.config.obs_register_init_gain,
            )
        self.action_expert.initialize_weights()

    def materialize_lazy_modules(self) -> None:
        """Initialize Lazy* parameters so checkpoint loading can inspect them safely."""
        device = self.action_in_proj.weight.device
        dtype = self.action_in_proj.weight.dtype
        batch_size = 1
        suffix_len = max(1, int(self.suffix_len))
        hidden_size = int(self.config.action_expert_cfg.hidden_size)
        num_layers = int(self.config.action_expert_cfg.num_hidden_layers)

        inputs_embeds = torch.zeros(
            (batch_size, suffix_len, hidden_size),
            device=device,
            dtype=dtype,
        )
        position_ids = torch.arange(suffix_len, device=device, dtype=torch.long)[
            None, :
        ]
        adarms_cond = torch.zeros(
            (batch_size, hidden_size),
            device=device,
            dtype=dtype,
        )
        encoder_seq_len = max(1, int(self.config.num_obs_registers or 1))
        attention_mask = torch.ones(
            (batch_size, 1, suffix_len, encoder_seq_len + suffix_len),
            device=device,
            dtype=torch.bool,
        )
        encoder_hidden_states = [
            torch.zeros(
                (batch_size, encoder_seq_len, self.vlm_hidden_size),
                device=device,
                dtype=dtype,
            )
            for _ in range(max(1, num_layers))
        ]

        training = self.action_expert.training
        self.action_expert.eval()
        with torch.no_grad():
            self.action_expert(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                adarms_cond=adarms_cond,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )
        self.action_expert.train(training)

    def apply_fsdp(self, param_type, reduce_type, output_dtype, mesh):
        """Helper function to apply FSDP to a module with given mixed precision policy."""

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_type,
            reduce_dtype=reduce_type,
            cast_forward_inputs=True,
        )
        self.vlm_bridge.apply_fsdp(mp_policy, mesh)

        fully_shard_layers(
            self.action_expert.blocks, mesh, mp_policy, num_to_prefetch=6
        )

        mp_policy_root = MixedPrecisionPolicy(
            param_dtype=param_type,
            reduce_dtype=reduce_type,
            output_dtype=output_dtype,
            cast_forward_inputs=True,
        )
        fully_shard(self, mesh=mesh, mp_policy=mp_policy_root)
        register_fsdp_forward_method(self, "compute_loss")
        # will c10 error if below is not registered
        register_fsdp_forward_method(self, "encode_prefix")
        register_fsdp_forward_method(self, "predict_suffix")
        register_fsdp_forward_method(self, "sample_actions")
        return self

    def encode_prefix(
        self, observation: "Observation"
    ) -> Tuple[torch.Tensor, "VLMOutputs", Dict]:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        # Prepare extra observation register tokens if configured
        extra_embs = None
        extra_pad = None
        extra_att = None
        if self.use_obs_register:
            bsize = observation.shape[0]
            extra_embs = einops.repeat(
                self.obs_registers, "s d -> b s d", b=bsize
            )
            extra_pad = einops.repeat(
                self.obs_registers_pad_masks, "s -> b s", b=bsize
            )
            extra_att = self.obs_registers_att_masks

        # Bridge handles model-specific preprocessing + transformer forward
        ce_loss, vlm_outputs, log_dict = self.vlm_bridge.encode(
            observation=observation,
            extra_embs=extra_embs,
            extra_pad_masks=extra_pad,
            extra_att_masks=extra_att,
            zero_pos_id_for_extra=self.config.zero_pos_id_for_obs_register,
            extra_attention_mask=self.config.causal_mask_obs_register,
        )
        return ce_loss, vlm_outputs, log_dict

    def construct_suffix_input(self, vlm_outputs: "VLMOutputs") -> SuffixInput:
        """Construct SuffixInput from VLMOutputs for caching purposes."""
        # only retain last N layers for action expert
        prefix_pad_masks = vlm_outputs.prefix_pad_masks
        action_expert_layers = self.config.action_expert_cfg.num_hidden_layers
        if self.config.vlm_layer_selection == "first":
            hidden_state_list = vlm_outputs.hidden_state_list[
                :, :action_expert_layers
            ]
        elif self.config.vlm_layer_selection == "last":
            hidden_state_list = vlm_outputs.hidden_state_list[
                :, -action_expert_layers:
            ]
        else:
            raise ValueError(
                "vlm_layer_selection must be 'first' or 'last', "
                f"got {self.config.vlm_layer_selection!r}"
            )
        # only use the last num_obs_registers tokens from the prefix for the expert
        if self.config.expert_only_use_register:
            torch.cuda.nvtx.range_push("select_obs_registers")
            num_registers = self.config.num_obs_registers
            prefix_pad_masks = prefix_pad_masks[:, -num_registers:]
            hidden_state_list = hidden_state_list[:, :, -num_registers:, :]
            torch.cuda.nvtx.range_pop()

        memory_tokens = getattr(vlm_outputs, "memory_tokens", None)
        memory_pad_masks = getattr(vlm_outputs, "memory_pad_masks", None)
        has_memory_tokens = (
            self.config.use_vision_token_memory
            and isinstance(memory_tokens, torch.Tensor)
            and memory_tokens.numel() > 0
        )
        if has_memory_tokens and memory_pad_masks is None:
            memory_pad_masks = torch.ones(
                memory_tokens.shape[:2],
                dtype=torch.bool,
                device=memory_tokens.device,
            )

        if (
            has_memory_tokens
            and self.config.vision_memory_integration == "expert"
        ):
            memory_hidden_state_list = memory_tokens[:, None].expand(
                -1, hidden_state_list.shape[1], -1, -1
            )
            hidden_state_list = torch.cat(
                [hidden_state_list, memory_hidden_state_list], dim=2
            )
            prefix_pad_masks = torch.cat(
                [prefix_pad_masks, memory_pad_masks], dim=1
            )

        if (
            has_memory_tokens
            and self.config.vision_memory_integration == "modulator"
        ):
            weights = memory_pad_masks.to(memory_tokens.dtype).unsqueeze(-1)
            pooled_memory = (memory_tokens * weights).sum(dim=1) / weights.sum(
                dim=1
            ).clamp_min(1.0)
            memory_cond = self.memory_mod_proj(pooled_memory)
        else:
            memory_cond = torch.zeros(
                (
                    prefix_pad_masks.shape[0],
                    self.config.action_expert_cfg.hidden_size,
                ),
                dtype=hidden_state_list.dtype,
                device=hidden_state_list.device,
            )

        suffix_input = SuffixInput(
            prefix_pad_masks=prefix_pad_masks,
            hidden_state_list=hidden_state_list,
            memory_cond=memory_cond,
            batch_size=vlm_outputs.shape,
        )
        return suffix_input

    def predict_suffix(
        self,
        state: at.Float[torch.Tensor, " batch horizon dim"],  # noqa: F722
        suffix_input: SuffixInput,
        noisy_actions,
        time,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Apply one denoising step of `noisy_actions` at a given timestep."""
        prefix_pad_masks = suffix_input.prefix_pad_masks

        torch.cuda.nvtx.range_push("embed_suffix")
        suffix_embs, suffix_pad_masks, suffix_att_masks, time_emb = (
            self._embed_suffix(state, noisy_actions, time)
        )
        if (
            self.config.use_vision_token_memory
            and self.config.vision_memory_integration == "modulator"
        ):
            time_emb = time_emb + suffix_input.memory_cond.to(time_emb.dtype)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("attention_mask")
        suffix_att_2d_masks = make_att_2d_masks(
            suffix_pad_masks, suffix_att_masks
        )
        prefix_pad_mask = einops.repeat(
            prefix_pad_masks, "b p -> b s p", s=self.suffix_len
        )
        full_att_2d_mask = torch.cat(
            [prefix_pad_mask, suffix_att_2d_masks], dim=2
        )
        full_att_mask = einops.rearrange(full_att_2d_mask, "b i j -> b 1 i j")
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("position_ids")
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = (
            prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        )
        torch.cuda.nvtx.range_pop()

        encoder_hidden_states = suffix_input.hidden_state_list.unbind(dim=1)
        disp_loss, suffix_out, log_dict = self.action_expert.forward(
            inputs_embeds=suffix_embs,
            position_ids=position_ids,
            adarms_cond=time_emb,
            attention_mask=full_att_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :, :]
        return disp_loss, self.action_out_proj(suffix_out), log_dict

    def compute_loss(
        self,
        data_sample: "DataSample",
    ) -> Tuple[torch.Tensor, Dict]:
        torch.cuda.nvtx.range_push("Model Encode Prefix")
        if self.config.freeze_vlm:
            with torch.no_grad():
                _, vlm_outputs, log_dict_prefix = self.encode_prefix(
                    observation=data_sample.observation
                )
            ce_loss = torch.zeros((), device=next(self.parameters()).device)
        else:
            ce_loss, vlm_outputs, log_dict_prefix = self.encode_prefix(
                observation=data_sample.observation
            )
        torch.cuda.nvtx.range_pop()

        if data_sample.action_chunk is not None:
            suffix_input = self.construct_suffix_input(vlm_outputs)
            if self.config.detach_encoder_output:
                torch.cuda.nvtx.range_push("Detach KV Cache")
                suffix_input = suffix_input.detach()
                torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Expand Data Sample")
            data_sample = repeat_batch(
                data_sample, self.config.num_noise_per_sample
            )
            suffix_input = repeat_batch(
                suffix_input, self.config.num_noise_per_sample
            )
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Apply Noise")
            actions = data_sample.action_chunk.actions
            selected_noise = sample_noise(
                actions.shape, actions.device, actions.dtype
            )
            u_t = selected_noise - actions
            timestep = sample_clamped_time(self.time_dist, data_sample.shape)
            noisy_actions = actions + timestep[:, None, None] * u_t
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("Model Predict Suffix")
            _, v_t, log_dict_suffix = self.predict_suffix(
                state=data_sample.observation.state,
                suffix_input=suffix_input,
                noisy_actions=noisy_actions,
                time=timestep,
            )
            torch.cuda.nvtx.range_pop()
            
            # Using Huber Loss (L1-like) for robustness
            flow_loss_elem = torch.nn.functional.huber_loss(
                v_t, u_t.to(v_t.dtype), reduction="none", delta=1.0
            )

            loss_mask = getattr(data_sample.action_chunk, "loss_mask", None)
            if loss_mask is not None:
                mask = loss_mask.to(flow_loss_elem.dtype)
                denom = mask.sum().clamp_min(1.0)
                flow_mse = (flow_loss_elem * mask).sum() / denom
                per_dim_num = (flow_loss_elem * mask).sum(dim=(0, 1))
                per_dim_den = mask.sum(dim=(0, 1)).clamp_min(1.0)
                flow_per_dim = per_dim_num / per_dim_den
            else:
                flow_mse = flow_loss_elem.mean()
                flow_per_dim = flow_loss_elem.mean(dim=(0, 1))
            
            log_dict_suffix["loss/flow_mse"] = flow_mse.detach()
            for dim_idx, dim_val in enumerate(flow_per_dim):
                log_dict_suffix[f"loss/flow_dim_{dim_idx}"] = dim_val.detach()
        else:
            flow_mse = 0.0
            log_dict_suffix = {}

        loss = flow_mse + self.config.ce_loss_weight * ce_loss

        log_dict = {**log_dict_prefix, **log_dict_suffix}

        return loss, log_dict

    @torch.inference_mode()
    def sample_actions(
        self, observation: "Observation", num_steps=10
    ) -> at.Float[torch.Tensor, " batch_size action_horizon action_dim"]:  # noqa: F722
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        torch.cuda.nvtx.range_push("VLM Encode Prefix")
        _, vlm_output, log_dict = self.encode_prefix(observation)
        suffix_input = self.construct_suffix_input(vlm_output)
        torch.cuda.nvtx.range_pop()

        bsize = observation.shape[0]
        device = observation.device
        dtype = observation.state.dtype

        actions_shape = (
            bsize,
            self.config.action_horizon,
            self.config.action_dim,
        )
        noise = sample_noise(actions_shape, device, dtype)

        dt_float = 1.0 / num_steps
        time_float = 1.0
        dt = torch.tensor(dt_float, dtype=dtype, device=device)
        time = torch.tensor(time_float, dtype=dtype, device=device)

        x_t = noise
        while time_float >= dt_float / 2:
            torch.cuda.nvtx.range_push("Predict Suffix in Sampling")
            _, v_t, _ = self.predict_suffix(
                observation.state,
                suffix_input=suffix_input,
                noisy_actions=x_t,
                time=time.expand(bsize),
            )

            x_t = x_t - dt * v_t
            time -= dt
            time_float -= dt_float
            torch.cuda.nvtx.range_pop()
        return x_t

    def _embed_suffix(  # noqa: F722
        self,
        state: at.Float[torch.Tensor, " batch_size state_history state_dim"],  # noqa: F722
        noisy_actions: at.Float[
            torch.Tensor, " batch_size action_horizon action_dim"  # noqa: F722
        ],
        time: at.Float[torch.Tensor, " batch_size"],  # noqa: F722
    ) -> Tuple[
        at.Float[torch.Tensor, " batch_size action_horizon hidden_dim"],  # noqa: F722
        at.Bool[torch.Tensor, " batch_size action_horizon"],  # noqa: F722
        at.Bool[torch.Tensor, " batch_size action_horizon"],  # noqa: F722
        at.Float[torch.Tensor, " batch_size hidden_dim"],  # noqa: F722
    ]:  # noqa: F722
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]

        # use float 32 for sinusoidal embedding
        time_fp32 = time.to(torch.float32)
        time_emb_fp32 = create_sinusoidal_pos_embedding(
            time_fp32,
            dimension=self.config.action_expert_cfg.hidden_size,
            min_period=4e-3,
            max_period=4.0,
            device=time.device,
            dtype=time_fp32.dtype,
        )
        time_emb = time_emb_fp32.to(time.dtype)

        time_emb = self.time_mlp(time_emb)

        action_emb = self.action_in_proj(noisy_actions)

        if self.config.use_state:
            state_emb = self.state_in_proj(state)
            if self.config.suffix_add_pos_emb:
                state_emb = (
                    state_emb + self.position_embedding_state[None, :, :]
                )
                action_emb = (
                    action_emb + self.position_embedding_action[None, :, :]
                )
            suffix_emb = torch.cat([state_emb, action_emb], dim=1)
        else:
            if self.config.suffix_add_pos_emb:
                action_emb = (
                    action_emb + self.position_embedding_action[None, :, :]
                )
            suffix_emb = action_emb

        bsize = action_emb.shape[0]
        pad_mask = einops.repeat(
            self.suffix_pad_mask, "action_horizon -> b action_horizon", b=bsize
        )
        att_mask = einops.repeat(
            self.suffix_att_mask, "action_horizon -> b action_horizon", b=bsize
        )

        return suffix_emb, pad_mask, att_mask, time_emb
