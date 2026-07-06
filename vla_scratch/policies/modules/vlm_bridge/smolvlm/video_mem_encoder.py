import math

import torch
import torch.nn as nn
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


class MEMVideoEncoder(nn.Module):
    def __init__(self, vision_encoder, mem_every_n_layers=4):
        super().__init__()
        object.__setattr__(self, "spatial_layers", vision_encoder.layers)
        self.mem_every_n_layers = mem_every_n_layers
        self.temporal_layer_indices = frozenset(
            i
            for i in range(len(self.spatial_layers))
            if (i + 1) % mem_every_n_layers == 0
        )
        self.hidden_size = vision_encoder.config.hidden_size

    def get_temporal_position_embedding(
        self, num_frames, hidden_size, device, dtype
    ):
        # current frame: t=0; past frames: t=-1,-2,...
        t = torch.arange(-num_frames + 1, 1, dtype=torch.float32, device=device)
        pe = torch.zeros(num_frames, hidden_size, dtype=torch.float32, device=device)
        position = t.unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32, device=device)
            * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.sin(position * div_term * 0.5)
        return pe.to(dtype)

    def forward(self, inputs_embeds, attention_mask=None, num_frames=1):
        hidden_states = inputs_embeds
        batch_times_frames, seq_len, hidden_dim = hidden_states.shape
        num_frames = int(num_frames)

        if batch_times_frames % num_frames != 0:
            raise ValueError(
                f"Batch size {batch_times_frames} is not divisible by "
                f"num_frames {num_frames}"
            )
        batch_size = batch_times_frames // num_frames
        device = hidden_states.device
        dtype = hidden_states.dtype

        use_temporal = num_frames > 1 and len(self.temporal_layer_indices) > 0
        if use_temporal:
            causal_mask = torch.ones(
                (1, 1, num_frames, num_frames), dtype=torch.bool, device=device
            ).tril()

            temp_pe = self.get_temporal_position_embedding(
                num_frames, hidden_dim, device, dtype
            )
            temp_pe = temp_pe.view(1, num_frames, 1, hidden_dim)
            temp_pe = temp_pe.expand(batch_size, num_frames, seq_len, hidden_dim)
            hidden_states = hidden_states + temp_pe.reshape(
                batch_times_frames, seq_len, hidden_dim
            )

        for layer_idx, spatial_layer in enumerate(self.spatial_layers):
            hidden_states = spatial_layer(hidden_states, attention_mask=attention_mask)

            if use_temporal and layer_idx in self.temporal_layer_indices:
                hidden_states = hidden_states.view(
                    batch_size, num_frames, seq_len, hidden_dim
                )
                hidden_states = hidden_states.transpose(1, 2)
                hidden_states = hidden_states.reshape(
                    batch_size * seq_len, num_frames, hidden_dim
                )
                hidden_states = spatial_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                )
                hidden_states = hidden_states.view(
                    batch_size, seq_len, num_frames, hidden_dim
                )
                hidden_states = hidden_states.transpose(1, 2)
                hidden_states = hidden_states.reshape(
                    batch_times_frames, seq_len, hidden_dim
                )

        hidden_states = hidden_states.view(
            batch_size, num_frames, seq_len, hidden_dim
        )
        return hidden_states[:, -1, :, :]


def encode_images_with_mem(causal_model, mem_encoder, pixel_values: torch.Tensor):
    if pixel_values.ndim != 6:
        raise ValueError(f"Expected 6D pixel_values for MEM, got {pixel_values.shape}")

    batch_size, num_images, num_frames, channels, height, width = pixel_values.shape
    pixel_values_flat = pixel_values.reshape(
        batch_size * num_images * num_frames,
        channels,
        height,
        width,
    )

    vision_model = causal_model.model.vision_model
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

    spatial_attention_mask = patch_attention_mask.view(pixel_values_flat.shape[0], -1)
    if vision_model.config._attn_implementation != "flash_attention_2":
        spatial_attention_mask = _prepare_4d_attention_mask(
            spatial_attention_mask,
            hidden_states.dtype,
        )
    elif not torch.any(~spatial_attention_mask):
        spatial_attention_mask = None

    hidden_states = mem_encoder(
        inputs_embeds=hidden_states,
        attention_mask=spatial_attention_mask,
        num_frames=num_frames,
    )
    last_hidden_state = vision_model.post_layernorm(hidden_states)
    return causal_model.model.connector(last_hidden_state)
