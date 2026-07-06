from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class VisionTokenMemoryConfig:
    enabled: bool = False
    max_tokens: int = 128
    selection: str = "even"
    integration: str = "context"
    candidate_tokens: int = 512
    token_drop_stride: int = 1
    token_per_image: int = 0
    add_pos_emb: bool = False


class VisionTokenMemory:
    def __init__(self):
        self.config = VisionTokenMemoryConfig()

    def configure(
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
    ) -> None:
        if selection not in {"even", "tokendrop"}:
            raise ValueError(
                "vision token memory selection must be 'even' or 'tokendrop', "
                f"got {selection!r}"
            )
        if integration not in {"context", "modulator", "expert"}:
            raise ValueError(
                "vision token memory integration must be 'context', "
                f"'modulator', or 'expert', got {integration!r}"
            )
        self.config = VisionTokenMemoryConfig(
            enabled=bool(enabled),
            max_tokens=int(max_tokens),
            selection=selection,
            integration=integration,
            candidate_tokens=int(candidate_tokens),
            token_drop_stride=max(1, int(token_drop_stride)),
            token_per_image=int(token_per_image),
            add_pos_emb=bool(add_pos_emb),
        )

    @staticmethod
    def even_indices(length: int, count: int, device: torch.device) -> torch.Tensor:
        if length <= 0 or count <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if length <= count:
            return torch.arange(length, dtype=torch.long, device=device)
        return torch.linspace(0, length - 1, count, device=device).round().long()

    @classmethod
    def limit_tokens_even(cls, tokens: torch.Tensor, max_tokens: int) -> torch.Tensor:
        if max_tokens <= 0 or tokens.shape[1] <= max_tokens:
            return tokens
        indices = cls.even_indices(tokens.shape[1], max_tokens, tokens.device)
        return tokens.index_select(1, indices)

    @staticmethod
    def pool_token_grid(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
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

        # z'_{i,j} = mean_{a,b < pool} z_{i*pool+a, j*pool+b}.
        grid = tokens.reshape(*prefix_shape, source_grid, source_grid, hidden)
        grid = grid.reshape(*prefix_shape, target_grid, pool, target_grid, pool, hidden)
        grid = grid.mean(dim=(-4, -2))
        return grid.reshape(*prefix_shape, target_tokens, hidden)

    @staticmethod
    def sinusoidal_scalar(values: torch.Tensor, dim: int) -> torch.Tensor:
        if dim <= 0:
            return values.new_empty(*values.shape, 0)
        half = max(1, dim // 2)
        freq = torch.linspace(0.0, 1.0, half, device=values.device, dtype=torch.float32)
        period = 10000.0 ** freq
        angles = values.to(torch.float32).unsqueeze(-1) / period
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if emb.shape[-1] < dim:
            emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
        return emb[..., :dim]

    @classmethod
    def position_embedding(
        cls,
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
            view = torch.arange(num_images, device=device, dtype=torch.float32)
            view = view / (num_images - 1)
        if grid > 1:
            y = y / (grid - 1)
            x = x / (grid - 1)

        yy, xx = torch.meshgrid(y, x, indexing="ij")
        tt = time[:, None, None, None].expand(hist_frames, num_images, grid, grid)
        vv = view[None, :, None, None].expand(hist_frames, num_images, grid, grid)
        yy = yy[None, None].expand(hist_frames, num_images, grid, grid)
        xx = xx[None, None].expand(hist_frames, num_images, grid, grid)
        pos = torch.cat(
            [
                cls.sinusoidal_scalar((tt + vv) * 0.5, t_dim),
                cls.sinusoidal_scalar(yy, y_dim),
                cls.sinusoidal_scalar(xx, x_dim),
            ],
            dim=-1,
        )
        pos = pos.reshape(1, hist_frames, num_images, tokens_per_image, hidden_size)
        return pos.expand(batch_size, -1, -1, -1, -1).to(dtype=dtype)

    @staticmethod
    def rgb_change_scores(
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
            batch_size, hist_frames, num_views, channels, grid, patch_h, grid, patch_w
        )
        patch_pixels = pixels.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(
            batch_size, hist_frames, num_views, tokens_per_image, -1
        )

        scores = torch.full(
            (batch_size, hist_frames, num_views, tokens_per_image),
            torch.finfo(history_values.dtype).min,
            dtype=history_values.dtype,
            device=history_values.device,
        )

        # s(t, v, p) = mean(|patch(t, v, p) - patch(t-stride, v, p)|).
        scores[:, 0] = 1000.0
        for frame_idx in range(stride, hist_frames, stride):
            prev_idx = max(0, frame_idx - stride)
            diff = (patch_pixels[:, frame_idx] - patch_pixels[:, prev_idx]).abs()
            diff = diff.mean(dim=-1)
            scores[:, frame_idx] = torch.where(
                diff >= 1e-4,
                diff,
                torch.full_like(diff, torch.finfo(diff.dtype).min),
            )
        return scores

    def encode(
        self,
        pixel_values: torch.Tensor,
        *,
        get_image_features: Callable[[torch.Tensor], torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        cfg = self.config
        if not cfg.enabled or pixel_values.ndim != 6:
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
            batch_size, hist_frames * num_images, channels, height, width
        )
        memory_tokens = get_image_features(encoder_values)
        max_tokens = cfg.max_tokens

        expected_images = batch_size * hist_frames * num_images
        if memory_tokens.ndim == 3 and memory_tokens.shape[0] == expected_images:
            memory_tokens = memory_tokens.reshape(
                batch_size,
                hist_frames * num_images,
                memory_tokens.shape[1],
                memory_tokens.shape[2],
            )
        elif memory_tokens.ndim != 4:
            memory_tokens = memory_tokens.reshape(
                batch_size, -1, memory_tokens.shape[-1]
            )
            memory_tokens = self.limit_tokens_even(memory_tokens, max_tokens)
            return memory_tokens.to(device=device, dtype=dtype)

        if cfg.token_per_image > 0:
            memory_tokens = self.pool_token_grid(memory_tokens, cfg.token_per_image)
        tokens_per_image = memory_tokens.shape[2]
        token_grid = memory_tokens.reshape(
            batch_size, hist_frames, num_images, tokens_per_image, memory_tokens.shape[-1]
        )
        if cfg.add_pos_emb:
            token_grid = token_grid + self.position_embedding(
                batch_size=batch_size,
                hist_frames=hist_frames,
                num_images=num_images,
                tokens_per_image=tokens_per_image,
                hidden_size=token_grid.shape[-1],
                device=token_grid.device,
                dtype=token_grid.dtype,
            )

        if cfg.selection == "tokendrop":
            memory_tokens = self._select_tokendrop(
                token_grid, history_values, tokens_per_image
            )
        else:
            memory_tokens = self._select_framesamp(token_grid, tokens_per_image)

        return memory_tokens.to(device=device, dtype=dtype)

    def _select_framesamp(
        self, token_grid: torch.Tensor, tokens_per_image: int
    ) -> torch.Tensor:
        cfg = self.config
        batch_size, hist_frames, num_images, _, hidden = token_grid.shape
        tokens_per_frame = num_images * tokens_per_image
        max_frames = hist_frames
        if cfg.max_tokens > 0:
            max_frames = max(1, cfg.max_tokens // max(1, tokens_per_frame))
        frame_indices = self.even_indices(hist_frames, max_frames, token_grid.device)
        tokens = token_grid.index_select(1, frame_indices)
        tokens = tokens.reshape(batch_size, -1, hidden)
        return self.limit_tokens_even(tokens, cfg.max_tokens)

    def _select_tokendrop(
        self,
        token_grid: torch.Tensor,
        history_values: torch.Tensor,
        tokens_per_image: int,
    ) -> torch.Tensor:
        cfg = self.config
        batch_size = token_grid.shape[0]
        flat_tokens = token_grid.reshape(batch_size, -1, token_grid.shape[-1])
        flat_scores = self.rgb_change_scores(
            history_values,
            tokens_per_image=tokens_per_image,
            stride=cfg.token_drop_stride,
        ).reshape(batch_size, -1)

        if cfg.candidate_tokens > 0 and flat_tokens.shape[1] > cfg.candidate_tokens:
            candidate = flat_scores.topk(cfg.candidate_tokens, dim=1).indices
            candidate_scores = flat_scores.gather(1, candidate)
            keep = min(cfg.max_tokens, candidate.shape[1]) if cfg.max_tokens > 0 else candidate.shape[1]
            selected = candidate.gather(1, candidate_scores.topk(keep, dim=1).indices)
        elif cfg.max_tokens > 0 and flat_tokens.shape[1] > cfg.max_tokens:
            selected = flat_scores.topk(cfg.max_tokens, dim=1).indices
        else:
            selected = torch.arange(
                flat_tokens.shape[1], device=flat_tokens.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)

        selected = selected.sort(dim=1).values
        gather_index = selected.unsqueeze(-1).expand(-1, -1, flat_tokens.shape[-1])
        return flat_tokens.gather(1, gather_index)
