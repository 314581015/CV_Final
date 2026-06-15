"""Lightweight C3 + Transformer block for Ultralytics YOLO models."""

from __future__ import annotations

import torch
from torch import nn


def _make_heads(channels: int, requested_heads: int) -> int:
    """Return a valid attention-head count that divides channels."""
    requested_heads = max(1, min(requested_heads, channels))
    for heads in range(requested_heads, 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class LiteTransformerBlock(nn.Module):
    """Small transformer encoder block applied on a feature map."""

    def __init__(self, channels: int, heads: int = 4, mlp_ratio: float = 1.5, dropout: float = 0.0):
        super().__init__()
        heads = _make_heads(channels, heads)
        hidden = max(channels, int(channels * mlp_ratio))

        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        norm_tokens = self.norm1(tokens)
        tokens = tokens + self.attn(norm_tokens, norm_tokens, norm_tokens, need_weights=False)[0]
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class C3TRLite(nn.Module):
    """C2f-style block with lightweight transformer layers in the inner path."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        heads: int = 4,
        expansion: float = 0.5,
        dropout: float = 0.0,
    ):
        super().__init__()
        from ultralytics.nn.modules import Conv

        self.c = max(1, int(c2 * expansion))
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(LiteTransformerBlock(self.c, heads=heads, dropout=dropout) for _ in range(n))
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(block(y[-1]) for block in self.m)
        out = self.cv2(torch.cat(y, 1))
        return out + x if self.shortcut else out


def infer_c2f_shape(module: nn.Module) -> tuple[int, int, int] | None:
    """Infer c1, c2, and repeat count from a C2f-like Ultralytics module."""
    if not (hasattr(module, "cv1") and hasattr(module, "cv2") and hasattr(module, "m")):
        return None

    cv1_conv = getattr(module.cv1, "conv", None)
    cv2_conv = getattr(module.cv2, "conv", None)
    if cv1_conv is None or cv2_conv is None:
        return None

    if not hasattr(cv1_conv, "in_channels") or not hasattr(cv2_conv, "out_channels"):
        return None

    repeats = len(module.m) if hasattr(module.m, "__len__") else 1
    return int(cv1_conv.in_channels), int(cv2_conv.out_channels), max(1, repeats)


def replace_late_c2f_with_c3tr_lite(
    model: nn.Module,
    num_blocks: int = 1,
    heads: int = 4,
    expansion: float = 0.5,
    dropout: float = 0.0,
) -> list[str]:
    """Replace the last N C2f-like blocks in an Ultralytics model."""
    candidates: list[tuple[str, nn.Module, str, nn.Module]] = []

    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if child.__class__.__name__ in {"C2f", "C3", "C2fCIB"} and infer_c2f_shape(child):
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                candidates.append((full_name, parent, child_name, child))

    if not candidates:
        raise RuntimeError("No replaceable C2f/C3/C2fCIB blocks were found in this model.")

    selected = candidates[-num_blocks:]
    replaced: list[str] = []
    for full_name, parent, child_name, child in selected:
        c1, c2, repeats = infer_c2f_shape(child)  # type: ignore[misc]
        new_block = C3TRLite(c1, c2, n=repeats, shortcut=False, heads=heads, expansion=expansion, dropout=dropout)
        for attr in ("i", "f", "type", "np"):
            if hasattr(child, attr):
                setattr(new_block, attr, getattr(child, attr))
        setattr(parent, child_name, new_block)
        replaced.append(f"{full_name}: {child.__class__.__name__}({c1}->{c2}, n={repeats}) -> C3TRLite")

    return replaced


def replace_indexed_c2f_with_c3tr_lite(
    model: nn.Module,
    target_indices: list[int],
    heads: int = 4,
    expansion: float = 0.5,
    dropout: float = 0.0,
) -> list[str]:
    """Replace specific top-level YOLO layer indices with C3TR-lite."""
    target_names = {str(i) for i in target_indices}
    replaced: list[str] = []

    if not hasattr(model, "model"):
        raise RuntimeError("Expected an Ultralytics model with a top-level 'model' Sequential.")

    for child_name, child in model.model.named_children():
        if child_name not in target_names:
            continue

        shape = infer_c2f_shape(child)
        if child.__class__.__name__ not in {"C2f", "C3", "C2fCIB"} or not shape:
            raise RuntimeError(
                f"Layer model.{child_name} is {child.__class__.__name__}, not a replaceable C2f/C3/C2fCIB block."
            )

        c1, c2, repeats = shape
        new_block = C3TRLite(c1, c2, n=repeats, shortcut=False, heads=heads, expansion=expansion, dropout=dropout)
        for attr in ("i", "f", "type", "np"):
            if hasattr(child, attr):
                setattr(new_block, attr, getattr(child, attr))
        setattr(model.model, child_name, new_block)
        replaced.append(f"model.{child_name}: {child.__class__.__name__}({c1}->{c2}, n={repeats}) -> C3TRLite")

    replaced_indices = {line.split(":")[0].split(".")[1] for line in replaced}
    missing = target_names - replaced_indices
    if missing:
        raise RuntimeError(f"Requested C3TR-lite target layers were not replaced: {sorted(missing)}")

    return replaced
