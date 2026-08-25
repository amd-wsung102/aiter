# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Hadamard / Random-Hadamard-Transform (RHT) helpers for FlyDSL MoE GEMM.

The FlyDSL stage1 fp4 epilogue can fuse a block-diagonal 32-point normalized
Walsh-Hadamard rotation (``fuse_rht=True``) over each MXFP4 scale block of the
SwiGLU output before fp4 quantization. This smooths outliers and reduces fp4
quantization error.

To keep the two-stage MoE numerically correct, the stage2 down-projection
weights ``W2`` (which contract over ``inter_dim``) must be pre-rotated by the
same 32-block Hadamard along ``inter_dim``. Because the normalized Hadamard
``H`` is orthonormal and symmetric (``H @ H^T = I``)::

    act' = act @ H              (fused in stage1)
    W2'  = W2  @ H              (offline, this module)
    act' @ W2'^T = act @ H @ H^T @ W2^T = act @ W2^T

so the low-precision GEMM operates on rotated (outlier-smoothed) operands while
the mathematical result is unchanged.

The kernel implements the natural-order (Sylvester) Hadamard via a separable
butterfly and multiplies by ``1/sqrt(32)``; the helpers below build the exact
same transform so offline weight rotation matches the fused activation rotation.
"""

from __future__ import annotations

import torch

__all__ = [
    "hadamard_matrix",
    "rotate_last_dim_hadamard",
    "rotate_moe_w2_hadamard",
    "rotate_activation_hadamard",
]

# MXFP4 scale block size; the fused RHT rotates within each 32-element block.
RHT_BLOCK = 32


def hadamard_matrix(
    n: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    normalized: bool = True,
) -> torch.Tensor:
    """Return the n x n natural-order (Sylvester) Walsh-Hadamard matrix.

    With ``normalized=True`` the matrix is scaled by ``1/sqrt(n)`` so it is
    orthonormal (matching the kernel's ``1/sqrt(32)`` normalization).
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard size must be a power of 2, got {n}")
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < n:
        top = torch.cat([h, h], dim=1)
        bot = torch.cat([h, -h], dim=1)
        h = torch.cat([top, bot], dim=0)
    if normalized:
        h = h / (float(n) ** 0.5)
    return h.to(device=device, dtype=dtype)


def rotate_last_dim_hadamard(x: torch.Tensor, block: int = RHT_BLOCK) -> torch.Tensor:
    """Apply the block-diagonal normalized Hadamard along the last dim of ``x``.

    The last dimension must be a multiple of ``block``. Each contiguous
    ``block``-element group ``g`` is transformed as ``y = H @ x`` where ``H`` is
    the normalized natural-order Hadamard (``H`` symmetric, so this matches the
    kernel's ``y_c = sum_{c'} H[c, c'] x_{c'}``). Computation is done in fp32 and
    cast back to the input dtype.
    """
    k = x.shape[-1]
    if k % block != 0:
        raise ValueError(f"last dim {k} not a multiple of block {block}")
    h = hadamard_matrix(block, device=x.device, dtype=torch.float32, normalized=True)
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    lead = xf.shape[:-1]
    xf = xf.reshape(*lead, k // block, block)
    # y[..., g, c] = sum_b H[c, b] * x[..., g, b]
    yf = torch.einsum("...gb,cb->...gc", xf, h)
    yf = yf.reshape(*lead, k)
    return yf.to(orig_dtype)


def rotate_moe_w2_hadamard(w2: torch.Tensor, block: int = RHT_BLOCK) -> torch.Tensor:
    """Pre-rotate stage2 W2 for a RHT-fused stage1.

    ``w2`` has shape ``[E, model_dim, inter_dim]`` and contracts over
    ``inter_dim`` (the last dim). Returns a rotated copy with the same shape and
    dtype, rotated by the same 32-block Hadamard the fused stage1 applies to the
    activation. Feed the returned weights through the normal fp4 quant +
    preshuffle path.
    """
    if w2.dim() != 3:
        raise ValueError(f"w2 must be [E, model_dim, inter_dim], got {tuple(w2.shape)}")
    return rotate_last_dim_hadamard(w2, block=block)


def rotate_activation_hadamard(
    act: torch.Tensor, block: int = RHT_BLOCK
) -> torch.Tensor:
    """Reference rotation of a stage1 activation ``[..., inter_dim]`` (torch ref).

    Mirrors the fused in-kernel RHT so a torch reference can replicate
    ``fp4_quant(RHT(SwiGLU(...)))``.
    """
    return rotate_last_dim_hadamard(act, block=block)
