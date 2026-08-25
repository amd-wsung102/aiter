# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the fused RHT (Hadamard) stage1 fp4 epilogue (a4w4).

Validates that fusing a block-diagonal 32-point Walsh-Hadamard rotation into the
stage1 fp4 quant epilogue (``fuse_rht=True``), together with an offline
Hadamard rotation of the stage2 W2 weights, preserves the two-stage MoE result
and does not increase (ideally reduces) fp4 quantization error vs a
high-precision reference.

Flow per case:
  HP reference : torch_moe(inp, w1, w2)                      (fp32 compute)
  baseline fp4 : stage1(fp4)         -> stage2(w2 fp4)       (no rotation)
  RHT      fp4 : stage1(fp4, rht)    -> stage2(rot(w2) fp4)  (rotation)

Both fp4 paths are compared to the HP reference; we assert the RHT path stays
within tolerance and report err_rht vs err_base.

Usage:
    python aiter/ops/flydsl/test_flydsl_moe_rht.py
    python aiter/ops/flydsl/test_flydsl_moe_rht.py -t 16 64 --outlier
"""

import argparse
import sys

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight, shuffle_weight_a16w4
from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.ops.flydsl.rht_utils import rotate_moe_w2_hadamard

torch.set_default_device("cuda")

Q_TYPE = QuantType.per_1x32
Q_DTYPE_A = dtypes.fp4x2
Q_DTYPE_W = dtypes.fp4x2


def _rel_err(ref: torch.Tensor, test: torch.Tensor) -> float:
    ref = ref.float()
    test = test.float()
    return (
        (test - ref).norm().item() / (ref.norm().item() + 1e-12)
    )


def _cos(ref: torch.Tensor, test: torch.Tensor) -> float:
    ref = ref.float().reshape(-1)
    test = test.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(ref, test, dim=0).item()


def _prep_w2_fp4(w2: torch.Tensor, E: int):
    """Quantize + preshuffle a bf16 w2 [E, model_dim, inter_dim] for stage2."""
    torch_quant = aiter.get_torch_quant(Q_TYPE)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=Q_DTYPE_W)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)
    w2_qt_shuf = shuffle_weight_a16w4(w2_qt, 16, False)
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, E, False)
    return w2_qt_shuf, w2_scale_shuf


def run_rht_e2e(
    token: int = 16,
    model_dim: int = 7168,
    inter_dim: int = 256,
    E: int = 256,
    topk: int = 8,
    block_m: int = 32,
    outlier: bool = False,
    dtype=torch.bfloat16,
    verbose: bool = True,
):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch_quant = aiter.get_torch_quant(Q_TYPE)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    if outlier:
        # Inject a few large-magnitude columns in w2 (contraction dim) to create
        # activation/weight outliers that RHT is designed to smooth.
        oc = torch.randint(0, inter_dim, (max(1, inter_dim // 32),))
        w2[:, :, oc] *= 8.0
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    # High-precision reference (fp32 compute, unquantized weights).
    hp = torch_moe(inp, w1, w2, topk_weights, topk_ids, activation=ActivationType.Silu)

    # Quantize stage1 inputs/weights (shared by both fp4 paths).
    a1_qt, a1_scale = torch_quant(inp, quant_dtype=Q_DTYPE_A)
    w1_qt, w1_scale = torch_quant(w1, quant_dtype=Q_DTYPE_W)
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w1_qt_shuf = shuffle_weight(w1_qt, (16, 16))
    w1_scale_shuf = e8m0_shuffle(w1_scale)

    # Stage2 weights: baseline (unrotated) and RHT (rotated) fp4.
    w2_base_shuf, w2_base_scale_shuf = _prep_w2_fp4(w2, E)
    w2_rot = rotate_moe_w2_hadamard(w2, block=32)
    w2_rot_shuf, w2_rot_scale_shuf = _prep_w2_fp4(w2_rot, E)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )
    a1_scale_sort = moe_mxfp4_sort(
        a1_scale[:token, :].view(token, 1, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        block_size=block_m,
    )

    out_dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"

    def run_path(fuse_rht, w2_shuf, w2_scale_shuf):
        a2, a2_scale = flydsl_moe_stage1(
            a=a1_qt,
            w1=w1_qt_shuf,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            topk=topk,
            tile_m=block_m,
            tile_n=256,
            tile_k=256,
            a_dtype="fp4",
            b_dtype="fp4",
            out_dtype="fp4",
            w1_scale=w1_scale_shuf,
            a1_scale=a1_scale_sort,
            sorted_weights=None,
            fuse_rht=fuse_rht,
        )
        out = flydsl_moe_stage2(
            inter_states=a2,
            w2=w2_shuf,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            topk=topk,
            tile_m=block_m,
            tile_n=256,
            tile_k=256,
            a_dtype="fp4",
            b_dtype="fp4",
            out_dtype=out_dtype_str,
            mode="atomic",
            w2_scale=w2_scale_shuf,
            a2_scale=a2_scale,
            sorted_weights=sorted_weights,
        )
        torch.cuda.synchronize()
        return out

    out_base = run_path(False, w2_base_shuf, w2_base_scale_shuf)
    out_rht = run_path(True, w2_rot_shuf, w2_rot_scale_shuf)

    err_base = _rel_err(hp, out_base)
    err_rht = _rel_err(hp, out_rht)
    cos_base = _cos(hp, out_base)
    cos_rht = _cos(hp, out_rht)

    if verbose:
        print(f"\n{'='*70}")
        print(
            f"[RHT E2E] token={token} dim=({model_dim},{inter_dim}) E={E} "
            f"topk={topk} bm={block_m} outlier={outlier}"
        )
        print(f"{'='*70}")
        print(f"  baseline fp4 : rel_err={err_base:.4f}  cos={cos_base:.4f}")
        print(f"  RHT      fp4 : rel_err={err_rht:.4f}  cos={cos_rht:.4f}")
        print(
            f"  err_rht/err_base = {err_rht / (err_base + 1e-12):.3f} "
            f"({'better' if err_rht <= err_base else 'worse'})"
        )

    return {
        "err_base": err_base,
        "err_rht": err_rht,
        "cos_base": cos_base,
        "cos_rht": cos_rht,
    }


def main():
    parser = argparse.ArgumentParser(description="FlyDSL MoE RHT (Hadamard) fp4 tests")
    parser.add_argument("-t", "--tokens", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--model-dim", type=int, default=2048)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("-E", "--experts", type=int, default=32)
    parser.add_argument("-k", "--topk", type=int, default=4)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--outlier", action="store_true")
    # RHT correctness: rel err must be below this vs the fp32 HP reference.
    # fp4 (e2m1) has ~1 mantissa bit, so ~0.3 relative error is expected; the
    # --outlier stress case (deliberate 8x outliers) sits a bit higher.
    parser.add_argument("--max-rel-err", type=float, default=0.35)
    # RHT must not be materially worse than the unrotated fp4 baseline.
    parser.add_argument("--max-ratio", type=float, default=1.10)
    args = parser.parse_args()

    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        print("[SKIP] FlyDSL not available")
        sys.exit(0)

    results = []
    for token in args.tokens:
        for outlier in ([False, True] if args.outlier else [False]):
            m = run_rht_e2e(
                token=token,
                model_dim=args.model_dim,
                inter_dim=args.inter_dim,
                E=args.experts,
                topk=args.topk,
                block_m=args.block_m,
                outlier=outlier,
            )
            ratio = m["err_rht"] / (m["err_base"] + 1e-12)
            passed = (m["err_rht"] <= args.max_rel_err) and (ratio <= args.max_ratio)
            results.append(
                (f"rht_e2e_t{token}_ol{int(outlier)}", passed, m, ratio)
            )

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for name, passed, m, ratio in results:
        print(
            f"  {'PASS' if passed else 'FAIL'}  {name:<24s}  "
            f"err_base={m['err_base']:.4f} err_rht={m['err_rht']:.4f} "
            f"ratio={ratio:.3f}"
        )
    n_pass = sum(1 for _, p, _, _ in results if p)
    print(f"\n  {n_pass}/{len(results)} passed")
    if n_pass != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
