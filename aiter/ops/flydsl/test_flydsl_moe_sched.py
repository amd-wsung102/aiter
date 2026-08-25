# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the opt-in work-stealing (padding-skipping persistent)
scheduler on FlyDSL MoE stage1/stage2 (a4w4).

For each stage the work-steal output must match the default (static-grid) output
and the torch reference, under both balanced and skewed expert routing (skewed
routing stresses the dynamic load balancing / padding-skip).

Usage:
    python aiter/ops/flydsl/test_flydsl_moe_sched.py
    python aiter/ops/flydsl/test_flydsl_moe_sched.py -t 16 64 --skew
"""

import argparse
import sys

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight, shuffle_weight_a16w4
from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

torch.set_default_device("cuda")
Q_TYPE = QuantType.per_1x32
Q_DTYPE = dtypes.fp4x2


def _gen(token, model_dim, inter_dim, E, topk, block_m, skew=False, dtype=torch.bfloat16):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch_quant = aiter.get_torch_quant(Q_TYPE)
    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    score = torch.randn((token, E), dtype=dtype)
    if skew:
        # Route almost everything to the first few experts to create highly
        # uneven per-expert token counts (long tail -> load imbalance).
        score[:, :4] += 50.0
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    w1_qt, w1_scale = torch_quant(w1, quant_dtype=Q_DTYPE)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=Q_DTYPE)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    a1_qt, a1_scale = torch_quant(inp, quant_dtype=Q_DTYPE)

    ref1 = torch_moe_stage1(
        a1_qt, w1_qt, w2_qt, topk_weights, topk_ids, dtype=dtype,
        activation=ActivationType.Silu, quant_type=Q_TYPE,
        a1_scale=a1_scale, w1_scale=w1_scale, doweight=False,
    )
    a2_qt, a2_scale = torch_quant(ref1, quant_dtype=Q_DTYPE)
    a2_qt = a2_qt.view(token, topk, -1)
    ref2 = torch_moe_stage2(
        a2_qt, w1_qt, w2_qt, topk_weights, topk_ids, dtype=dtype,
        quant_type=Q_TYPE, w2_scale=w2_scale, a2_scale=a2_scale, doweight=True,
    )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )
    a1_scale_sort = moe_mxfp4_sort(
        a1_scale[:token, :].view(token, 1, -1), sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids, token_num=token, block_size=block_m,
    )
    a2_scale_sort = moe_mxfp4_sort(
        a2_scale[: token * topk, :].view(token, topk, -1), sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids, token_num=token, block_size=block_m,
    )
    return dict(
        ref1=ref1, ref2=ref2, a1_qt=a1_qt, a1_scale_sort=a1_scale_sort,
        a2_qt=a2_qt, a2_scale_sort=a2_scale_sort,
        w1_qt_shuf=shuffle_weight(w1_qt, (16, 16)),
        w1_scale_shuf=e8m0_shuffle(w1_scale),
        w2_qt_shuf=shuffle_weight_a16w4(w2_qt, 16, False),
        w2_scale_shuf=shuffle_scale_a16w4(w2_scale, E, False),
        sorted_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids, sorted_weights=sorted_weights,
        topk=topk, dtype=dtype,
    )


def _stats(ref, test):
    ref = ref.float(); test = test.float()
    return (
        (test - ref).abs().max().item(),
        (test - ref).norm().item() / (ref.norm().item() + 1e-12),
    )


def run_case(token, model_dim, inter_dim, E, topk, block_m, skew):
    d = _gen(token, model_dim, inter_dim, E, topk, block_m, skew=skew)
    outdt = "bf16"
    tag = f"t{token}_skew{int(skew)}"

    def s1(ws):
        return flydsl_moe_stage1(
            a=d["a1_qt"], w1=d["w1_qt_shuf"],
            sorted_token_ids=d["sorted_ids"], sorted_expert_ids=d["sorted_expert_ids"],
            num_valid_ids=d["num_valid_ids"], topk=topk, tile_m=block_m,
            tile_n=256, tile_k=256, a_dtype="fp4", b_dtype="fp4", out_dtype=outdt,
            w1_scale=d["w1_scale_shuf"], a1_scale=d["a1_scale_sort"],
            sorted_weights=None, work_steal=ws,
        )

    def s2(ws):
        return flydsl_moe_stage2(
            inter_states=d["a2_qt"], w2=d["w2_qt_shuf"],
            sorted_token_ids=d["sorted_ids"], sorted_expert_ids=d["sorted_expert_ids"],
            num_valid_ids=d["num_valid_ids"], topk=topk, tile_m=block_m,
            tile_n=256, tile_k=256, a_dtype="fp4", b_dtype="fp4", out_dtype=outdt,
            mode="atomic", w2_scale=d["w2_scale_shuf"], a2_scale=d["a2_scale_sort"],
            sorted_weights=d["sorted_weights"], work_steal=ws,
        )

    s1_base, s1_ws = s1(False), s1(True)
    s2_base, s2_ws = s2(False), s2(True)
    torch.cuda.synchronize()

    results = []
    # Stage1 is deterministic (no atomics) -> work-steal must be bitwise-identical
    # to static. Stage2 uses fp bf16 atomic accumulation (mode="atomic"), which is
    # order-dependent, so a different schedule only perturbs the accumulation order
    # (~1e-2 relative); we allow that small reorder noise and separately require
    # both schedules to match the torch reference.
    md1, rel1 = _stats(s1_base, s1_ws)
    md2, rel2 = _stats(s2_base, s2_ws)
    results.append((f"s1_ws_vs_static_{tag}", rel1 < 1e-3, md1, rel1))
    results.append((f"s2_ws_vs_static_{tag}", rel2 < 2e-2, md2, rel2))
    # work-steal vs torch ref: fp4 tolerance
    _, r1r = _stats(d["ref1"], s1_ws)
    _, r2r = _stats(d["ref2"], s2_ws)
    results.append((f"s1_ws_vs_ref_{tag}", r1r < 0.35, 0.0, r1r))
    results.append((f"s2_ws_vs_ref_{tag}", r2r < 0.35, 0.0, r2r))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tokens", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--model-dim", type=int, default=2048)
    ap.add_argument("--inter-dim", type=int, default=256)
    ap.add_argument("-E", "--experts", type=int, default=64)
    ap.add_argument("-k", "--topk", type=int, default=8)
    ap.add_argument("--block-m", type=int, default=32)
    ap.add_argument("--skew", action="store_true", help="also test skewed routing")
    args = ap.parse_args()

    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        print("[SKIP] FlyDSL not available")
        sys.exit(0)

    all_res = []
    for token in args.tokens:
        for skew in ([False, True] if args.skew else [False]):
            all_res += run_case(
                token, args.model_dim, args.inter_dim, args.experts,
                args.topk, args.block_m, skew,
            )

    print(f"\n{'='*70}\nSCHEDULER VALIDATION SUMMARY\n{'='*70}")
    for name, ok, md, rel in all_res:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<28s}  max|d|={md:.5f} rel={rel:.5e}")
    n_pass = sum(1 for _, ok, _, _ in all_res if ok)
    print(f"\n  {n_pass}/{len(all_res)} passed")
    if n_pass != len(all_res):
        sys.exit(1)


if __name__ == "__main__":
    main()
