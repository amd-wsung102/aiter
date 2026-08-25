# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel-level before/after perf harness for the two FlyDSL MoE GEMM
optimizations (a4w4):

  1. Fused RHT (Hadamard) in the stage1 fp4 epilogue  -> stage1 (+/- fuse_rht)
  2. Work-stealing (padding-skipping persistent) scheduler -> stage1 & stage2
     (static vs work_steal)

Each kernel is timed in isolation with aiter.test_common.run_perftest (CUDA
events). Reports median us and speedup for decode-like and prefill-like token
counts.

Usage:
    python aiter/ops/flydsl/bench_flydsl_moe_opt.py
    python aiter/ops/flydsl/bench_flydsl_moe_opt.py --model-dim 7168 -E 256 -k 8 \
        --decode 16 32 64 --prefill 1024
"""

import argparse
import sys

import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight, shuffle_weight_a16w4
from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
from aiter.ops.flydsl.rht_utils import rotate_moe_w2_hadamard
from aiter.test_common import run_perftest

torch.set_default_device("cuda")
Q_TYPE = QuantType.per_1x32
Q_DTYPE = dtypes.fp4x2


def _build(token, model_dim, inter_dim, E, topk, block_m, dtype=torch.bfloat16):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    tq = aiter.get_torch_quant(Q_TYPE)
    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    score = torch.randn((token, E), dtype=dtype)
    tw, tid = fused_topk(inp, score, topk, True)

    a1_qt, a1_scale = tq(inp, quant_dtype=Q_DTYPE)
    w1_qt, w1_scale = tq(w1, quant_dtype=Q_DTYPE)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    # stage1 fp4 reference activation to build a realistic stage2 input.
    ref1 = torch_moe_stage1(
        a1_qt, w1_qt, w2.view(E, model_dim, inter_dim), tw, tid, dtype=dtype,
        activation=ActivationType.Silu, quant_type=Q_TYPE,
        a1_scale=a1_scale, w1_scale=w1_scale, doweight=False,
    )
    a2_qt, a2_scale = tq(ref1, quant_dtype=Q_DTYPE)
    a2_qt = a2_qt.view(token, topk, -1)

    w2_qt, w2_scale = tq(w2, quant_dtype=Q_DTYPE)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    w2_rot = rotate_moe_w2_hadamard(w2, block=32)
    w2r_qt, w2r_scale = tq(w2_rot, quant_dtype=Q_DTYPE)
    w2r_qt = w2r_qt.view(E, model_dim, inter_dim // 2)

    sids, sw, seids, nvi, _ = moe_sorting(tid, tw, E, model_dim, dtype, block_m)
    a1s_sort = moe_mxfp4_sort(a1_scale[:token].view(token, 1, -1), sorted_ids=sids,
                              num_valid_ids=nvi, token_num=token, block_size=block_m)
    a2s_sort = moe_mxfp4_sort(a2_scale[: token * topk].view(token, topk, -1),
                              sorted_ids=sids, num_valid_ids=nvi, token_num=token,
                              block_size=block_m)
    return dict(
        a1_qt=a1_qt, a1s_sort=a1s_sort, a2_qt=a2_qt, a2s_sort=a2s_sort,
        w1_qt_shuf=shuffle_weight(w1_qt, (16, 16)), w1_scale_shuf=e8m0_shuffle(w1_scale),
        w2_qt_shuf=shuffle_weight_a16w4(w2_qt, 16, False),
        w2_scale_shuf=shuffle_scale_a16w4(w2_scale, E, False),
        w2r_qt_shuf=shuffle_weight_a16w4(w2r_qt, 16, False),
        w2r_scale_shuf=shuffle_scale_a16w4(w2r_scale, E, False),
        sids=sids, seids=seids, nvi=nvi, sw=sw, topk=topk,
    )


def _time(fn, iters, warmup):
    # use_cuda_event=True times with CUDA events directly and avoids the
    # ROCTracer/profiler path (which can be flaky across many kernels).
    _, us = run_perftest(
        fn, num_iters=iters, num_warmup=warmup, use_cuda_event=True
    )
    return us


def bench(token, model_dim, inter_dim, E, topk, block_m, iters, warmup):
    d = _build(token, model_dim, inter_dim, E, topk, block_m)
    tn, tk = 256, 256

    def s1(out_dtype, fuse_rht=False, work_steal=False, w2=False):
        return lambda: flydsl_moe_stage1(
            a=d["a1_qt"], w1=d["w1_qt_shuf"], sorted_token_ids=d["sids"],
            sorted_expert_ids=d["seids"], num_valid_ids=d["nvi"], topk=topk,
            tile_m=block_m, tile_n=tn, tile_k=tk, a_dtype="fp4", b_dtype="fp4",
            out_dtype=out_dtype, w1_scale=d["w1_scale_shuf"], a1_scale=d["a1s_sort"],
            sorted_weights=None, fuse_rht=fuse_rht, work_steal=work_steal,
        )

    def s2(work_steal=False, rot=False):
        w2q = d["w2r_qt_shuf"] if rot else d["w2_qt_shuf"]
        w2s = d["w2r_scale_shuf"] if rot else d["w2_scale_shuf"]
        return lambda: flydsl_moe_stage2(
            inter_states=d["a2_qt"], w2=w2q, sorted_token_ids=d["sids"],
            sorted_expert_ids=d["seids"], num_valid_ids=d["nvi"], topk=topk,
            tile_m=block_m, tile_n=tn, tile_k=tk, a_dtype="fp4", b_dtype="fp4",
            out_dtype="bf16", mode="atomic", w2_scale=w2s, a2_scale=d["a2s_sort"],
            sorted_weights=d["sw"], work_steal=work_steal,
        )

    # Warm compile each variant once (excluded from timing).
    variants = {
        "s1_fp4_base":  s1("fp4", fuse_rht=False, work_steal=False),
        "s1_fp4_rht":   s1("fp4", fuse_rht=True, work_steal=False),
        "s1_fp4_ws":    s1("fp4", fuse_rht=False, work_steal=True),
        "s1_fp4_rht_ws": s1("fp4", fuse_rht=True, work_steal=True),
        "s2_base":      s2(work_steal=False),
        "s2_ws":        s2(work_steal=True),
    }
    for fn in variants.values():
        fn()
    torch.cuda.synchronize()

    us = {k: _time(fn, iters, warmup) for k, fn in variants.items()}

    def spd(base, opt):
        return us[base] / us[opt] if us[opt] > 0 else float("nan")

    print(f"\n{'='*78}")
    print(f"token={token} model_dim={model_dim} inter_dim={inter_dim} E={E} topk={topk} bm={block_m}")
    print(f"{'-'*78}")
    print(f"  stage1 fp4  baseline        : {us['s1_fp4_base']:8.2f} us")
    print(f"  stage1 fp4  +RHT            : {us['s1_fp4_rht']:8.2f} us   "
          f"(RHT overhead x{us['s1_fp4_rht']/us['s1_fp4_base']:.3f})")
    print(f"  stage1 fp4  +work_steal     : {us['s1_fp4_ws']:8.2f} us   "
          f"(speedup x{spd('s1_fp4_base','s1_fp4_ws'):.3f})")
    print(f"  stage1 fp4  +RHT+work_steal : {us['s1_fp4_rht_ws']:8.2f} us   "
          f"(speedup vs base x{spd('s1_fp4_base','s1_fp4_rht_ws'):.3f})")
    print(f"  stage2      baseline        : {us['s2_base']:8.2f} us")
    print(f"  stage2      +work_steal     : {us['s2_ws']:8.2f} us   "
          f"(speedup x{spd('s2_base','s2_ws'):.3f})")
    return token, us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dim", type=int, default=4096)
    ap.add_argument("--inter-dim", type=int, default=256)
    ap.add_argument("-E", "--experts", type=int, default=256)
    ap.add_argument("-k", "--topk", type=int, default=8)
    ap.add_argument("--block-m", type=int, default=32)
    ap.add_argument("--decode", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--prefill", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        print("[SKIP] FlyDSL not available")
        sys.exit(0)

    rows = []
    for token in list(args.decode) + list(args.prefill):
        try:
            rows.append(bench(token, args.model_dim, args.inter_dim, args.experts,
                              args.topk, args.block_m, args.iters, args.warmup))
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
        finally:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    print(f"\n{'='*78}\nDONE ({len(rows)} shapes)\n{'='*78}")


if __name__ == "__main__":
    main()
