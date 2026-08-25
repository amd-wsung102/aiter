# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end accuracy + performance comparison of RHT (fused Hadamard) through
the PRODUCTION fused_moe entry point (a4w4, fp4-intermediate Path A).

RHT only affects the fp4-quantized stage1 output (the intermediate activation
that feeds stage2), so it is only meaningful in the "fp4 intermediate" config
(KimiK3-style: fp4-fused stage1 + v2-layout stage2, skip_inter_quant). To force
that config we write a tiny tuned CSV whose kernelName1 carries the `_fp4`
suffix and kernelName2 is a `flydsl_moe2_layout_*` (v2) kernel, and point
AITER_CONFIG_FMOE at it BEFORE importing aiter.

For each token count we compare, against an fp32 high-precision reference:
  - baseline : fused_moe(...)                        (fp4 intermediate, no RHT)
  - RHT      : fused_moe(..., fuse_rht=True) + W2 pre-rotated by the same 32-block
               Hadamard (rotate_moe_w2_hadamard)

and report relative error + median latency.

Usage:
    python aiter/ops/flydsl/e2e_fused_moe_rht.py
    python aiter/ops/flydsl/e2e_fused_moe_rht.py -t 16 64 --model-dim 2048 --inter-dim 256 -E 64 -k 8
"""

import argparse
import os
import sys
import tempfile

import torch


def _write_tuned_csv(path, *, cu_num, tokens, model_dim, inter_dim, E, topk, block_m):
    header = (
        "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,"
        "q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,"
        "us2,kernelName2,err2,us,run_1stage,xbf16,flat,tflops,bw,_tag\n"
    )
    # fp4-fused stage1 (_fp4) + v2-layout stage2 (skip_inter_quant -> fp4 intermediate).
    kn1 = f"flydsl_moe1_afp4_wfp4_bf16_t{block_m}x128x256_w2_fp4"
    kn2 = f"flydsl_moe2_layout_afp4_wfp4_bf16_t{block_m}x128x128_atomic_sbm{block_m}"
    rows = [header]
    for tok in tokens:
        rows.append(
            f"gfx950,{cu_num},{tok},{model_dim},{inter_dim},{E},{topk},"
            f"ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,"
            f"torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,{block_m},0,"
            f"1.0,{kn1},0.0%,1.0,{kn2},0.0%,2.0,0,0,0,0,0,\n"
        )
    with open(path, "w") as f:
        f.writelines(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tokens", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--model-dim", type=int, default=2048)
    ap.add_argument("--inter-dim", type=int, default=256)
    ap.add_argument("-E", "--experts", type=int, default=64)
    ap.add_argument("-k", "--topk", type=int, default=8)
    ap.add_argument("--block-m", type=int, default=32)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--max-rel-err", type=float, default=0.35)
    args = ap.parse_args()

    # Must be set before importing aiter so the tuned CSV is loaded at import.
    csv_path = os.path.join(tempfile.gettempdir(), "e2e_rht_tuned_fmoe.csv")
    _write_tuned_csv(
        csv_path, cu_num=256, tokens=sorted(set(args.tokens)),
        model_dim=args.model_dim, inter_dim=args.inter_dim, E=args.experts,
        topk=args.topk, block_m=args.block_m,
    )
    os.environ["AITER_CONFIG_FMOE"] = csv_path

    import aiter
    import aiter.fused_moe as fm
    from aiter import QuantType, dtypes, ActivationType
    from aiter.fused_moe import fused_moe, fused_topk, torch_moe
    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility import fp4_utils
    from aiter.ops.flydsl.rht_utils import rotate_moe_w2_hadamard
    from aiter.ops.flydsl.utils import is_flydsl_available
    from aiter.test_common import run_perftest

    torch.set_default_device("cuda")

    if not is_flydsl_available():
        print("[SKIP] FlyDSL not available")
        sys.exit(0)

    # Detect which stage1 path fused_moe takes (want Path A = _flydsl_stage1_wrapper).
    path_seen = {"name": None}
    _orig_s1 = fm._flydsl_stage1_wrapper

    def _probe_s1(*a, **k):
        path_seen["name"] = "_flydsl_stage1_wrapper"
        return _orig_s1(*a, **k)

    fm._flydsl_stage1_wrapper = _probe_s1

    Q = QuantType.per_1x32
    QD = dtypes.fp4x2
    E, model_dim, inter_dim, topk, bm = (
        args.experts, args.model_dim, args.inter_dim, args.topk, args.block_m,
    )
    tq = aiter.get_torch_quant(Q)

    def prep_weights(w1_bf16, w2_bf16):
        w1_qt, w1_scale = tq(w1_bf16, quant_dtype=QD)
        w2_qt, w2_scale = tq(w2_bf16, quant_dtype=QD)
        w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
        w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
        w1s = shuffle_weight(w1_qt, (16, 16))
        w2s = shuffle_weight_a16w4(w2_qt, 16, False)
        w1sc = fp4_utils.e8m0_shuffle(w1_scale)
        w2sc = shuffle_scale_a16w4(w2_scale, E, False)
        w1s.is_shuffled = True
        w2s.is_shuffled = True
        return w1s, w1sc, w2s, w2sc

    def rel_err(ref, test):
        ref = ref.float(); test = test.float()
        return (test - ref).norm().item() / (ref.norm().item() + 1e-12)

    results = []
    for token in sorted(set(args.tokens)):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        inp = torch.randn((token, model_dim), dtype=torch.bfloat16) / 10
        w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=torch.bfloat16) / 10
        w2 = torch.randn((E, model_dim, inter_dim), dtype=torch.bfloat16) / 10
        score = torch.randn((token, E), dtype=torch.bfloat16)
        tw, tid = fused_topk(inp, score, topk, True)

        # fp32 high-precision reference (unquantized, unrotated).
        hp = torch_moe(inp, w1, w2, tw, tid, activation=ActivationType.Silu)

        # baseline fp4-intermediate weights + rotated-W2 weights.
        w1s, w1sc, w2s, w2sc = prep_weights(w1, w2)
        w2_rot = rotate_moe_w2_hadamard(w2, block=32)
        w1s_r, w1sc_r, w2s_r, w2sc_r = prep_weights(w1, w2_rot)

        def call(fuse_rht, w1s, w1sc, w2s, w2sc):
            return fused_moe(
                inp, w1s, w2s, tw, tid, quant_type=Q,
                w1_scale=w1sc, w2_scale=w2sc,
                activation=ActivationType.Silu, fuse_rht=fuse_rht,
            )

        path_seen["name"] = None
        out_base = call(False, w1s, w1sc, w2s, w2sc)
        base_path = path_seen["name"]
        out_rht = call(True, w1s_r, w1sc_r, w2s_r, w2sc_r)
        torch.cuda.synchronize()

        err_base = rel_err(hp, out_base)
        err_rht = rel_err(hp, out_rht)

        _, us_base = run_perftest(
            lambda: call(False, w1s, w1sc, w2s, w2sc),
            num_iters=args.iters, num_warmup=args.warmup, use_cuda_event=True,
        )
        _, us_rht = run_perftest(
            lambda: call(True, w1s_r, w1sc_r, w2s_r, w2sc_r),
            num_iters=args.iters, num_warmup=args.warmup, use_cuda_event=True,
        )

        results.append((token, base_path, err_base, err_rht, us_base, us_rht))
        print(f"\n{'='*74}")
        print(f"token={token} dim=({model_dim},{inter_dim}) E={E} topk={topk} "
              f"path={base_path} nan={out_base.isnan().any().item()}")
        print(f"  baseline fp4  : rel_err={err_base:.4f}  {us_base:8.2f} us")
        print(f"  RHT      fp4  : rel_err={err_rht:.4f}  {us_rht:8.2f} us  "
              f"(err x{err_rht/(err_base+1e-12):.3f}, lat x{us_rht/(us_base+1e-12):.3f})")

    print(f"\n{'='*74}\nSUMMARY (fused_moe a4w4, fp4 intermediate)\n{'='*74}")
    ok = True
    for token, bp, eb, er, ub, ur in results:
        pth = "PathA" if bp == "_flydsl_stage1_wrapper" else str(bp)
        good = (bp == "_flydsl_stage1_wrapper") and (er <= args.max_rel_err)
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'} t{token:<5d} {pth:<20s} "
              f"err_base={eb:.4f} err_rht={er:.4f} ratio={er/(eb+1e-12):.3f} "
              f"us_base={ub:.2f} us_rht={ur:.2f}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
