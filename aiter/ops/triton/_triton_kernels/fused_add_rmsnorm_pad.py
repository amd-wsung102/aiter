import triton
import triton.language as tl


CONFIGS = [
    triton.Config({"BLOCK_SIZE_N": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE_N": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 512}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE_N": 1024}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_SIZE_N": 2048}, num_warps=8, num_stages=4),
]


@triton.jit
def _rmsmorm_op(row, weight, n_cols, epsilon):
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)
    rms_norm = row * norm_factor * weight
    return rms_norm


@triton.autotune(
    configs=CONFIGS,
    key=["N", "N_OUT"],
)
@triton.jit
def _fused_add_rmsnorm_pad(
    x_ptr, # input matrix
    res_ptr, # optional residual input matrix
    out_ptr, # output matrix
    res_out_ptr, # optional residual output matrix
    weight_ptr, # weight matrix
    eps, # epsilon
    M, # number of input rows
    N, # number of input columns
    N_OUT, # number of output columns
    x_stride_m, # row stride of input matrix
    x_stride_n, # column stride of input matrix
    res_stride_m, # row stride of residual input matrix
    res_stride_n, # column stride of residual input matrix
    out_stride_m, # row stride of output matrix
    out_stride_n, # column stride of output matrix
    res_out_stride_m, # row stride of residual output matrix
    res_out_stride_n, # column stride of residual output matrix
    HAS_RES: tl.constexpr, # compile time variable
    BLOCK_SIZE_N: tl.constexpr,
):
    tl.assume(x_stride_m > 0) # tells the Triton compiler to assume the condition is always true for optimization
    tl.assume(x_stride_n > 0)
    tl.assume(res_stride_m > 0)
    tl.assume(res_stride_n > 0)
    tl.assume(out_stride_m > 0)
    tl.assume(out_stride_n > 0)

    pid_m = tl.program_id(0) # int m = blockIdx.x; with one block per row
    tl.assume(pid_m >= 0)

    # tl.arange is vector of lanes inside the block
    n_offs = tl.arange(0, BLOCK_SIZE_N) # a vector of lane IDs, each lane ID is like threadIdx.x
                                        # and maps to one column index in the block (program)
    mask = n_offs < N # CUDA analogy: if (n < N) load/store; else use 0. This is used for tail handling.
    x = tl.load(
        x_ptr + pid_m * x_stride_m + n_offs * x_stride_n, # CUDA analogy: x[m * stride_m + n * stride_n]. This is like a strided 2D tensor
        mask=mask, # if n_offs < N, load this value into this col, else use 0.0
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    if HAS_RES: # compile time variable
        res = tl.load(
            res_ptr + pid_m * res_stride_m + n_offs * res_stride_n, # CUDA analogy: res[m * stride_m + n * stride_n]
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        x = x + res

    w = tl.load(
        weight_ptr + n_offs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    out = _rmsmorm_op(x, w, N, eps).to(out_ptr.dtype.element_ty)

    tl.store(
        out_ptr + pid_m * out_stride_m + n_offs * out_stride_n,
        out,
        mask=(n_offs < N_OUT),
    )
    if HAS_RES:
        tl.store(
            res_out_ptr + pid_m * res_out_stride_m + n_offs * res_out_stride_n,
            x,
            mask=mask,
        )
