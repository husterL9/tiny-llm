import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    wt = mx.transpose(w)
    out = mx.matmul(x, wt)
    if bias is not None:
        # bias: (O,)
        # 自动广播到 (..., O)
        out = out + bias
    return out


def silu(x: mx.array) -> mx.array:
    sigmoid_x=1/(1+mx.exp(-x))
    return x*sigmoid_x
