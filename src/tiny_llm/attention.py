import mlx.core as mx
from .basics import softmax, linear
import math
def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    shape=query.shape
    dim_l=shape[-2]
    dim_d=shape[-1]
    batch_size=math.prod(shape[:-2])
    if scale is None:
        scale = 1.0 / (dim_d ** 0.5)
    if mask is None:
        mask = mx.zeros((*shape[:-2], dim_l, dim_l), dtype=query.dtype)

    # flatten Q/K/V
    q_flat = query.reshape(batch_size, dim_l, dim_d)
    k_flat = key.reshape(batch_size, dim_l, dim_d)
    v_flat = value.reshape(batch_size, dim_l, dim_d)
    mask_flat = mask.reshape(batch_size, dim_l, dim_l)

    o_flat = mx.zeros((batch_size, dim_l, dim_d), dtype=query.dtype)

    # 遍历batch，计算每一个[l,d]*[l,d].transpose得到P矩阵l*l,然后对P矩阵进行缩放
    # 然后再softmax操作
    # 得到S矩阵，S*V得到O矩阵
    for i in range(batch_size):
        p_i=mx.matmul(q_flat[i],k_flat[i].T)
        p_i*=scale
        p_i+=mask_flat[i]
        s_i = mx.softmax(p_i, axis=-1)
        o_i=mx.matmul(s_i,v_flat[i])
        o_flat[i] = o_i
    output = o_flat.reshape(*shape)
    return output

class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        pass

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        pass


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
