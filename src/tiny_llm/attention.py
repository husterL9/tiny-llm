import mlx.core as mx
from .basics import softmax, linear
import math
# 这里的query、key、value应该分别对应的是Q、K、V
def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    # shape=(*BATCH_SIZE, DIM_L, DIM_D)
    shape=query.shape
    dim_l=shape[-2]
    dim_d=shape[-1]
    # batch_size=math.prod(shape[:-2])
    if scale is None:
        scale = 1.0 / (dim_d ** 0.5)
    scores=mx.matmul(query,mx.swapaxes(key, -1, -2))*scale
    if mask is None:
        pass
    else:
        scores+=mask
        
    # flatten Q/K/V
    # q_flat = query.reshape(batch_size, dim_l, dim_d)
    # k_flat = key.reshape(batch_size, dim_l, dim_d)
    # v_flat = value.reshape(batch_size, dim_l, dim_d)
    # mask_flat = mask.reshape(batch_size, dim_l, dim_l)
    # o_flat = mx.zeros((batch_size, dim_l, dim_d), dtype=query.dtype)

    # 遍历batch，计算每一个[l,d]*[l,d].transpose得到P矩阵l*l,然后对P矩阵进行缩放
    # 然后再softmax操作
    # 得到S矩阵，S*V得到O矩阵
    # for i in range(batch_size):
    #     p_i=mx.matmul(q_flat[i],k_flat[i].T)
    #     p_i*=scale
    #     p_i+=mask_flat[i]
    #     s_i = mx.softmax(p_i, axis=-1)
    #     o_i=mx.matmul(s_i,v_flat[i])
    #     o_flat[i] = o_i
    # output = o_flat.reshape(*shape)

    attention_map=mx.softmax(scores,-1)
    attention=mx.matmul(attention_map,value)
    return attention

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
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.wq=wq
        self.wk=wk
        self.wv=wv
        self.wo=wo
# 这里的query、key、value应该分别对应的是输入X
# API 设计成 mha(query, key, value)，是为了支持 Self-Attention 和 Cross-Attention。
# Self-Attention 只是其中一种特例，它把三者都设为同一个 X。
    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        Q=linear(query,self.wq)  # (N.., L, H*D) 
        K=linear(key,self.wk)
        V=linear(value,self.wv)
        head_dim = self.hidden_size //self.num_heads
        new_shape = Q.shape[:-1] + (self.num_heads, head_dim)
        Q = Q.reshape(new_shape)
        K = K.reshape(new_shape)
        V = V.reshape(new_shape)
        perm = list(range(len(Q.shape)))
        # 最后两个是 (-3=L), (-2=H), (-1=D)
        perm[-3], perm[-2] = perm[-2], perm[-3]
        Q = mx.transpose(Q, perm)
        K = mx.transpose(K, perm)
        V = mx.transpose(V, perm)
        if mask is not None:
            # 先加一个 batch 维度，再 broadcast
                mask_shape=mask.shape
                target_shape = Q.shape[:-2] + mask_shape  # (*batch, H, L, L)
                need_ones = len(target_shape) - mask.ndim
                pad_shape = (1,) * need_ones + mask.shape
                mask = mask.reshape(pad_shape)
                mask = mx.broadcast_to(mask, target_shape)
        O= scaled_dot_product_attention_simple(Q,K,V,mask=mask)
        print(O.shape)
        perm_O=list(range(len(O.shape)))
        perm_O[-3],perm_O[-2]=perm_O[-2],perm_O[-3]
        O=mx.transpose(O,perm_O)
        O=O.reshape(O.shape[:-2] + (head_dim * self.num_heads,))
        O=mx.matmul(O,mx.transpose(self.wo))
        return O



def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    offset = max(S - L, 0)

    i = mx.arange(L).reshape((L, 1))  # (L,1)
    j = mx.arange(S).reshape((1, S))  # (1,S)

    allowed = j <= (i + offset)       # (L,S) boolean
    neg_inf = mx.array(-mx.inf, dtype=dtype)
    zero = mx.array(0.0, dtype=dtype)

    return mx.where(allowed, zero, neg_inf)


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    # q_shape = (N.., H_q, L, D)
    # kv_shape = (N.., H, S, D)
    q_shape=query.shape
    H_q=q_shape[-3]
    L= q_shape[-2]
    D= q_shape[-1]
    kv_shape=key.shape
    H=kv_shape[-3]
    S= kv_shape[-2]
    prefix = q_shape[:-3]
    assert H_q % H == 0, f"H_kv must divide H_q, got H_q={H_q}, H_kv={H}"
    n_repeat=H_q//H
    q_grouped=query.reshape(*prefix,H,n_repeat,L,D)
    k_broad=key.reshape(*prefix,H,1,S,D)
    v_broad=value.reshape(*prefix,H,1,S,D)
    if mask is None:
        # 默认不加 mask（等价于加 0）
        pass
    elif isinstance(mask, str):
        if mask.lower() != "causal":
            raise ValueError("mask string only supports 'causal'")
        mask=causal_mask(L,S,dtype=query.dtype)
        mask = mask.reshape((1,) * (q_grouped.ndim - 2) + (L, S))
    else : 
        mask = mx.broadcast_to(mask, (*prefix, H_q, L, S))
        mask = mask.reshape(*prefix, H, n_repeat, L, S)
    attention=scaled_dot_product_attention_simple(q_grouped,k_broad,v_broad,scale,mask)
    attention=attention.reshape(*prefix,H*n_repeat,L,D)
    return attention


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass
