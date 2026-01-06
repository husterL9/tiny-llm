import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.num_kv_heads=num_kv_heads
        self.wq=wq
        self.wk=wk
        self.wv=wv
        self.wo=wo
        self.bq=bq
        self.bk=bk
        self.bv=bv
        self.max_seq_len=max_seq_len
        self.theta=theta

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # Q:(B,L,E)
        Q=linear(x,self.wq,self.bq)
        K=linear(x,self.wk,self.bk)
        V=linear(x,self.wv,self.bv)
        D=self.hidden_size//self.num_heads
        q_shape=Q.shape
        k_shape=K.shape
        E=q_shape[-1]
        L=q_shape[-2]
        S=k_shape[-2]
        prefix=q_shape[:-2]
        assert E % self.num_heads == 0, f"E={E} must be divisible by h={self.num_heads}"
        # (B, L, h, d)
        Q = Q.reshape(*prefix, L, self.num_heads, D)   
        K = K.reshape(*prefix, S, self.num_kv_heads,D)  
        V = V.reshape(*prefix, S, self.num_kv_heads, D)
        rope = RoPE(D, self.max_seq_len, self.theta, traditional=False)
        Q = rope(Q, offset=slice(0, L))
        K=rope(K,offset=slice(0, L))
        Q = mx.swapaxes(Q, -3, -2) 
        K=mx.swapaxes(K,-3, -2)
        V=mx.swapaxes(V,-3,-2)
        # (N...,H*n_repeat,L,D)
        attention= scaled_dot_product_attention_grouped(Q,K,V,mask=mask)
        attention=mx.swapaxes(attention,-3,-2)
        attention=attention.reshape(*prefix,L,E)
        output=linear(attention, self.wo)
        return output



class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        #intermediate_size (dimension of the hidden layer in MLP)
        hidden_dim: int,
        # w_gate/w_up: I x E
        w_gate: mx.array,
        w_up: mx.array,
        # w_down: E x I
        w_down: mx.array,
    ):
        self.dim=dim
        self.hidden_dim=hidden_dim
        self.w_gate=w_gate
        self.w_up=w_up
        self.w_down=w_down

    def __call__(self, x: mx.array) -> mx.array:
        # x: N.. x L x E
        x_gate=mx.matmul(x,mx.transpose(self.w_gate,[-1,-2]))
        # N.. x L x I
        silu_x_gate=silu(x_gate)
        x_up=mx.matmul(x,mx.transpose(self.w_up,[-1,-2]))
        x_intermediate=x_up*silu_x_gate
        output=mx.matmul(x_intermediate,mx.transpose(self.w_down,[-1,-2]))
        return output

class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
