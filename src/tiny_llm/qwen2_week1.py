import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear
from typing import List

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
        self.rope=RoPE(self.hidden_size//self.num_heads, self.max_seq_len, self.theta, traditional=False)
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
        Q = self.rope(Q, offset=slice(0, L))
        K=self.rope(K,offset=slice(0, L))
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
       self.qwen2MultiHeadAttention= Qwen2MultiHeadAttention(hidden_size,num_attention_heads,num_kv_heads,wq,wk,wv,wo,bq,bk,bv,max_seq_len,theta)
       self.qwen2MLP=Qwen2MLP(hidden_size,intermediate_size,w_gate,w_up,w_down)
       self.input_RMSNorm=RMSNorm(hidden_size,w_input_layernorm,rms_norm_eps)
       self.post_attention_RMSNorm=RMSNorm(hidden_size,w_post_attention_layernorm,rms_norm_eps)
    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        input_layernorm=self.input_RMSNorm(x)
        qwen2MultiHeadAttention=self.qwen2MultiHeadAttention(input_layernorm,mask)
        input_residual=x+qwen2MultiHeadAttention
        post_attention_layernorm=self.post_attention_RMSNorm(input_residual)
        mlp=self.qwen2MLP(post_attention_layernorm)
        post_attention_residual=input_residual+mlp
        return post_attention_residual


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        # print(mlx_model.args)
        # print("\n",mlx_model.model)
        # print("\n",vars(mlx_model))
        # print(type(mlx_model))
        args=mlx_model.args
        model=mlx_model.model
        self.embed_tokens_weight=dequantize_linear(model.embed_tokens)
        self.embedding=Embedding(args.vocab_size,args.hidden_size,self.embed_tokens_weight)
        self.layers_inner:List[Qwen2TransformerBlock] = []
        for i in range(mlx_model.args.num_hidden_layers):
            current_layer=model.layers[i]
            wq=dequantize_linear(current_layer.self_attn.q_proj)
            wk=dequantize_linear(current_layer.self_attn.k_proj)
            wv=dequantize_linear(current_layer.self_attn.v_proj)
            wo=dequantize_linear(current_layer.self_attn.o_proj)
            w_gate=dequantize_linear(current_layer.mlp.gate_proj)
            w_up=dequantize_linear(current_layer.mlp.up_proj)
            w_down=dequantize_linear(current_layer.mlp.down_proj)
            w_input_layernorm=current_layer.input_layernorm.weight
            w_post_attention_layernorm=current_layer.post_attention_layernorm.weight
            layer=Qwen2TransformerBlock(args.num_attention_heads,
                                        args.num_key_value_heads,
                                        args.hidden_size,
                                        args.intermediate_size,
                                        args.rms_norm_eps,
                                        wq,
                                        wk,
                                        wv,
                                        wo,
                                        current_layer.self_attn.q_proj.bias,
                                        current_layer.self_attn.k_proj.bias,
                                        current_layer.self_attn.v_proj.bias,
                                        w_gate,
                                        w_up,
                                        w_down,
                                        w_input_layernorm,
                                        w_post_attention_layernorm,
                                        args.max_position_embeddings,
                                        args.rope_theta                        
                                        )
            self.layers_inner.append(layer)
        self.rMS_norm = RMSNorm(
            args.hidden_size,
            weight=model.norm.weight,
            eps=args.rms_norm_eps,
        )
        if args.tie_word_embeddings is True:
            self.w_lm_head = None
        else:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        
    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        h=self.embedding(inputs)
        for _, transformer_block in enumerate(self.layers_inner):
            h = transformer_block(h,mask="causal")
        h=self.rMS_norm(h)
        if self.w_lm_head is not None:
            return linear(h, self.w_lm_head)
        else:
            return self.embedding.as_linear(h)

