import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any, List
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights
from .kv_cache import TinyKvCache


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.num_kv_heads=num_kv_heads
        self.wq=dequantize_linear(wq)
        self.wk=dequantize_linear(wk)
        self.wv=dequantize_linear(wv)
        self.wo=dequantize_linear(wo)
        self.bq=bq
        self.bk=bk
        self.bv=bv
        self.max_seq_len=max_seq_len
        self.theta=theta
        self.rope=RoPE(self.hidden_size//self.num_heads, self.max_seq_len, self.theta, traditional=False)
  

    def __call__(
        self,
        x: mx.array,
        offsets: list[int],
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        # x: B, L_Q, E
        Q=linear(x,self.wq,self.bq)
        K_new=linear(x,self.wk,self.bk)
        V_new=linear(x,self.wv,self.bv)
        D=self.hidden_size//self.num_heads
        q_shape=Q.shape
        E=q_shape[-1]
        # L_Q 是当前步长,即新token的个数
        L_Q=q_shape[-2]
        prefix=q_shape[:-2]
        K_new = K_new.reshape(*prefix, L_Q, self.num_kv_heads, D)
        V_new = V_new.reshape(*prefix, L_Q, self.num_kv_heads, D)
        # (B, L_Q, h, d)
        Q = Q.reshape(*prefix, L_Q, self.num_heads, D)   
        if isinstance(offsets, int):
            offset_slice = slice(int(offsets), int(offsets + L_Q))
        else:
            offset_slice = [slice(int(o), int(o + L_Q)) for o in offsets]
        Q = self.rope(Q,offset=offset_slice)
        K_new = self.rope(K_new,offset=offset_slice)
        # K，V (B,  L_Q+L, num_kv_heads, D)   
        K,V=cache.update_and_fetch(K_new,V_new)
        # K与Q的维度L不同,Q是新的token，K是全部的
        assert E % self.num_heads == 0, f"E={E} must be divisible by h={self.num_heads}"
        Q = mx.swapaxes(Q, -3, -2) 
        K = mx.swapaxes(K,-3, -2)
        V = mx.swapaxes(V,-3,-2)
         # (N...,H_q*n_repeat,L_Q,D)
        attention= scaled_dot_product_attention_grouped(Q,K,V,mask=mask)
        attention=mx.swapaxes(attention,-3,-2)
        attention=attention.reshape(*prefix,L_Q,E)
        output=linear(attention, self.wo)
        return output


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim=dim
        self.hidden_dim=hidden_dim
        self.w_gate=dequantize_linear(w_gate)
        self.w_up=dequantize_linear(w_up)
        self.w_down=dequantize_linear(w_down)

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
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_flash_attention: bool = False,
    ):
       self.qwen2MultiHeadAttention= Qwen2MultiHeadAttention(hidden_size,num_attention_heads,num_kv_heads,wq,wk,wv,wo,bq,bk,bv,max_seq_len,theta)
       self.qwen2MLP=Qwen2MLP(hidden_size,intermediate_size,w_gate,w_up,w_down)
       self.input_RMSNorm=RMSNorm(hidden_size,w_input_layernorm,rms_norm_eps)
       self.post_attention_RMSNorm=RMSNorm(hidden_size,w_post_attention_layernorm,rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        input_layernorm=self.input_RMSNorm(x)
        qwen2MultiHeadAttention=self.qwen2MultiHeadAttention(input_layernorm,offset,cache,mask)
        input_residual=x+qwen2MultiHeadAttention
        post_attention_layernorm=self.post_attention_RMSNorm(input_residual)
        mlp=self.qwen2MLP(post_attention_layernorm)
        post_attention_residual=input_residual+mlp
        return post_attention_residual


class Qwen2ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        enable_flash_attn: bool = False,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        args=mlx_model.args
        self.args=args
        model=mlx_model.model
        # self.embed_tokens_weight=QuantizedWeights.from_mlx_layer(model.embed_tokens)
        self.embed_tokens_weight=dequantize_linear(mlx_model.model.embed_tokens)
        self.embedding=Embedding(args.vocab_size,args.hidden_size,self.embed_tokens_weight)
        self.layers_inner:List[Qwen2TransformerBlock] = []
        for i in range(mlx_model.args.num_hidden_layers):
            current_layer=model.layers[i]
            wq = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.q_proj
            )
            wk = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.k_proj
            )
            wv = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.v_proj
            )
            wo = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].self_attn.o_proj
            )
            w_gate = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].mlp.gate_proj
            )
            w_up = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].mlp.up_proj
            )
            w_down = QuantizedWeights.from_mlx_layer(
                mlx_model.model.layers[i].mlp.down_proj
            )
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
        offset: int,
        cache: list[TinyKvCache],
    ) -> mx.array:
        h=self.embedding(inputs)
        for i, transformer_block in enumerate(self.layers_inner):
            h = transformer_block(h,offset,mask="causal",cache=cache[i])
        h=self.rMS_norm(h)
        if self.w_lm_head is not None:
            return linear(h, self.w_lm_head)
        else:
            return self.embedding.as_linear(h)

