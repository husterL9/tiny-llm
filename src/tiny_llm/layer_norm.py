import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim=dim
        #(D,)
        self.weight=weight
        self.eps=eps

    def __call__(self, x: mx.array) -> mx.array:
        x_f32 = x.astype(mx.float32)
        #(N..,)
        mean_sq = mx.mean(x_f32 * x_f32, axis=-1,keepdims=True)
        rms = mx.sqrt(mean_sq + self.eps) 
        #(N..,D)
        y = (x_f32 / rms).astype(x.dtype)
        y = y * self.weight
        return y

        
