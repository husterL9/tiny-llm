import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        assert dims % 2 == 0
        half_dim=dims//2
        base_freqs = 1.0 / (base ** (mx.arange(0, half_dim) / half_dim))
        # positions: (L, 1)
        positions=mx.arange(0,seq_len)[:,None]
        freqs=positions*base_freqs[None,:]
        self.cos_freqs,self.sin_freqs=mx.cos(freqs), mx.sin(freqs)
        self.traditional=traditional
        
    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
      x = x.transpose(0, 2, 1, 3)
      N, H, L, D = x.shape
      half_dim = D // 2
      if offset is None:
        cos = self.cos_freqs[:L]      # (L, D/2)
        sin = self.sin_freqs[:L]
      else:
        cos = self.cos_freqs[offset]  # (L_slice, D/2)
        sin = self.sin_freqs[offset]
        cos = cos[None, None, :, :]       # (1,1,L,D/2)
        sin = sin[None, None, :, :]       # (1,1,L,D/2)
      if self.traditional is True:
        x_even = x[..., 0::2]             # (N,H,L,D/2)
        x_odd  = x[..., 1::2]             # (N,H,L,D/2)
        x_even_rotary = x_even * cos - x_odd * sin
        x_odd_rotary = x_even * sin + x_odd * cos
        x_rotary=mx.stack([x_even_rotary, x_odd_rotary], axis=-1)
        x_rotary = x_rotary.reshape(N, H, L, D)                   # (N,H,L,D)
        return x_rotary.transpose(0,2,1,3)
      else :
        x1,x2=x[...,:half_dim],x[...,half_dim:]
        x_rotary1=x1*cos-x2*sin
        x_rotary2=x1*sin+x2*cos
        x_rotary = mx.concatenate([x_rotary1, x_rotary2], axis=-1)
        return x_rotary.transpose(0,2,1,3)
