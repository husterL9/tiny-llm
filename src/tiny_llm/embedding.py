import mlx.core as mx

# Embedding::__call__
# weight: vocab_size x embedding_dim
# Input: N.. (tokens)
# Output: N.. x embedding_dim (vectors)

# Embedding::as_linear
# weight: vocab_size x embedding_dim
# Input: N.. x embedding_dim
# Output: N.. x vocab_size

class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return mx.take(self.weight, x, axis=0)

    def as_linear(self, x: mx.array) -> mx.array:
        return mx.matmul(x,self.weight.T)
