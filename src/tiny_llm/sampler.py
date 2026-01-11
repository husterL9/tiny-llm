import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if top_k is not None and top_k > 0:
             V = logprobs.shape[-1]
             k = min(top_k, V)
             # indice的维度和logprobs一样是 (1,V)
             indices = mx.argpartition(logprobs, V - k, axis=-1)[..., V - k:]
             mask = mx.zeros(logprobs.shape, dtype=mx.int32)              
             mask = mask.at[0,indices[0]].add(1)                               
             logprobs = mx.where(mask, logprobs, -mx.inf)
        if top_p is not None and top_p > 0:
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = logprobs[:, sorted_idx]
            cumsum = mx.cumsum(mx.exp(sorted_logprobs), axis=-1)
            mask_elements = cumsum < top_p
            mask_elements[..., 0] = True
            logprobs[:, sorted_idx] = mx.where(mask_elements, sorted_logprobs, -mx.inf)
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        logprobs=logprobs/temp
        return mx.random.categorical(logprobs)

    return sample
