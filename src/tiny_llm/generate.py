import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        output_logits=model(y[None])
        # y: N.. x S, where in week 1 we don't implement batch, so N.. = 1
        # output_logits: N.. x S x vocab_size
        logits = output_logits[:, -1, :]
        logprobs= logits - mx.logsumexp(logits, keepdims=True)
        # (1,1)
        if sampler is  None:  
            next_token = mx.argmax(logits, axis=-1)
        else:
            next_token=sampler(logprobs)
        return next_token 
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    inputs=mx.array(tokenizer.encode(prompt))
    while(True):
       next_token = _step(model,inputs)
       inputs = mx.concat([inputs, next_token])
       if next_token.item()==tokenizer.eos_token_id:
           break
       detokenizer.add_token(next_token.item())
       print(detokenizer.last_segment, end="", flush=True)



def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
