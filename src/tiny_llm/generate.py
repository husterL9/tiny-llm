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
        output_logits=model(y)
        logits = output_logits[:, -1, :]
        # (1,1)
        next_token = mx.argmax(logits, axis=-1,keepdims=True)
        return next_token 
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    inputs=mx.array(tokenizer.encode(prompt))
    inputs = inputs[None, :]
    while(True):
       next_token = _step(model,inputs)
       inputs = mx.concatenate([inputs, next_token],axis=1)
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
