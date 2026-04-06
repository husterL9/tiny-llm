from abc import ABC, abstractmethod
from typing import Optional

import mlx.core as mx

from tiny_llm.attention import causal_mask


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            A tuple of the updated key-value cache, the updated value, the sequence length, and the mask.
            In week 2 day 1, we only need to return the updated key-value cache, the updated value.
            In week 2 day 6/7, we need to return the updated key-value cache, the updated value, the sequence length, and the mask.
            so that the batching kv cache can use this information to generate the mask.
        """


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        self.kv_caches: list[TinyKvCache] = [None] * max_active_requests
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        # S = max(S_i of the batch)
        # L = mask_length (input parameter)
        # keys: 1, H, S_i, D
        # values: 1, H, S_i, D
        B, H, _, D = key.shape
        S=0
        S_i_arr=[0]*self.max_active_requests
        data = []
        for b in range(B):
            if self.kv_caches[b] is None:
                data.append(None)
                continue
            full_key,full_value,S_i,_= self.kv_caches[b].update_and_fetch(key[b:b+1],value[b:b+1])
            data.append((full_key, full_value, S_i))
            S_i_arr[b]=S_i
            if S_i>S:
                S=S_i
        batched_keys = mx.zeros((B, H, S, D), dtype=key.dtype)
        batched_values = mx.zeros((B, H, S, D), dtype=key.dtype)
        assert mask_length is not None
        masks = mx.full(
    (self.max_active_requests,1, mask_length, S),-mx.inf, dtype=key.dtype,)

        # batched_keys[i, :, (S-S_i):S, :] = keys[i, :, :, :]
        # batched_values[i, :, (S-S_i):S, :] = values[i, :, :, :]
        # mask[i, :, 0:L, (S-S_i):S] = causal_mask(L, S_i)
        for b in range(B):
            if self.kv_caches[b] is None:
                masks[b, :, :] = causal_mask(mask_length, S, dtype=key.dtype)
                continue
            full_key, full_value, S_i = data[b]
            batched_keys[b:b+1, :, S - S_i:S, :] = full_key
            batched_values[b:b+1, :, S - S_i:S, :] = full_value
            masks[b, :, 0:mask_length, (S-S_i_arr[b]):S]=causal_mask(mask_length, S_i_arr[b],dtype=key.dtype)

        return  batched_keys, batched_values, S, masks

    def add_request(self, prefilled: TinyKvCache, id: int):
        if id >= self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        self.kv_caches[id] = prefilled


    def remove_request(self, id: int):
        self.kv_caches[id]=None


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        if self.key_values is None:
            self.key_values=(key,value)
            keys,vals=key,value
            
        else:
            pre_keys,pre_vals=self.key_values
            keys=mx.concat([pre_keys,key],axis=-2)
            vals=mx.concat([pre_vals,value],axis=-2)
            self.key_values=(keys,vals)
        mask_length = keys.shape[-2]
        return keys,vals,mask_length,mask