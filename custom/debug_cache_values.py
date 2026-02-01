#!/usr/bin/env python3
"""
Debug script to instrument K/V cache writes and reads to compare actual values.
This will help identify if K/V values are changing between write and read.
"""
import torch

# Monkey-patch to intercept cache writes and reads
_cache_writes = {}
_cache_reads = {}

# Patch triton_reshape_and_cache_flash
import vllm.v1.attention.ops.triton_reshape_and_cache_flash as reshape_module

original_reshape_and_cache = reshape_module.triton_reshape_and_cache_flash

def instrumented_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale):
    # Capture K/V statistics before writing
    if key.shape[0] > 0:
        layer_id = len(_cache_writes)  # Approximate layer by count
        _cache_writes[f"layer_{layer_id}_tokens_{key.shape[0]}"] = {
            'k_mean': key[0].float().mean().item(),
            'k_std': key[0].float().std().item(),
            'k_min': key[0].float().min().item(),
            'k_max': key[0].float().max().item(),
            'v_mean': value[0].float().mean().item(),
            'v_std': value[0].float().std().item(),
            'tokens': key.shape[0],
            'slot_mapping': slot_mapping[0].item() if slot_mapping.numel() > 0 else None,
        }
        print(f"[CACHE_WRITE_INTERCEPT] layer={layer_id} tokens={key.shape[0]} " 
              f"slot={slot_mapping[0].item() if slot_mapping.numel() > 0 else 'N/A'} "
              f"k_mean={key[0].float().mean().item():.10f}")
    
    # Call original
    return original_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)

reshape_module.triton_reshape_and_cache_flash = instrumented_reshape_and_cache

print("="*80)
print("CACHE VALUE INSTRUMENTATION INSTALLED")
print("="*80)
print("Capturing K/V statistics at write time")
print("Run your vLLM test now...")
print("="*80)
