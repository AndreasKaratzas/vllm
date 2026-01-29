#!/usr/bin/env python3
"""
Capture hidden states and logits at each layer to find where divergence happens
We'll modify vLLM temporarily to save these values
"""

import torch
import os

# File to store captured tensors
CAPTURE_DIR = "/tmp/vllm_captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

def instrument_model():
    """
    Add hooks to capture intermediate values
    Returns cleanup function
    """
    hooks = []
    captures = {}

    def make_forward_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                captures[name] = output.detach().clone()
                # Save to file
                torch.save(output.detach().cpu(), f"{CAPTURE_DIR}/{name}.pt")
                print(f"[CAPTURE] {name}: shape={output.shape}, "
                      f"mean={output.float().mean().item():.6f}, "
                      f"std={output.float().std().item():.6f}")
            return output
        return hook

    # We need to find the model and add hooks
    # This is tricky because the model is inside the vLLM server process

    def cleanup():
        for h in hooks:
            h.remove()

    return cleanup, captures

print(f"""
This script prepares for capturing intermediate values.

To actually use it, you need to:

1. Modify the vLLM test to import this and call instrument_model()
2. Run the test twice (cache miss and cache hit)
3. Compare the captured tensors

Alternatively, let's try a simpler approach...
""")

# Simpler approach: Just compare the FINAL logits before sampling
print("""
SIMPLER DEBUGGING APPROACH:
===========================

Instead of instrumenting the model, let's just:

1. Modify vLLM to save logits BEFORE sampling
2. Compare logits from run 1 (cache miss) vs run 2 (cache hit)
3. See if they're different for the same input

Let me create a patch for this...
""")
