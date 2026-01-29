#!/usr/bin/env python3
"""Debug script to capture and compare layer outputs across multiple runs.

This script helps identify which layers first diverge when prefix caching
produces non-deterministic results.
"""

import torch
from vllm import LLM, SamplingParams

# Hook to capture layer outputs
layer_outputs = {}

def forward_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        layer_outputs[name] = output.detach().cpu()
    return hook

def main():
    print("="*80)
    print("DEBUG: Layer Output Comparison for Prefix Cache Non-Determinism")
    print("="*80)

    # Test with cache
    print("\nInitializing LLM...")
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        dtype="bfloat16",
        max_model_len=512,
        enforce_eager=True
    )

    # Register hooks
    print("Registering forward hooks on attention layers...")
    hook_count = 0
    for name, module in llm.llm_engine.model_executor.driver_worker.model_runner.model.named_modules():
        if 'layer' in name and 'attn' in name:
            module.register_forward_hook(forward_hook(name))
            hook_count += 1
    print(f"Registered {hook_count} hooks")

    # Run 3 times
    prompt = "You are a helpful assistant.\nHow many countries are in the EU?"
    print(f"\nPrompt: {prompt}")
    print("\nRunning 3 inference runs...")

    all_outputs = []
    for run_id in range(3):
        layer_outputs.clear()
        output = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=24, seed=0))

        # Save layer outputs
        torch.save(layer_outputs, f"/tmp/layer_outputs_run{run_id}.pt")
        output_text = output[0].outputs[0].text
        print(f"  Run {run_id}: {output_text}")
        all_outputs.append(output_text)

    # Check if outputs are identical
    print("\n" + "="*80)
    if all_outputs[0] == all_outputs[1] == all_outputs[2]:
        print("✅ All runs produced IDENTICAL outputs!")
    else:
        print("❌ Runs produced DIFFERENT outputs:")
        for i, out in enumerate(all_outputs):
            print(f"  Run {i}: {out}")

    # Compare layer outputs
    print("\n" + "="*80)
    print("Layer-by-Layer Divergence Analysis")
    print("="*80)

    outputs = [torch.load(f"/tmp/layer_outputs_run{i}.pt") for i in range(3)]

    max_diff_overall = 0.0
    first_diverging_layer = None

    for layer_name in sorted(outputs[0].keys()):
        diff_01 = (outputs[0][layer_name] - outputs[1][layer_name]).abs().max().item()
        diff_02 = (outputs[0][layer_name] - outputs[2][layer_name]).abs().max().item()
        diff_12 = (outputs[1][layer_name] - outputs[2][layer_name]).abs().max().item()

        max_diff = max(diff_01, diff_02, diff_12)
        if max_diff > max_diff_overall:
            max_diff_overall = max_diff

        if max_diff > 1e-6 and first_diverging_layer is None:
            first_diverging_layer = layer_name

        status = "✅" if max_diff < 1e-6 else "❌"
        print(f"{status} {layer_name:60s} max_diff(R0-R1)={diff_01:.8f}, max_diff(R0-R2)={diff_02:.8f}, max_diff(R1-R2)={diff_12:.8f}")

    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    if first_diverging_layer:
        print(f"First diverging layer: {first_diverging_layer}")
        print(f"Maximum difference overall: {max_diff_overall:.8f}")
    else:
        print("All layers produced identical outputs (within 1e-6 tolerance)")
    print("\nOutput files saved to:")
    for i in range(3):
        print(f"  /tmp/layer_outputs_run{i}.pt")

if __name__ == "__main__":
    main()
