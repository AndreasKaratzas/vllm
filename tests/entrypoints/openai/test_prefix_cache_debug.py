"""
Comprehensive test to diagnose prefix caching determinism issues on gfx950.
Run with: pytest -v -s test_prefix_cache_debug.py
"""

import subprocess
import pytest
import pytest_asyncio
import httpx
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"
GEN_ENDPOINT = "/inference/v1/generate"


def get_hf_reference(tokenizer, messages, use_cache=True, num_runs=1):
    """
    Generate reference output using HuggingFace directly.
    Now captures top-5 logprobs for each position.
    """
    cache_str = "WITH" if use_cache else "WITHOUT"
    print(f"\n{'='*80}")
    print(f"GENERATING HUGGINGFACE REFERENCE ({cache_str} KV CACHE)")
    print(f"{'='*80}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    token_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False,
    )
    
    print(f"Input tokens: {len(token_ids)}")
    print(f"use_cache: {use_cache}")
    
    input_ids = torch.tensor([token_ids], device=model.device)
    
    results = []
    
    for run_idx in range(num_runs):
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=use_cache,
            )
        
        generated_ids = outputs.sequences[0, len(token_ids):].tolist()
        
        # Get top-5 logprobs for each generated token
        top5_logprobs = []
        logprobs = []
        for i, scores in enumerate(outputs.scores):
            log_probs = torch.log_softmax(scores[0], dim=-1)
            
            # Get top 5
            top5_values, top5_indices = torch.topk(log_probs, k=5)
            top5_list = [
                {'token_id': idx.item(), 'logprob': val.item(), 'token': tokenizer.decode([idx.item()])}
                for val, idx in zip(top5_values, top5_indices)
            ]
            top5_logprobs.append(top5_list)
            
            # Also store the logprob of the actual generated token
            token_id = generated_ids[i]
            logprobs.append(log_probs[token_id].item())
        
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        results.append({
            'run': run_idx + 1,
            'tokens': generated_ids,
            'logprobs': logprobs,
            'top5_logprobs': top5_logprobs,
            'decoded': decoded,
        })
        
        print(f"  Run {run_idx + 1}: '{decoded}'")
        if run_idx == 0:
            print(f"  Tokens: {generated_ids}")
            print(f"  Logprobs: {[f'{lp:.6f}' for lp in logprobs]}")
    
    # Check determinism across runs
    if num_runs > 1:
        all_same = all(
            tuple(r['tokens']) == tuple(results[0]['tokens']) 
            for r in results
        )
        print(f"\n  HF Determinism ({num_runs} runs): {'✓ All same' if all_same else '✗ Differs'}")
    
    print(f"{'='*80}\n")
    
    # Clean up to free GPU memory
    del model
    torch.cuda.empty_cache()
    
    # Return first result plus metadata
    return {
        'tokens': results[0]['tokens'],
        'logprobs': results[0]['logprobs'],
        'top5_logprobs': results[0]['top5_logprobs'],
        'decoded': results[0]['decoded'],
        'input_token_ids': token_ids,
        'use_cache': use_cache,
        'all_runs': results,
    }


async def run_vllm_requests(client, token_ids, tokenizer, num_runs=5, max_tokens=10):
    """Helper to run multiple vLLM requests and collect results with top-5 logprobs."""
    results = []
    
    for i in range(num_runs):
        payload = {
            "model": MODEL_NAME,
            "token_ids": token_ids,
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "detokenize": False,
                "logprobs": 5,  # Request top 5
            },
            "stream": False,
        }
        resp = await client.post(GEN_ENDPOINT, json=payload)
        data = resp.json()
        
        tokens = data["choices"][0]["token_ids"]
        logprobs_data = data["choices"][0].get("logprobs", {}).get("content", [])
        
        # Extract top-5 logprobs for each position
        top5_logprobs = []
        logprobs = []
        
        for lp_entry in logprobs_data:
            # The selected token's logprob
            logprobs.append(lp_entry.get("logprob"))
            
            # Top logprobs (includes the selected token + alternatives)
            top_lps = lp_entry.get("top_logprobs", [])
            top5_list = [
                {
                    'token_id': tp.get("token_id", None),
                    'logprob': tp.get("logprob"),
                    'token': tp.get("token", tokenizer.decode([tp.get("token_id")]) if tp.get("token_id") else "?")
                }
                for tp in top_lps[:5]
            ]
            top5_logprobs.append(top5_list)
        
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        
        results.append({
            'run': i + 1,
            'tokens': tokens,
            'logprobs': logprobs,
            'top5_logprobs': top5_logprobs,
            'decoded': decoded,
        })
        
        print(f"  Run {i+1}: '{decoded}'")
    
    return results


def print_top5_table(name, result, tokenizer, max_positions=10):
    """
    Print a nice table with top-5 logprobs.
    Rows: Rank 1-5 + Selected token
    Columns: Each token position
    """
    top5_data = result.get('top5_logprobs', [])
    tokens = result.get('tokens', [])
    
    num_positions = min(max_positions, len(top5_data))
    
    if num_positions == 0:
        print(f"\n{name}: No data available")
        return
    
    # Column width for each position
    col_width = 18
    
    print(f"\n{'='*120}")
    print(f"TOP-5 LOGPROBS: {name}")
    print(f"{'='*120}")
    
    # Header row with position numbers and generated tokens
    header = f"{'Rank':<6}"
    for pos in range(num_positions):
        token_id = tokens[pos] if pos < len(tokens) else 0
        token_str = repr(tokenizer.decode([token_id]))
        if len(token_str) > col_width - 2:
            token_str = token_str[:col_width-5] + "..."
        header += f" Pos{pos} {token_str:>{col_width-6}}"
    print(header)
    print("-" * (6 + num_positions * (col_width + 1)))
    
    # Rows for ranks 1-5
    for rank in range(5):
        row = f"#{rank+1:<5}"
        for pos in range(num_positions):
            if pos < len(top5_data) and rank < len(top5_data[pos]):
                entry = top5_data[pos][rank]
                tok_str = repr(entry['token'])
                if len(tok_str) > 8:
                    tok_str = tok_str[:5] + ".."
                logprob = entry['logprob']
                cell = f"{tok_str}: {logprob:+.4f}"
            else:
                cell = "N/A"
            row += f" {cell:>{col_width}}"
        print(row)
    
    # Separator
    print("-" * (6 + num_positions * (col_width + 1)))
    
    # Row showing which token was actually selected
    selected_row = f"{'SEL':<6}"
    for pos in range(num_positions):
        if pos < len(tokens):
            token_id = tokens[pos]
            tok_str = repr(tokenizer.decode([token_id]))
            if len(tok_str) > 8:
                tok_str = tok_str[:5] + ".."
            logprob = result['logprobs'][pos] if pos < len(result['logprobs']) else 0
            cell = f"{tok_str}: {logprob:+.4f}"
        else:
            cell = "N/A"
        selected_row += f" {cell:>{col_width}}"
    print(selected_row)
    
    print(f"{'='*120}")
    print(f"Decoded: '{result['decoded']}'")
    print()


def print_top5_comparison_table(results_dict, tokenizer, max_positions=10):
    """
    Print a comparison table showing top-5 for multiple methods side by side.
    results_dict: {'Method Name': result_data, ...}
    """
    print(f"\n{'='*140}")
    print("TOP-5 LOGPROBS COMPARISON (showing rank #1 token and logprob for each method)")
    print(f"{'='*140}")
    
    methods = list(results_dict.keys())
    num_methods = len(methods)
    
    # Determine number of positions from first result
    first_result = list(results_dict.values())[0]
    num_positions = min(max_positions, len(first_result.get('top5_logprobs', [])))
    
    col_width = 20
    
    # Header
    header = f"{'Pos':<4} {'Token':<10}"
    for method in methods:
        method_short = method[:col_width-2]
        header += f" {method_short:^{col_width}}"
    print(header)
    print("-" * (14 + num_methods * (col_width + 1)))
    
    # For each position, show the top-1 token and logprob from each method
    for pos in range(num_positions):
        # Get token from first method
        first_tokens = first_result.get('tokens', [])
        token_id = first_tokens[pos] if pos < len(first_tokens) else 0
        token_str = repr(tokenizer.decode([token_id]))[:8]
        
        row = f"{pos:<4} {token_str:<10}"
        
        for method in methods:
            result = results_dict[method]
            top5 = result.get('top5_logprobs', [])
            
            if pos < len(top5) and len(top5[pos]) > 0:
                top1 = top5[pos][0]
                tok = repr(top1['token'])[:6]
                lp = top1['logprob']
                cell = f"{tok}: {lp:+.4f}"
            else:
                cell = "N/A"
            
            row += f" {cell:^{col_width}}"
        
        print(row)
    
    print("-" * (14 + num_methods * (col_width + 1)))


def print_full_comparison_table(hf_with_cache, hf_without_cache, vllm_with_pc, vllm_without_pc, tokenizer):
    """Print a comprehensive comparison table."""
    print(f"\n{'='*140}")
    print("FULL COMPARISON TABLE")
    print(f"{'='*140}")
    print(f"{'Pos':<4} {'Token':<10} {'HF+Cache':<12} {'HF-Cache':<12} "
          f"{'vLLM+PC R1':<12} {'vLLM+PC R2':<12} {'vLLM-PC R1':<12} {'vLLM-PC R2':<12}")
    print("-" * 140)
    
    num_positions = min(
        10,
        len(hf_with_cache['logprobs']),
        len(hf_without_cache['logprobs']),
        len(vllm_with_pc[0]['logprobs']) if vllm_with_pc else 0,
        len(vllm_without_pc[0]['logprobs']) if vllm_without_pc else 0,
    )
    
    for pos in range(num_positions):
        hf_cache = hf_with_cache['logprobs'][pos]
        hf_nocache = hf_without_cache['logprobs'][pos]
        vllm_pc_r1 = vllm_with_pc[0]['logprobs'][pos]
        vllm_pc_r2 = vllm_with_pc[1]['logprobs'][pos]
        vllm_nopc_r1 = vllm_without_pc[0]['logprobs'][pos]
        vllm_nopc_r2 = vllm_without_pc[1]['logprobs'][pos]
        
        token_id = vllm_with_pc[0]['tokens'][pos] if pos < len(vllm_with_pc[0]['tokens']) else 0
        token_str = repr(tokenizer.decode([token_id]))[:8]
        
        print(f"{pos:<4} {token_str:<10} {hf_cache:<12.6f} {hf_nocache:<12.6f} "
              f"{vllm_pc_r1:<12.6f} {vllm_pc_r2:<12.6f} {vllm_nopc_r1:<12.6f} {vllm_nopc_r2:<12.6f}")
    
    print("-" * 140)


def print_difference_analysis(hf_with_cache, hf_without_cache, vllm_with_pc, vllm_without_pc, tokenizer):
    """Print analysis of differences."""
    print(f"\n{'='*100}")
    print("DIFFERENCE ANALYSIS")
    print(f"{'='*100}")
    print(f"{'Pos':<4} {'Token':<10} {'HF+C vs HF-C':<14} {'vLLM+PC R1-R2':<14} "
          f"{'vLLM-PC R1-R2':<14} {'vLLM+PC vs HF':<14} {'vLLM-PC vs HF':<14}")
    print("-" * 100)
    
    num_positions = min(
        10,
        len(hf_with_cache['logprobs']),
        len(hf_without_cache['logprobs']),
        len(vllm_with_pc[0]['logprobs']) if vllm_with_pc else 0,
    )
    
    for pos in range(num_positions):
        hf_cache = hf_with_cache['logprobs'][pos]
        hf_nocache = hf_without_cache['logprobs'][pos]
        vllm_pc_r1 = vllm_with_pc[0]['logprobs'][pos]
        vllm_pc_r2 = vllm_with_pc[1]['logprobs'][pos]
        vllm_nopc_r1 = vllm_without_pc[0]['logprobs'][pos]
        vllm_nopc_r2 = vllm_without_pc[1]['logprobs'][pos]
        
        token_id = vllm_with_pc[0]['tokens'][pos] if pos < len(vllm_with_pc[0]['tokens']) else 0
        token_str = repr(tokenizer.decode([token_id]))[:8]
        
        # Calculate differences
        diff_hf = hf_cache - hf_nocache
        diff_vllm_pc = vllm_pc_r1 - vllm_pc_r2
        diff_vllm_nopc = vllm_nopc_r1 - vllm_nopc_r2
        diff_vllm_pc_hf = vllm_pc_r2 - hf_nocache  # Use R2 and HF without cache as reference
        diff_vllm_nopc_hf = vllm_nopc_r2 - hf_nocache
        
        # Format with markers for significant differences
        def fmt(val, threshold=1e-6):
            marker = "✗" if abs(val) > threshold else "✓"
            return f"{marker} {val:+.6f}"
        
        print(f"{pos:<4} {token_str:<10} {fmt(diff_hf, 1e-6):<14} {fmt(diff_vllm_pc):<14} "
              f"{fmt(diff_vllm_nopc):<14} {fmt(diff_vllm_pc_hf, 0.01):<14} {fmt(diff_vllm_nopc_hf, 0.01):<14}")
    
    print("-" * 100)


@pytest.mark.asyncio
async def test_hf_cache_comparison():
    """Test HuggingFace with and without KV cache."""
    print(f"\n{'='*80}")
    print("TEST: HUGGINGFACE CACHE COMPARISON")
    print(f"{'='*80}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many countries are in the EU?"},
    ]
    
    # Run HF with cache (3 runs to check determinism)
    hf_with_cache = get_hf_reference(tokenizer, messages, use_cache=True, num_runs=3)
    
    # Run HF without cache (3 runs to check determinism)
    hf_without_cache = get_hf_reference(tokenizer, messages, use_cache=False, num_runs=3)
    
    # Print top-5 tables
    print_top5_table("HuggingFace WITH KV Cache", hf_with_cache, tokenizer)
    print_top5_table("HuggingFace WITHOUT KV Cache", hf_without_cache, tokenizer)
    
    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON: HF WITH CACHE vs HF WITHOUT CACHE")
    print(f"{'='*80}")
    print(f"{'Pos':<4} {'Token':<12} {'HF+Cache':<14} {'HF-Cache':<14} {'Difference':<14} {'Status'}")
    print("-" * 80)
    
    for pos in range(min(10, len(hf_with_cache['logprobs']))):
        lp_cache = hf_with_cache['logprobs'][pos]
        lp_nocache = hf_without_cache['logprobs'][pos]
        diff = lp_cache - lp_nocache
        
        token_id = hf_with_cache['tokens'][pos]
        token_str = repr(tokenizer.decode([token_id]))[:10]
        
        status = "✓ SAME" if abs(diff) < 1e-9 else "✗ DIFFERS"
        
        print(f"{pos:<4} {token_str:<12} {lp_cache:<14.6f} {lp_nocache:<14.6f} {diff:+.6e}     {status}")
    
    print("-" * 80)
    
    # Token comparison
    tokens_match = tuple(hf_with_cache['tokens']) == tuple(hf_without_cache['tokens'])
    print(f"\nTokens match: {'✓ YES' if tokens_match else '✗ NO'}")
    print(f"  With cache:    '{hf_with_cache['decoded']}'")
    print(f"  Without cache: '{hf_without_cache['decoded']}'")
    
    print(f"{'='*80}\n")


@pytest.mark.asyncio
async def test_side_by_side():
    """Run all configurations and compare side-by-side."""
    print(f"\n{'='*80}")
    print("TEST: COMPREHENSIVE SIDE-BY-SIDE COMPARISON")
    print(f"{'='*80}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How many countries are in the EU?"},
    ]
    
    # === HuggingFace references ===
    print("\n" + "="*80)
    print("STEP 1: HUGGINGFACE REFERENCES")
    print("="*80)
    
    hf_with_cache = get_hf_reference(tokenizer, messages, use_cache=True, num_runs=3)
    hf_without_cache = get_hf_reference(tokenizer, messages, use_cache=False, num_runs=3)
    
    token_ids = hf_with_cache['input_token_ids']
    
    # === vLLM with prefix caching ===
    print("\n" + "="*80)
    print("STEP 2: VLLM WITH PREFIX CACHING")
    print("="*80)
    
    args_with_pc = [
        "--dtype", "bfloat16",
        "--max-model-len", "1024",
        "--enforce-eager",
        "--generation-config", "vllm",
    ]
    
    with RemoteOpenAIServer(MODEL_NAME, args_with_pc) as server:
        transport = httpx.AsyncHTTPTransport(uds=server.uds) if server.uds else None
        headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
        
        async with httpx.AsyncClient(
            transport=transport,
            base_url=server.url_root,
            timeout=600,
            headers=headers,
        ) as client:
            print("\nRunning 5 requests with prefix caching ENABLED...")
            vllm_with_pc = await run_vllm_requests(client, token_ids, tokenizer, num_runs=5)
    
    # === vLLM without prefix caching ===
    print("\n" + "="*80)
    print("STEP 3: VLLM WITHOUT PREFIX CACHING")
    print("="*80)
    
    args_without_pc = [
        "--dtype", "bfloat16",
        "--max-model-len", "1024",
        "--enforce-eager",
        "--generation-config", "vllm",
        "--no-enable-prefix-caching",
    ]
    
    with RemoteOpenAIServer(MODEL_NAME, args_without_pc) as server:
        transport = httpx.AsyncHTTPTransport(uds=server.uds) if server.uds else None
        headers = {"Authorization": f"Bearer {server.DUMMY_API_KEY}"}
        
        async with httpx.AsyncClient(
            transport=transport,
            base_url=server.url_root,
            timeout=600,
            headers=headers,
        ) as client:
            print("\nRunning 5 requests with prefix caching DISABLED...")
            vllm_without_pc = await run_vllm_requests(client, token_ids, tokenizer, num_runs=5)
    
    # === Print TOP-5 tables for each method ===
    print("\n" + "="*80)
    print("TOP-5 LOGPROBS TABLES")
    print("="*80)
    
    print_top5_table("HuggingFace WITH KV Cache", hf_with_cache, tokenizer)
    print_top5_table("HuggingFace WITHOUT KV Cache", hf_without_cache, tokenizer)
    print_top5_table("vLLM WITH Prefix Caching (Run 1)", vllm_with_pc[0], tokenizer)
    print_top5_table("vLLM WITH Prefix Caching (Run 2)", vllm_with_pc[1], tokenizer)
    print_top5_table("vLLM WITHOUT Prefix Caching (Run 1)", vllm_without_pc[0], tokenizer)
    print_top5_table("vLLM WITHOUT Prefix Caching (Run 2)", vllm_without_pc[1], tokenizer)
    
    # === Print comparison table ===
    print_top5_comparison_table({
        'HF+Cache': hf_with_cache,
        'HF-Cache': hf_without_cache,
        'vLLM+PC R1': vllm_with_pc[0],
        'vLLM+PC R2': vllm_with_pc[1],
        'vLLM-PC R1': vllm_without_pc[0],
        'vLLM-PC R2': vllm_without_pc[1],
    }, tokenizer)
    
    # === Print full comparison ===
    print_full_comparison_table(hf_with_cache, hf_without_cache, vllm_with_pc, vllm_without_pc, tokenizer)
    print_difference_analysis(hf_with_cache, hf_without_cache, vllm_with_pc, vllm_without_pc, tokenizer)
    
    # === Final Analysis ===
    print(f"\n{'='*80}")
    print("FINAL ANALYSIS")
    print(f"{'='*80}")

    # HuggingFace analysis
    print(f"\n--- HUGGINGFACE ---")
    hf_cache_match = tuple(hf_with_cache['tokens']) == tuple(hf_without_cache['tokens'])
    hf_with_cache_deterministic = all(
        tuple(r['tokens']) == tuple(hf_with_cache['all_runs'][0]['tokens'])
        for r in hf_with_cache['all_runs']
    )
    hf_without_cache_deterministic = all(
        tuple(r['tokens']) == tuple(hf_without_cache['all_runs'][0]['tokens'])
        for r in hf_without_cache['all_runs']
    )
    
    print(f"  HF with cache deterministic:    {'✓' if hf_with_cache_deterministic else '✗'}")
    print(f"  HF without cache deterministic: {'✓' if hf_without_cache_deterministic else '✗'}")
    print(f"  HF+cache == HF-cache:           {'✓' if hf_cache_match else '✗'}")
    if not hf_cache_match:
        print(f"    With cache:    '{hf_with_cache['decoded']}'")
        print(f"    Without cache: '{hf_without_cache['decoded']}'")
    
    # vLLM with prefix caching
    print(f"\n--- VLLM WITH PREFIX CACHING ---")
    pc_all_same = all(tuple(r['tokens']) == tuple(vllm_with_pc[0]['tokens']) for r in vllm_with_pc)
    pc_first_differs = (
        tuple(vllm_with_pc[0]['tokens']) != tuple(vllm_with_pc[1]['tokens']) and
        all(tuple(r['tokens']) == tuple(vllm_with_pc[1]['tokens']) for r in vllm_with_pc[1:])
    )
    
    if pc_all_same:
        print(f"  Deterministic: ✓ All 5 runs identical")
    elif pc_first_differs:
        print(f"  Deterministic: ✗ First run differs (PREFIX CACHE BUG!)")
    else:
        print(f"  Deterministic: ✗ Multiple different outputs")
    
    # vLLM without prefix caching
    print(f"\n--- VLLM WITHOUT PREFIX CACHING ---")
    nopc_all_same = all(tuple(r['tokens']) == tuple(vllm_without_pc[0]['tokens']) for r in vllm_without_pc)
    
    if nopc_all_same:
        print(f"  Deterministic: ✓ All 5 runs identical")
    else:
        print(f"  Deterministic: ✗ Runs differ")
    
    # Cross-comparison
    print(f"\n--- CROSS-COMPARISON ---")
    
    # Use Run 2 for vLLM comparisons (to avoid prefix cache bug on first run)
    vllm_pc_tokens = tuple(vllm_with_pc[1]['tokens'])
    vllm_nopc_tokens = tuple(vllm_without_pc[1]['tokens'])
    hf_tokens = tuple(hf_without_cache['tokens'])  # Use HF without cache as reference
    
    print(f"  vLLM+PC Run2 == vLLM-PC Run2: {'✓' if vllm_pc_tokens == vllm_nopc_tokens else '✗'}")
    print(f"  vLLM+PC Run2 == HF-cache:     {'✓' if vllm_pc_tokens == hf_tokens else '✗'}")
    print(f"  vLLM-PC Run2 == HF-cache:     {'✓' if vllm_nopc_tokens == hf_tokens else '✗'}")
    
    if vllm_pc_tokens != hf_tokens:
        print(f"\n  Decoded outputs:")
        print(f"    HF-cache:     '{hf_without_cache['decoded']}'")
        print(f"    vLLM+PC Run2: '{vllm_with_pc[1]['decoded']}'")
        print(f"    vLLM-PC Run2: '{vllm_without_pc[1]['decoded']}'")
    
    # === Conclusion ===
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    
    issues = []
    
    if pc_first_differs:
        issues.append("PREFIX CACHING BUG: First vLLM+PC request differs from subsequent")
    
    if not pc_all_same and not pc_first_differs:
        issues.append("vLLM+PC NON-DETERMINISTIC: Multiple different outputs")
    
    if not nopc_all_same:
        issues.append("vLLM-PC NON-DETERMINISTIC: Even without prefix caching")
    
    if vllm_nopc_tokens != hf_tokens:
        issues.append("vLLM DIFFERS FROM HF: vLLM output doesn't match HuggingFace")
    
    if not hf_cache_match:
        issues.append("HF CACHE AFFECTS OUTPUT: HF with/without cache produce different results")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("✓ No issues found - all configurations are deterministic and match")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
