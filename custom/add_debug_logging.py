#!/usr/bin/env python3
"""
Patch vLLM to add debug logging at critical points
This will help us trace WHERE the BF16 divergence happens
"""

import sys

# Patches to add
PATCH_LOCATIONS = [
    # Patch 1: Log when KV cache is written
    {
        'file': '/app/vllm/vllm/v1/attention/backends/flash_attn.py',
        'function': 'do_kv_cache_update',
        'insert_after_line': 'def do_kv_cache_update(',
        'code': '''
    import torch
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"[KV_WRITE] layer={layer.layer_idx if hasattr(layer, 'layer_idx') else '?'}, "
                  f"key_shape={key.shape}, key_dtype={key.dtype}, "
                  f"key_mean={key.float().mean().item():.6f}, key_std={key.float().std().item():.6f}")
'''
    },

    # Patch 2: Log when attention is computed
    {
        'file': '/app/vllm/vllm/v1/attention/backends/flash_attn.py',
        'function': 'forward',
        'insert_after_line': 'def forward(',
        'code': '''
    import torch
    import logging
    logger = logging.getLogger(__name__)
    cache_hit = attn_metadata.num_decode_tokens > 0
    logger.warning(f"[ATTN_FWD] layer={layer.layer_idx if hasattr(layer, 'layer_idx') else '?'}, "
                  f"cache_hit={cache_hit}, query_shape={query.shape}, query_dtype={query.dtype}, "
                  f"query_mean={query.float().mean().item():.6f}")
'''
    },

    # Patch 3: Log attention output
    {
        'file': '/app/vllm/vllm/v1/attention/backends/flash_attn.py',
        'function': 'forward',
        'insert_before_line': 'return output',
        'code': '''
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"[ATTN_OUT] layer={layer.layer_idx if hasattr(layer, 'layer_idx') else '?'}, "
                  f"output_shape={output.shape}, output_mean={output.float().mean().item():.6f}, "
                  f"output_std={output.float().std().item():.6f}")
'''
    },
]

def apply_patches():
    """Apply debug logging patches to vLLM"""
    print("Applying debug logging patches...")

    for i, patch in enumerate(PATCH_LOCATIONS):
        print(f"\nPatch {i+1}: {patch['file']} - {patch['function']}")

        try:
            with open(patch['file'], 'r') as f:
                lines = f.readlines()

            # Find insertion point
            if 'insert_after_line' in patch:
                search_line = patch['insert_after_line']
                for idx, line in enumerate(lines):
                    if search_line in line:
                        # Insert after this line
                        indent = len(line) - len(line.lstrip())
                        code_lines = patch['code'].split('\n')
                        # Add proper indentation
                        indented_code = '\n'.join(
                            ' ' * (indent + 4) + l if l.strip() else ''
                            for l in code_lines
                        ) + '\n'
                        lines.insert(idx + 1, indented_code)
                        print(f"  ✓ Inserted after line {idx + 1}")
                        break
            elif 'insert_before_line' in patch:
                search_line = patch['insert_before_line']
                for idx, line in enumerate(lines):
                    if search_line in line:
                        # Insert before this line
                        indent = len(line) - len(line.lstrip())
                        code_lines = patch['code'].split('\n')
                        indented_code = '\n'.join(
                            ' ' * indent + l if l.strip() else ''
                            for l in code_lines
                        ) + '\n'
                        lines.insert(idx, indented_code)
                        print(f"  ✓ Inserted before line {idx + 1}")
                        break

            # Write back
            with open(patch['file'], 'w') as f:
                f.writelines(lines)

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\n✓ All patches applied!")
    print("\nNow run your test and check the logs for [KV_WRITE], [ATTN_FWD], [ATTN_OUT] messages")

def remove_patches():
    """Remove debug logging patches (restore from git)"""
    import subprocess
    print("Removing patches (git checkout)...")

    files = set(p['file'] for p in PATCH_LOCATIONS)
    for f in files:
        subprocess.run(['git', 'checkout', f], cwd='/app/vllm')

    print("✓ Patches removed")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'remove':
        remove_patches()
    else:
        apply_patches()
        print("\nTo remove patches later, run:")
        print("  python add_debug_logging.py remove")
