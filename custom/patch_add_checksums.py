#!/usr/bin/env python3
"""
Add checksum logging to vLLM to track where BF16 diverges
This will print a hash of tensors at each critical point
"""

PATCH_CODE = '''
# Add this at the top of the file
import hashlib
import torch

def tensor_checksum(tensor, name=""):
    """Compute a checksum of a tensor for debugging"""
    if tensor is None:
        return "None"
    # Convert to float32 for consistent hashing
    data = tensor.detach().float().cpu().numpy().tobytes()
    hash_val = hashlib.md5(data).hexdigest()[:8]
    mean = tensor.float().mean().item()
    return f"{name}[{hash_val}] mean={mean:.6f}"
'''

FILES_TO_PATCH = {
    # Model forward pass
    '/app/vllm/vllm/v1/worker/gpu_model_runner.py': [
        ('execute_model', 'after', 'def execute_model(', '''
        # Log input
        if hasattr(self, '_debug_run_counter'):
            self._debug_run_counter += 1
        else:
            self._debug_run_counter = 1
        print(f"\\n[EXECUTE_MODEL] Run #{self._debug_run_counter}")
        '''),
    ],

    # Attention layer
    '/app/vllm/vllm/v1/attention/backends/flash_attn.py': [
        ('forward', 'start', 'def forward(', '''
        # Check if this is cache hit or miss
        is_decode = attn_metadata.num_decode_tokens > 0
        is_prefill = attn_metadata.num_prefill_tokens > 0
        print(f"[ATTN] layer={getattr(layer, 'layer_idx', '?')}, "
              f"decode={is_decode}, prefill={is_prefill}")
        print(f"  Q: {tensor_checksum(query, 'Q')}")
        '''),

        ('forward', 'before_return', 'return output', '''
        print(f"[ATTN_OUT] {tensor_checksum(output, 'out')}")
        '''),

        ('do_kv_cache_update', 'start', 'def do_kv_cache_update(', '''
        print(f"[KV_WRITE] layer={getattr(layer, 'layer_idx', '?')}")
        print(f"  K: {tensor_checksum(key, 'K')}")
        print(f"  V: {tensor_checksum(value, 'V')}")
        '''),
    ],
}

def apply_patch(file_path, function_name, position, marker, code):
    """Apply a single patch to a file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        if 'tensor_checksum' not in content:
            # Add helper function at top
            lines = content.split('\n')
            # Find first import
            import_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_idx = i + 1

            # Insert helper after imports
            helper = PATCH_CODE.strip().split('\n')
            for i, line in enumerate(helper):
                lines.insert(import_idx + i, line)

            content = '\n'.join(lines)

        # Now add the specific patch
        if position == 'after' or position == 'start':
            # Find the function definition
            import re
            pattern = re.escape(marker) + r'.*?:'
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                insert_pos = match.end()
                # Find next newline
                next_newline = content.find('\n', insert_pos)
                if next_newline != -1:
                    # Insert code with proper indentation
                    indent = '    '  # Assume 4-space indent
                    code_to_insert = '\n' + indent + code.strip().replace('\n', '\n' + indent)
                    content = content[:next_newline] + code_to_insert + content[next_newline:]
                    print(f"✓ Patched {function_name} in {file_path}")
                else:
                    print(f"✗ Could not find newline after {function_name}")
            else:
                print(f"✗ Could not find {function_name} in {file_path}")

        elif position == 'before_return':
            # Find return statement in function
            import re
            func_pattern = re.escape(marker.split('(')[0]) + r'\(.*?\):'
            func_match = re.search(func_pattern, content, re.MULTILINE | re.DOTALL)
            if func_match:
                # Find the return statement after this function
                func_start = func_match.end()
                # Look for next 'def ' to find function end
                next_func = content.find('\ndef ', func_start)
                if next_func == -1:
                    next_func = len(content)

                func_content = content[func_start:next_func]

                # Find last 'return' before function end
                return_matches = list(re.finditer(r'\n(\s+)return ', func_content))
                if return_matches:
                    last_return = return_matches[-1]
                    indent = last_return.group(1)
                    insert_pos = func_start + last_return.start() + 1  # +1 for newline
                    code_to_insert = indent + code.strip().replace('\n', '\n' + indent) + '\n'
                    content = content[:insert_pos] + code_to_insert + content[insert_pos:]
                    print(f"✓ Patched {function_name} (before return) in {file_path}")
                else:
                    print(f"✗ Could not find return in {function_name}")
            else:
                print(f"✗ Could not find {function_name}")

        # Write back
        with open(file_path, 'w') as f:
            f.write(content)

        return True

    except Exception as e:
        print(f"✗ Error patching {file_path}: {e}")
        return False

def main():
    print("="*80)
    print("ADDING CHECKSUM LOGGING TO vLLM")
    print("="*80)

    for file_path, patches in FILES_TO_PATCH.items():
        print(f"\nPatching {file_path}...")
        for function_name, position, marker, code in patches:
            apply_patch(file_path, function_name, position, marker, code)

    print("\n" + "="*80)
    print("DONE! Run your test and look for [EXECUTE_MODEL], [ATTN], [KV_WRITE], [ATTN_OUT]")
    print("="*80)
    print("\nTo revert:")
    print("  cd /app/vllm && git checkout vllm/v1/")

if __name__ == "__main__":
    main()
