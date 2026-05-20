# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce the ROCm AITER tiny-mixtral top-k logprob mismatch.

This mirrors the failing AMD CI row from
tests/models/language/generation/test_common.py:

    VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=0 \
      HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
      python tools/vllm-rocm/aiter_tiny_mixtral_repro.py

The script compares Hugging Face greedy generation with vLLM greedy
generation. It exits non-zero when a generated token mismatch is also a top-k
logprob mismatch, which is the condition that fails the parity test.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor


DEFAULT_MODEL = "TitanML/tiny-mixtral"
DEFAULT_PROMPT_FILE = (
    Path(__file__).resolve().parents[2] / "tests" / "prompts" / "example.txt"
)


def _set_env(name: str, value: str) -> None:
    if value == "env":
        return
    os.environ[name] = "1" if value == "on" else "0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HF and vLLM tiny-mixtral top-k logprobs under AITER."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--prompt",
        action="append",
        help=(
            "Prompt to compare. May be passed more than once. Defaults to the "
            "test_common.py example prompt file."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_FILE),
        help="Prompt file to use when --prompt is not provided.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--num-logprobs", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--aiter", choices=("on", "off", "env"), default="on")
    parser.add_argument("--aiter-linear", choices=("on", "off", "env"), default="env")
    parser.add_argument("--aiter-moe", choices=("on", "off", "env"), default="env")
    parser.add_argument("--aiter-rmsnorm", choices=("on", "off", "env"), default="off")
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Print diagnostics but return exit code 0 even on mismatch.",
    )
    parser.add_argument(
        "--continue-after-mismatch",
        action="store_true",
        help=(
            "Keep comparing after top-k-compatible token divergence. The pytest "
            "helper stops after the first mismatch."
        ),
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    _set_env("VLLM_ROCM_USE_AITER", args.aiter)
    _set_env("VLLM_ROCM_USE_AITER_LINEAR", args.aiter_linear)
    _set_env("VLLM_ROCM_USE_AITER_MOE", args.aiter_moe)
    _set_env("VLLM_ROCM_USE_AITER_RMSNORM", args.aiter_rmsnorm)

    # Match tests/models/language/generation/conftest.py.
    os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")


def configure_torch_for_rocm_hf() -> None:
    # Match the ROCm HF accuracy guard in the pytest session hook.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return args.prompt
    with open(args.prompt_file) as prompt_file:
        return prompt_file.readlines()


def hf_generate_logprobs(
    model_name: str, prompts: list[str], max_tokens: int, num_logprobs: int
) -> tuple[list[tuple[list[int], str, list[dict[int, float]]]], Any]:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model = model.to(device="cuda")
    model.eval()

    output_embeddings = model.get_output_embeddings()
    outputs = []
    for prompt in prompts:
        inputs = processor(text=prompt, return_tensors="pt")
        try:
            inputs = inputs.to(dtype=torch.bfloat16)
        except TypeError:
            pass
        inputs = inputs.to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                use_cache=True,
                do_sample=False,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        seq_logprobs = []
        for tok_idx, hidden_state in enumerate(output.hidden_states):
            last_hidden_states = hidden_state[-1][0]
            logits = torch.matmul(
                last_hidden_states.to(
                    device=output_embeddings.weight.device,
                    dtype=output_embeddings.weight.dtype,
                ),
                output_embeddings.weight.t(),
            )
            if getattr(output_embeddings, "bias", None) is not None:
                logits += output_embeddings.bias.unsqueeze(0)

            logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            if tok_idx == 0:
                logprobs = logprobs[-1, :].reshape(1, -1)
            topk = logprobs.topk(num_logprobs)
            seq_logprobs.append(
                {
                    token_id.item(): logprob.item()
                    for token_id, logprob in zip(topk.indices[0], topk.values[0])
                }
            )

        output_len = len(seq_logprobs)
        output_ids = output.sequences[0][-output_len:].tolist()
        outputs.append(
            (output_ids, tokenizer.decode(output_ids), seq_logprobs)
        )

    return outputs, tokenizer


def vllm_generate_logprobs(
    model_name: str, prompts: list[str], max_tokens: int, num_logprobs: int
) -> list[tuple[list[int], str, list[Mapping[int, Any]]]]:
    # Import after configure_environment(), since vLLM reads AITER env vars at
    # import time.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        max_model_len=1024,
        block_size=16,
        max_num_seqs=1,
        disable_log_stats=True,
        enable_chunked_prefill=False,
        compilation_config={"cudagraph_capture_sizes": [1, 2]},
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=num_logprobs,
    )
    request_outputs = llm.generate(prompts, params, use_tqdm=False)
    return [
        (
            list(request_output.outputs[0].token_ids),
            request_output.outputs[0].text,
            request_output.outputs[0].logprobs or [],
        )
        for request_output in request_outputs
    ]


def _logprob_value(value: Any) -> float:
    return float(value.logprob if hasattr(value, "logprob") else value)


def format_topk(topk: Mapping[int, Any], tokenizer: Any) -> str:
    items = sorted(topk.items(), key=lambda item: _logprob_value(item[1]), reverse=True)
    return "\n".join(
        f"    {rank:>2}. id={token_id:<6} logprob={_logprob_value(value): .6f} "
        f"token={tokenizer.decode([token_id])!r}"
        for rank, (token_id, value) in enumerate(items, start=1)
    )


def print_environment() -> None:
    names = [
        "VLLM_ROCM_USE_AITER",
        "VLLM_ROCM_USE_AITER_LINEAR",
        "VLLM_ROCM_USE_AITER_MOE",
        "VLLM_ROCM_USE_AITER_RMSNORM",
        "VLLM_ROCM_USE_SKINNY_GEMM",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ]
    print("Environment:")
    for name in names:
        print(f"  {name}={os.environ.get(name, '<unset>')}")


def compare_outputs(
    hf_ids: list[int],
    hf_text: str,
    hf_logprobs: list[dict[int, float]],
    vllm_ids: list[int],
    vllm_text: str,
    vllm_logprobs: list[Mapping[int, Any]],
    tokenizer: Any,
    continue_after_mismatch: bool,
    prompt_index: int,
) -> bool:
    print(f"\nPrompt {prompt_index}:")
    print(f"HF output ids:   {hf_ids}")
    print(f"vLLM output ids: {vllm_ids}")
    print(f"HF text:   {hf_text!r}")
    print(f"vLLM text: {vllm_text!r}")

    mismatches = 0
    for idx, (hf_id, vllm_id) in enumerate(zip(hf_ids, vllm_ids)):
        if hf_id == vllm_id:
            continue

        mismatches += 1
        hf_topk = hf_logprobs[idx]
        vllm_topk = vllm_logprobs[idx]
        hf_in_vllm_topk = hf_id in vllm_topk
        vllm_in_hf_topk = vllm_id in hf_topk
        is_topk_compatible = hf_in_vllm_topk and vllm_in_hf_topk

        if is_topk_compatible:
            print(
                f"\nGenerated-token mismatch at step {idx} is top-k compatible:"
            )
        else:
            print(f"\nFirst top-k violation at generated step {idx}:")
        print(f"  matched prefix ids: {hf_ids[:idx]}")
        print(f"  HF token:   id={hf_id} token={tokenizer.decode([hf_id])!r}")
        print(f"  vLLM token: id={vllm_id} token={tokenizer.decode([vllm_id])!r}")
        print(f"  HF token in vLLM top-k: {hf_in_vllm_topk}")
        print(f"  vLLM token in HF top-k: {vllm_in_hf_topk}")

        if not is_topk_compatible:
            print("\nHF top-k:")
            print(format_topk(hf_topk, tokenizer))
            print("\nvLLM top-k:")
            print(format_topk(vllm_topk, tokenizer))
            return False
        if not continue_after_mismatch:
            return True

    same_length = len(hf_ids) == len(vllm_ids)
    if same_length:
        if mismatches:
            print(f"\nNo top-k violation found across {mismatches} token mismatches.")
        else:
            print("\nNo generated-token mismatch found.")
    else:
        print(f"\nOutput length mismatch: HF={len(hf_ids)} vLLM={len(vllm_ids)}")
    return same_length


def compare_batch_outputs(
    hf_outputs: list[tuple[list[int], str, list[dict[int, float]]]],
    vllm_outputs: list[tuple[list[int], str, list[Mapping[int, Any]]]],
    tokenizer: Any,
    continue_after_mismatch: bool,
) -> bool:
    ok = True
    for prompt_index, (hf_output, vllm_output) in enumerate(
        zip(hf_outputs, vllm_outputs)
    ):
        hf_ids, hf_text, hf_logprobs = hf_output
        vllm_ids, vllm_text, vllm_logprobs = vllm_output
        prompt_ok = compare_outputs(
            hf_ids,
            hf_text,
            hf_logprobs,
            vllm_ids,
            vllm_text,
            vllm_logprobs,
            tokenizer,
            continue_after_mismatch,
            prompt_index,
        )
        ok = ok and prompt_ok
    return ok


def main() -> int:
    args = parse_args()
    configure_environment(args)
    configure_torch_for_rocm_hf()
    print_environment()
    prompts = load_prompts(args)
    print(f"\nLoaded {len(prompts)} prompt(s)")

    ok = True
    for repeat_index in range(args.repeat):
        print(f"\n=== Iteration {repeat_index + 1}/{args.repeat} ===")
        hf_outputs, tokenizer = hf_generate_logprobs(
            args.model, prompts, args.max_tokens, args.num_logprobs
        )
        vllm_outputs = vllm_generate_logprobs(
            args.model, prompts, args.max_tokens, args.num_logprobs
        )
        iteration_ok = compare_batch_outputs(
            hf_outputs,
            vllm_outputs,
            tokenizer,
            args.continue_after_mismatch,
        )
        ok = ok and iteration_ok
        if not iteration_ok and not args.continue_after_mismatch:
            break

    if ok:
        print("\nResult: PASS")
        return 0
    print("\nResult: FAIL")
    return 0 if args.no_fail else 1


if __name__ == "__main__":
    sys.exit(main())
