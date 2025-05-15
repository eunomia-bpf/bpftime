#!/usr/bin/env python3
"""
best_split_elegant.py

A script to benchmark split-point performance for LLaMA2-13b by copying partial forward pass hidden states
from CPU to GPU and executing the remaining layers on GPU for various two-point splits, including end-to-end latency measurement.
"""
import sys
import time
import logging
import argparse
from itertools import combinations

# Dependency check
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ModuleNotFoundError as e:
    sys.stderr.write(f"Error: Missing required module '{e.name}'. Install with 'pip install {e.name}'\n")
    sys.exit(1)

# Default configurations
default_model_path = "../cuda-probe-test/llama2-13b-chat-hf/model"
default_tokenizer_path = "../cuda-probe-test/llama2-13b-chat-hf/tokenizer"
default_prompt = (
    "In a quiet village nestled between two mountains, a young girl named Lila "
    "discovers an ancient, shimmering stone…"
)
default_max_len = 1024

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def load_models(model_path, tokenizer_path, dtype, device_cpu, device_gpu):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cpu = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        output_hidden_states=True,
        trust_remote_code=False
    ).eval().to(device_cpu)

    model_gpu = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        output_hidden_states=False,
        trust_remote_code=False
    ).eval().to(device_gpu)

    return tokenizer, model_cpu, model_gpu


def get_hidden_states(prompt, tokenizer, model_cpu, max_len, device_cpu):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )
    attention_mask = inputs.get("attention_mask")
    inputs = {k: v.to(device_cpu) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model_cpu(**inputs, use_cache=False)
    return outputs.hidden_states, attention_mask


def measure_split(hidden_states, model_gpu, attention_mask_cpu, i, j, device_gpu):
    # Copy hidden state i from CPU to GPU
    h_i = hidden_states[i]
    t0 = time.time()
    h_gpu = h_i.to(device_gpu)
    torch.cuda.synchronize()
    t_copy = time.time() - t0

    # Ensure mask dtype matches
    if attention_mask_cpu is not None:
        attn_mask = attention_mask_cpu.to(device_gpu).to(h_gpu.dtype)
    else:
        attn_mask = None

    # Middle segment
    t0 = time.time()
    mid = h_gpu
    for layer in model_gpu.model.layers[i:j]:
        out = layer(mid, attention_mask=attn_mask)
        mid = out[0] if isinstance(out, (tuple, list)) else out
    torch.cuda.synchronize()
    t_mid = time.time() - t0

    # Final segment
    t0 = time.time()
    fin = mid
    for layer in model_gpu.model.layers[j:]:
        out = layer(fin, attention_mask=attn_mask)
        fin = out[0] if isinstance(out, (tuple, list)) else out
    torch.cuda.synchronize()
    t_fin = time.time() - t0

    return t_copy, t_mid, t_fin


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLaMA2 split-point performance with E2E timing.")
    parser.add_argument("--model_path", default=default_model_path)
    parser.add_argument("--tokenizer_path", default=default_tokenizer_path)
    parser.add_argument("--prompt", default=default_prompt)
    parser.add_argument("--max_len", type=int, default=default_max_len)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", nargs=2, type=int, metavar=("I", "J"),
                        help="Test a single split point I J instead of all combinations.")
    parser.add_argument("--test", action="store_true", help="Run sanity test and exit.")
    args = parser.parse_args()

    device_cpu = torch.device("cpu")
    device_gpu = torch.device(f"cuda:{args.gpu}")

    tokenizer, model_cpu, model_gpu = load_models(
        args.model_path, args.tokenizer_path, torch.float32, device_cpu, device_gpu
    )

    # Measure CPU forward pass time
    t0_cpu = time.time()
    hidden_states, attention_mask = get_hidden_states(
        args.prompt, tokenizer, model_cpu, args.max_len, device_cpu
    )
    cpu_fwd_time = time.time() - t0_cpu
    logging.info(f"CPU forward pass time: {cpu_fwd_time:.4f}s")

    if args.test:
        logging.info("Running sanity test for split 1,2 with E2E timing")
        t_copy, t_mid, t_fin = measure_split(hidden_states, model_gpu, attention_mask, 1, 2, device_gpu)
        e2e = cpu_fwd_time + t_copy + t_mid + t_fin
        print(f"Sanity test results: cpu={cpu_fwd_time:.4f}s, copy={t_copy:.4f}s, mid={t_mid:.4f}s, fin={t_fin:.4f}s, e2e={e2e:.4f}s")
        sys.exit(0)

    splits = [tuple(args.split)] if args.split else list(combinations(range(1, len(hidden_states)), 2))

    results = []
    for i, j in splits:
        logging.info(f"Measuring split ({i},{j})")
        t_copy, t_mid, t_fin = measure_split(hidden_states, model_gpu, attention_mask, i, j, device_gpu)
        t_e2e = cpu_fwd_time + t_copy + t_mid + t_fin
        results.append((i, j, cpu_fwd_time, t_copy, t_mid, t_fin, t_e2e))
        print(
            f"({i},{j}) e2e={t_e2e:.3f}s  "
            f"cpu={cpu_fwd_time:.3f}s, copy={t_copy:.3f}s, mid={t_mid:.3f}s, fin={t_fin:.3f}s"
        )

    if results:
        best = min(results, key=lambda x: x[-1])
        i, j, *_ , best_e2e = best
        print(f"Best split: ({i},{j}) e2e={best_e2e:.3f}s")

if __name__ == "__main__":
    main()
