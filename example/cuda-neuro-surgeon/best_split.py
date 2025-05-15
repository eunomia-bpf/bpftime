#!/usr/bin/env python3
"""
best_split_hidden.py

A script to benchmark split-point performance for LLaMA2-13b hidden representations
by executing the first i layers on CPU and the remaining layers on GPU for each split (i, j),
measuring end-to-end inference time (CPU + copy + GPU segments).
"""
import sys
import time
import logging
import argparse
import torch
from itertools import combinations
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Default configurations
default_model_path = "../cuda-probe-test/llama2-13b-chat-hf/model"
default_tokenizer_path = "../cuda-probe-test/llama2-13b-chat-hf/tokenizer"
default_prompt = (
    "In a quiet village nestled between two mountains, a young girl named Lila "
    "discovers an ancient, shimmering stone…"
)
default_max_len = 1024

def load_models(model_path, tokenizer_path, dtype, device_cpu, device_gpu):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Full model for CPU (we'll run first i layers on CPU)
    model_cpu = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        output_hidden_states=False,
        trust_remote_code=False
    ).eval().to(device_cpu)

    # Separate instance for GPU (to run layers i:j and j:)
    model_gpu = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        output_hidden_states=False,
        trust_remote_code=False
    ).eval().to(device_gpu)

    # Disable KV cache for consistency
    model_cpu.config.use_cache = False
    model_gpu.config.use_cache = False

    return tokenizer, model_cpu, model_gpu


def measure_split(inputs, attention_mask, tokenizer, model_cpu, model_gpu, split, device_cpu, device_gpu):
    i, j = split
    # --- CPU partial forward up to layer i ---
    start_cpu = time.time()
    input_ids = inputs['input_ids'].to(device_cpu)
    
    # 将 attention_mask 转换为与 model 使用的 dtype 相同
    attn_mask = attention_mask.to(device_cpu)
    
    # embedding + positional
    hidden = model_cpu.model.embed_tokens(input_ids)
    
    # 确保 attention_mask 的类型与 hidden 相同
    attn_mask = attn_mask.to(dtype=hidden.dtype)
    
    # pass first i layers
    for layer in model_cpu.model.layers[:i]:
        hidden = layer(hidden, attention_mask=attn_mask)[0]
    cpu_time = time.time() - start_cpu

    # --- Copy hidden state to GPU ---
    start_copy = time.time()
    h_gpu = hidden.to(device_gpu)
    torch.cuda.synchronize()
    copy_time = time.time() - start_copy

    # --- GPU middle segment i->j ---
    start_mid = time.time()
    # 确保 GPU attention_mask 类型正确
    attn_mask_gpu = attn_mask.to(device_gpu).to(dtype=h_gpu.dtype)
    mid = h_gpu
    for layer in model_gpu.model.layers[i:j]:
        mid = layer(mid, attention_mask=attn_mask_gpu)[0]
    torch.cuda.synchronize()
    mid_time = time.time() - start_mid

    # --- GPU final segment j->end ---
    start_fin = time.time()
    fin = mid
    for layer in model_gpu.model.layers[j:]:
        fin = layer(fin, attention_mask=attn_mask_gpu)[0]
    torch.cuda.synchronize()
    fin_time = time.time() - start_fin

    total_time = cpu_time + copy_time + mid_time + fin_time
    return cpu_time, copy_time, mid_time, fin_time, total_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLaMA2 hidden split-point end-to-end performance.")
    parser.add_argument("--model_path", default=default_model_path)
    parser.add_argument("--tokenizer_path", default=default_tokenizer_path)
    parser.add_argument("--prompt", default=default_prompt)
    parser.add_argument("--max_len", type=int, default=default_max_len)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", nargs=2, type=int, metavar=("I","J"), help="Test single split I J")
    parser.add_argument("--test", action="store_true", help="Sanity test on split (1,2)")
    args = parser.parse_args()

    device_cpu = torch.device("cpu")
    device_gpu = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    tokenizer, model_cpu, model_gpu = load_models(
        args.model_path, args.tokenizer_path, torch.float32, device_cpu, device_gpu
    )

    # Tokenize inputs once
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_len
    )
    attention_mask = inputs.get('attention_mask')

    splits = [tuple(args.split)] if args.split else list(combinations(range(1, len(model_cpu.model.layers)), 2))

    if args.test:
        cpu_t, copy_t, mid_t, fin_t, total = measure_split(
            inputs, attention_mask, tokenizer, model_cpu, model_gpu, (1,2), device_cpu, device_gpu
        )
        print(f"Sanity (1,2): cpu={cpu_t:.4f}s copy={copy_t:.4f}s mid={mid_t:.4f}s fin={fin_t:.4f}s total={total:.4f}s")
        return

    best = None
    for split in splits:
        cpu_t, copy_t, mid_t, fin_t, total = measure_split(
            inputs, attention_mask, tokenizer, model_cpu, model_gpu, split, device_cpu, device_gpu
        )
        print(f"Split {split}: cpu={cpu_t:.3f}s copy={copy_t:.3f}s mid={mid_t:.3f}s fin={fin_t:.3f}s total={total:.3f}s")
        if best is None or total < best[1]:
            best = (split, total)

    print(f"Best split {best[0]} with total={best[1]:.3f}s")

if __name__ == "__main__":
    main()
