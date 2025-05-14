#!/usr/bin/env python3
# hybrid_llama_inference.py
#
# Copyright 2024  The PhoenixOS Authors
# Licensed under the Apache 2.0 license.

import os
import time
import torch
import threading
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

# ---------- configuration ----------------------------------------------------
MODEL_PATH     = "../cuda-probe-test/llama2-13b-chat-hf/model"
TOKENIZER_PATH = "../cuda-probe-test/llama2-13b-chat-hf/tokenizer"
SPLIT_LAYER    = 5          # 0-based index: first GPU layer
GPU_DEVICE     = 0          # CUDA device ID
DTYPE          = torch.float16

# Example user prompt
EXAMPLE_PROMPT = """
In a quiet village nestled between two mountains, a young girl named Lila
... (your prompt here) ...
""".strip()

SYSTEM_PROMPT = (
    "Act as an expert storyteller.  Create detailed, engaging characters "
    "with distinct personalities, motivations and arcs that drive the plot."
)

# ---------- helper class -----------------------------------------------------
class HybridLLaMAInference:
    def __init__(self, model_path, tokenizer_path,
                 split_layer=SPLIT_LAYER, gpu_device=GPU_DEVICE,
                 torch_dtype=DTYPE):

        # Build a device-map: first `split_layer` blocks on CPU, rest on GPU
        self.device_map = {
            **{f"model.layers.{i}": "cpu"   for i in range(split_layer)},
            **{f"model.layers.{i}": gpu_device for i in range(split_layer, 40)},
            "model.embed_tokens": "cpu",
            "model.norm":          gpu_device,
            "lm_head":             gpu_device,
        }

        print("→ loading model …")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=False,
        )
        torch.backends.cudnn.enabled = True

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        print(f"Process PID:      {os.getpid()}")
        print(f"Layers 0-{split_layer-1}: CPU")
        print(f"Layers {split_layer}-end: GPU:{gpu_device}")
        print("Model loaded ✔︎\n")

    # -------------------------------------------------------------------------
    def _wrap_prompt(self, user_prompt: str, system_prompt: str) -> str:
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS,  E_SYS  = "<<SYS>>\n", "\n<</SYS>>\n\n"
        return f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{user_prompt.strip()} {E_INST}\n\n"

    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def infer(self, user_prompt: str, batch_size: int = 1,
              max_new_tokens: int = 256, stream: bool = True):

        prompt = self._wrap_prompt(user_prompt, SYSTEM_PROMPT)
        inputs = self.tokenizer(
            [prompt] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        # Use iterator-based streamer for token-by-token output
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Launch generation in a separate thread
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        gen_thread.start()

        # Consume tokens and report throughput each second
        token_count = 0
        start_time = time.time()
        last_print = start_time

        for token in streamer:
            print(token, end="", flush=True)
            token_count += 1

            now = time.time()
            if now - last_print >= 1.0:
                elapsed = now - start_time
                tps = token_count / elapsed
                print(f"\n[STATS] {tps:.2f} tokens/s\n", end="", flush=True)
                last_print = now

        # Wait for generation to finish
        gen_thread.join()

        # Final stats
        total_elapsed = time.time() - start_time
        print(f"\n[STATS] total tokens  : {token_count}")
        print(f"[STATS] total duration: {total_elapsed:.2f} s")
        print(f"[STATS] avg tps        : {token_count/total_elapsed:.2f} tok/s")
        return None


# ---------- main -------------------------------------------------------------
def main():
    engine = HybridLLaMAInference(
        MODEL_PATH, TOKENIZER_PATH,
        split_layer=SPLIT_LAYER,
        gpu_device=GPU_DEVICE,
        torch_dtype=DTYPE,
    )

    print("=== Demo run =====================================================")
    engine.infer(EXAMPLE_PROMPT, batch_size=1, max_new_tokens=256)
    print("=================================================================\n")


if __name__ == "__main__":
    main()
