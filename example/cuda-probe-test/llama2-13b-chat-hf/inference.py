# Copyright 2024 The PhoenixOS Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import transformers
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from torch.profiler import profile, record_function, ProfilerActivity

coldstart_start_time = time.time()

model = AutoModelForCausalLM.from_pretrained('./model')
tokenizer = AutoTokenizer.from_pretrained('./tokenizer')

# fp16
# model = model.half()

# copy to device
model = model.to('cuda:0')

# model = AutoModelForCausalLM.from_pretrained('/nvme/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb/', ignore_mismatched_sizes=True).to('cuda:0')
# tokenizer = AutoTokenizer.from_pretrained('/nvme/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb/')

print(f"process id: {os.getpid()}")

torch.backends.cudnn.enabled = True
device = 'cuda:0'

def infer(user_prompt, batch_size=1):
    system_prompt = "Act as an expert in writing captivating stories. Your task is to create detailed and engaging characters for a story based on the following abstract. Each character should have a distinct personality, background, motivations, and development arc that aligns with the story's themes and direction. Consider how these characters interact with each other and how their individual journeys contribute to the overall narrative. Make sure to include both protagonists and antagonists, giving each a unique voice and perspective. Your characters should be relatable and compelling to the readers, driving the story forward and enhancing its emotional impact.\n"

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    prompt = f"{B_INST} {B_SYS}{system_prompt.strip()}{E_SYS}{user_prompt.strip()} {E_INST}\n\n"
    inputs = tokenizer([prompt for _ in range(0, batch_size)], return_tensors="pt", return_token_type_ids=False).to(device)

    streamer = TextStreamer(tokenizer)
    coldstart_end_time = time.time()
    print(f'[STATISTICS] coldstart duration: {coldstart_end_time-coldstart_start_time:.2f} s')

    # streaming
    start_time = time.time()
    generated_texts = model.generate(**inputs, streamer=streamer, max_length=1024)
    # generated_texts = model.generate(**inputs, max_length=512)
    end_time = time.time()

    # calculate throughput
    text_length = 0
    for text in generated_texts:
        text_length += list(text.size())[0]
    elapsed_time = end_time - start_time
    throughput = text_length / elapsed_time
    print(f'[STATISTICS] Duration: {elapsed_time:.2f} s')
    print(f'[STATISTICS] #Tokens: {text_length}')
    print(f'[STATISTICS] LatencyPerToken: {elapsed_time/text_length*1000:.2f} ms')
    print(f'[STATISTICS] Throughput: {throughput:.2f} characters per second')

    del inputs, generated_texts

    return


if __name__ == '__main__':
    user_prompt = "In a quiet village nestled between two mountains, a young girl named Lila discovers an ancient, shimmering stone that grants her the ability to communicate with the stars. As she learns their secrets, she finds herself drawn into a cosmic conflict between light and darkness. With the fate of her village hanging in the balance, Lila must unite her community and harness the power of the stars to restore harmony before the shadows consume everything she loves."

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for i in range(0, 1):
            infer(user_prompt=user_prompt, batch_size=1)
            print("\n\n\n")
    prof.export_chrome_trace("trace.json")
