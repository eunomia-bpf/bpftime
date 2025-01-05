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


model = AutoModelForCausalLM.from_pretrained('/data/huggingface/hub/Llama-2-13b-chat-hf/')
tokenizer = AutoTokenizer.from_pretrained('/data/huggingface/hub/Llama-2-13b-chat-hf/')

# fp16
model = model.half()

# copy to device
model = model.to('cuda:0')

print(f"process id: {os.getpid()}")

torch.backends.cudnn.enabled = False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def infer(user_prompt, batch_size=1):
    system_prompt = "Act as an expert in writing captivating stories. Your task is to create detailed and engaging characters for a story based on the following abstract. Each character should have a distinct personality, background, motivations, and development arc that aligns with the story's themes and direction. Consider how these characters interact with each other and how their individual journeys contribute to the overall narrative. Make sure to include both protagonists and antagonists, giving each a unique voice and perspective. Your characters should be relatable and compelling to the readers, driving the story forward and enhancing its emotional impact.\n"

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    ttft_start_time = time.time()
    prompt = f"{B_INST} {B_SYS}{system_prompt.strip()}{E_SYS}{user_prompt.strip()} {E_INST}\n\n"
    inputs = tokenizer([prompt for _ in range(0, batch_size)], return_tensors="pt", return_token_type_ids=False).to(device)

    lengths = inputs['input_ids'].shape[1]
    max_sequence_length = max(inputs['input_ids'].shape[0], lengths)

    generated_texts = model.generate(**inputs, max_length=max_sequence_length+1)
    # generated_texts = model.generate(**inputs, max_length=512)
    ttft_end_time = time.time()

    ttft_time = ttft_end_time - ttft_start_time
    print(f'[STATISTICS] TTFT: {ttft_time:.2f} s')
    return


if __name__ == '__main__':
    user_prompt = "In a quiet village nestled between two mountains, a young girl named Lila discovers an ancient, shimmering stone that grants her the ability to communicate with the stars. As she learns their secrets, she finds herself drawn into a cosmic conflict between light and darkness. With the fate of her village hanging in the balance, Lila must unite her community and harness the power of the stars to restore harmony before the shadows consume everything she loves."

    infer(user_prompt=user_prompt, batch_size=1)
    print("\n\n\n")
