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

# model = AutoModelForCausalLM.from_pretrained('/data/huggingface/hub/Llama-2-13b-chat-hf/').to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained('/data/huggingface/hub/Llama-2-13b-chat-hf/')

print(f"process id: {os.getpid()}")

torch.backends.cudnn.enabled = False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    print("stucking, now you can ckpt me")
    while(1):
        pass
