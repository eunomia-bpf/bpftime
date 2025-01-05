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
import transformers
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_token = os.getenv('HF_TOKEN')
login(token = hf_token)

model_id = 'meta-llama/Llama-2-7b-chat-hf'
model_path = './model'
tokenizer_path = './tokenizer'

# download model parameter
if not os.path.exists(model_path):
    os.makedirs(model_path)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cpu')
model.save_pretrained(model_path)

# download tokenizer parameter
if not os.path.exists(tokenizer_path):
    os.makedirs(tokenizer_path)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(tokenizer_path)

exit(0)
