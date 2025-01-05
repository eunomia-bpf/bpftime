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
import llama
import torch
import transformers
import time
import pandas as pd
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer


model_path = './model'
tokenizer_path = './tokenizer'
torch.backends.cudnn.enabled = False
print(f"process id: {os.getpid()}")


class TextDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.labels = []
        self.input_ids = []
        self.attn_masks = []
        for txt in txt_list:
            encodings_dict = tokenizer(txt, truncation = True, max_length = max_length, padding = "max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx): return self.input_ids[idx], self.attn_masks[idx]


def train():
    # load model and tokenizer
    print("loading model and tokenizer...")
    model = llama.LLaMAForCausalLM.from_pretrained(model_path).cuda()
    tokenizer = llama.LLaMATokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    # load datasets
    print("loading datasets...")
    texts = pd.read_csv("./dataset/elon_musk_tweets.csv")['text']
    dataset = TextDataset(texts, tokenizer, max_length = max([len(tokenizer.encode(text)) for text in texts]))
    train_dataset, val_dataset = random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])

    # setup train process
    training_args = TrainingArguments(  
                        save_steps = 5000,
                        warmup_steps = 10,
                        logging_steps = 100,
                        weight_decay = 0.05,
                        num_train_epochs = 1,
                        logging_dir = './logs',
                        output_dir = './results',
                        per_device_eval_batch_size = 1,
                        per_device_train_batch_size = 1
                    )

    # trainer
    print("start training...")
    Trainer(
        model = model,
        args = training_args,
        eval_dataset = val_dataset,
        train_dataset = train_dataset,
        data_collator = lambda data: {'input_ids': torch.stack([f[0] for f in data]), 'attention_mask': torch.stack([f[1] for f in data]), 'labels': torch.stack([f[0] for f in data])}
    ).train()

    # test outputs
    print("train done...")
    sample_outputs = model.generate(
                        tokenizer('', return_tensors="pt").input_ids.cuda(),
                        do_sample = True,
                        top_k = 50,
                        max_length = 300,
                        top_p = 0.95,
                        temperature = 1.0
                    )
    print(sample_outputs)


if __name__ == '__main__':
    train()
