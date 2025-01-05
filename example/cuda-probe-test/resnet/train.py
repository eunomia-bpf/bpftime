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
print(f"process id: {os.getpid()}")

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from utils.readData import read_dataset
from utils.ResNet import ResNet50, ResNet101, ResNet152
import time

print(f"process id: {os.getpid()}")

# configurations
torch.backends.cudnn.enabled = False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
assert(device != 'cpu')
criterion = nn.CrossEntropyLoss().to(device)
batch_size = 32
n_epochs = 1
lr = 0.1
print_statistics_per_iter = False

# load models
model = ResNet152()
model = model.to(device)


def run_train():
    iter_durations = []

    train_loader, _, _ = read_dataset(batch_size=batch_size,pic_path='dataset')

    for _ in range(1, n_epochs+1):
        # keep track of training and validation loss
        train_loss = 0.0

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        
        nb_iteration = 0

        model.train()
        for _, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            start_t = time.time()

            # move data to gpu
            data = data.to(device)
            target = target.to(device)
            
            # forward
            output = model(data).to(device)

            # backward
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

            nb_iteration += 1
            
            # NOTE: we force sync here to make sure the execution is done
            torch.cuda.default_stream(0).synchronize()

            end_t = time.time()

            iter_durations.append(int(round((end_t-start_t) * 1000)))
            if print_statistics_per_iter:
                print(f"itetration {nb_iteration} duration: {int(round((end_t-start_t) * 1000))} ms")

            if nb_iteration == 64:
                print(f"reach {nb_iteration}, break")
                break

        np_iter_durations = np.array(iter_durations)
        throughput_list_str = "0, "
        time_list_str = "0, "
        time_accu = 0 #s
        for i, duration in enumerate(np_iter_durations):
            time_accu += duration / 1000
            if i != len(np_iter_durations) - 1:
                throughput_list_str += f"{60000/duration:.2f}, "
                time_list_str += f"{time_accu:.2f}, "
            else:
                throughput_list_str += f"{60000/duration:.2f}"
                time_list_str += f"{time_accu:.2f}"

        print(f"throughput list: {throughput_list_str}")
        print(f"time list: {time_list_str}")
        print(
            f"latency:"
            f" p10({np.percentile(np_iter_durations, 10)} ms), "
            f" p50({np.percentile(np_iter_durations, 50)} ms), "
            f" p99({np.percentile(np_iter_durations, 99)} ms), "
            f" mean({np.mean(np_iter_durations)} ms)"
        )


if __name__ == '__main__':
    run_train()
