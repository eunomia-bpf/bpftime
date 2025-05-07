#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim

class DummyModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super(DummyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

def infinite_train_loop():
    # Select device: GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, loss, and optimizer
    model = DummyModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    iteration = 0
    while True:
        iteration += 1
        # Generate random input and target tensors
        inputs = torch.randn(64, 100, device=device)
        targets = torch.randn(64, 10, device=device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print status every 100 iterations
        if iteration % 100 == 0:
            print(f"[Iteration {iteration:08d}] loss = {loss.item():.6f}")

if __name__ == "__main__":
    try:
        infinite_train_loop()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
