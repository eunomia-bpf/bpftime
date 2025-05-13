# train_mnist_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 10 # 训练轮数
LR = 0.01   # 学习率
MNIST_DATA_PATH = './data'
MODEL_SAVE_PATH_PYTORCH = './mnist_mlp_pytorch.pth'
WEIGHTS_SAVE_PATH_CPP = './mlp_weights_for_cpp/' # 给C++用的权重目录

# 确保权重目录存在
os.makedirs(WEIGHTS_SAVE_PATH_CPP, exist_ok=True)

# --- MLP Model Definition (Input: 784, Hidden: 128, Output: 10) ---
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28*28) # 将图像展平
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def save_mlp_weights_for_cpp(model, path_prefix):
    print(f"\n--- Saving MLP weights for C++ to {path_prefix} ---")
    # FC1
    w1 = model.fc1.weight.data.cpu().numpy()
    b1 = model.fc1.bias.data.cpu().numpy()
    w1.astype(np.float32).tofile(os.path.join(path_prefix, "fc1_weights.bin"))
    b1.astype(np.float32).tofile(os.path.join(path_prefix, "fc1_bias.bin"))
    print(f"FC1: weights shape {w1.shape} (saved as {w1.shape[0]}x{w1.shape[1]}), biases shape {b1.shape}")

    # FC2
    w2 = model.fc2.weight.data.cpu().numpy()
    b2 = model.fc2.bias.data.cpu().numpy()
    w2.astype(np.float32).tofile(os.path.join(path_prefix, "fc2_weights.bin"))
    b2.astype(np.float32).tofile(os.path.join(path_prefix, "fc2_bias.bin"))
    print(f"FC2: weights shape {w2.shape} (saved as {w2.shape[0]}x{w2.shape[1]}), biases shape {b2.shape}")
    print("MLP weights for C++ saved.")


def main():
    print(f"Using device: {DEVICE}")

    # --- Data Loading and Preprocessing ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST数据集的均值和标准差
    ])

    train_dataset = torchvision.datasets.MNIST(root=MNIST_DATA_PATH, train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.MNIST(root=MNIST_DATA_PATH, train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Model, Criterion, Optimizer ---
    model = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss() # CrossEntropyLoss内部包含了Softmax
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    print("--- Starting Training ---")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            if (i + 1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        print(f"Epoch [{epoch+1}/{EPOCHS}] COMPLETED: Avg Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

    print("--- Finished Training ---")

    # --- Evaluate Model ---
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'\nAccuracy of the model on the {total} test images: {accuracy:.2f} %')

    # --- Save Model for PyTorch Inference ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH_PYTORCH)
    print(f"\nPyTorch model state_dict saved to {MODEL_SAVE_PATH_PYTORCH}")

    # --- Save Weights for C++ Inference ---
    save_mlp_weights_for_cpp(model, WEIGHTS_SAVE_PATH_CPP)

    print("\nTraining script finished.")

if __name__ == '__main__':
    main()
