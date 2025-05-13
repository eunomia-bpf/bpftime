# infer_mnist_pytorch.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset # 导入 Subset
import os

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128 # 推理时可以设置一个合理的批大小
MNIST_DATA_PATH = '../train/data'
MODEL_LOAD_PATH_PYTORCH = '../train/mnist_mlp_pytorch.pth'
MAX_IMAGES_TO_TEST_PYTORCH = 2000 # 新增：最多测试的图片数量

# --- MLP Model Definition (必须与训练时模型结构一致) ---
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    print(f"Using device: {DEVICE}")

    # --- Data Loading and Preprocessing ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_test_dataset = torchvision.datasets.MNIST(
        root=MNIST_DATA_PATH,
        train=False,
        download=True,
        transform=transform
    )

    # --- Create a Subset of the test dataset ---
    num_available_test_images = len(full_test_dataset)
    actual_images_to_test = min(num_available_test_images, MAX_IMAGES_TO_TEST_PYTORCH)
    
    print(f"Total available test images: {num_available_test_images}")
    print(f"Will test a maximum of {MAX_IMAGES_TO_TEST_PYTORCH} images.")
    print(f"Actual number of images to test: {actual_images_to_test}")

    if actual_images_to_test == 0:
        print("No images to test. Exiting.")
        return

    # Create indices for the subset
    indices = list(range(actual_images_to_test))
    subset_test_dataset = Subset(full_test_dataset, indices)

    test_loader = DataLoader(
        dataset=subset_test_dataset, # 使用子数据集
        batch_size=BATCH_SIZE,
        shuffle=False, # 推理时通常不需要打乱
        num_workers=2
    )

    # --- Load Model ---
    model = MLP().to(DEVICE)
    if not os.path.exists(MODEL_LOAD_PATH_PYTORCH):
        print(f"Error: Model file not found at {MODEL_LOAD_PATH_PYTORCH}")
        print("Please run the training script first (train_mnist_pytorch.py).")
        return
        
    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH_PYTORCH, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the model architecture in this script matches the saved model.")
        return
        
    model.eval() # 设置为评估模式
    print(f"Model loaded from {MODEL_LOAD_PATH_PYTORCH}")

    print("\n--- Starting PyTorch Inference ---")
    correct = 0
    total = 0 # total 应该等于 actual_images_to_test
    with torch.no_grad(): # 推理时不需要计算梯度
        for images, labels in test_loader: # test_loader 现在只包含子集数据
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 0.0
    if total > 0 :
        accuracy = 100 * correct / total
        print(f'\nAccuracy of the loaded model on {total} test images: {accuracy:.2f} % ({correct}/{total})')
    else:
        print("No images were processed by the test_loader.")


    # 推理单个样本示例 (从子集中取第一个，如果子集非空)
    if len(subset_test_dataset) > 0:
        sample_image, sample_label = subset_test_dataset[0] # 从子数据集中获取
        sample_image_gpu = sample_image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(sample_image_gpu)
            _, predicted_class = torch.max(output, 1)
        print(f"\nSingle sample inference (from the tested subset):")
        print(f"  True label: {sample_label}, Predicted: {predicted_class.item()}")

    print("\nPyTorch inference script finished.")

if __name__ == '__main__':
    main()
