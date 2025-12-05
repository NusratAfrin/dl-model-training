# ================================================================
# MNIST 3-Layer Convolutional Neural Network (CNN) Training Script
# ---------------------------------------------------------------
# This script builds, trains, and evaluates a 3-layer CNN using the
# MNIST handwritten digits dataset. It performs the following:
#
# 1. Loads and preprocesses MNIST (normalization + batching)
# 2. Defines a simple 3-layer CNN architecture with:
#       - 3 convolutional layers (ReLU + MaxPool)
#       - Fully connected layers for classification
# 3. Trains the model using cross-entropy loss and Adam optimizer
# 4. Evaluates accuracy and loss on the test dataset each epoch
#
# This version runs entirely on CPU — no CUDA required.
# ================================================================

# IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# LOAD MNIST DATA
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)


# DEFINE 3-LAYER CNN
class CNN3Layer(nn.Module):
    def __init__(self):
        super(CNN3Layer, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # -> 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> 64x14x14
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # -> 128x7x7

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))  # (32, 28, 28)
        x = self.pool(x)              # (32, 14, 14)

        x = self.relu(self.conv2(x))  # (64, 14, 14)
        x = self.pool(x)              # (64, 7, 7)

        x = self.relu(self.conv3(x))  # (128, 7, 7)

        x = x.view(x.size(0), -1)     # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# TRAIN FUNCTION
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total


# EVALUATION FUNCTION
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total


# TRAIN THE MODEL
model = CNN3Layer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f"Epoch {epoch+1}/{num_epochs} "
          f"- Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
          f"- Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
