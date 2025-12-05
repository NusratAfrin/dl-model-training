# ================================================================
# Train VGG19 on CIFAR-10 (CPU-friendly version)
# ================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset


# -------- CIFAR-10 LOADER ----------
def get_loaders(batch=64, subset_size=8000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    # Subset for faster CPU training
    train_data = Subset(train_data, list(range(subset_size)))

    train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch, shuffle=False)
    return train_loader, test_loader


# -------- TRAINING ----------
def train_one_epoch(model, loader, loss_fn, opt):
    model.train()
    correct, total, total_loss = 0, 0, 0

    for x, y in loader:
        out = model(x)
        loss = loss_fn(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    return total_loss / total, correct / total


def evaluate(model, loader, loss_fn):
    model.eval()
    correct, total, total_loss = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            loss = loss_fn(out, y)

            total_loss += loss.item() * x.size(0)
            _, pred = out.max(1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    return total_loss / total, correct / total


# -------- MAIN ----------
def main():
    train_loader, test_loader = get_loaders()

    model = models.vgg19(pretrained=False)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)

    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(2):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, opt)
        te_loss, te_acc = evaluate(model, test_loader, loss_fn)

        print(f"Epoch {epoch+1} | Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f}")


if __name__ == "__main__":
    main()
