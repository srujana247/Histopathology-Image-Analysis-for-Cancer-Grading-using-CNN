import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# --------------------------
# CONFIG
# --------------------------
DATA_DIR = "data"  # Your dataset path
BATCH_SIZE = 32
EPOCHS = 20
IMG_SIZE = 224
SAVE_DIR = "models"  # Folder to save trained models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --------------------------
# DATASET
# --------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
test_data  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", train_data.classes)

# --------------------------
# CNN FROM SCRATCH
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * (IMG_SIZE//8) * (IMG_SIZE//8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --------------------------
# TRAINING FUNCTION
# --------------------------
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(model, train_loader, test_loader, name, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    train_losses, train_accs = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        

        print(f"[{name}] Epoch {epoch+1}/{EPOCHS}, "
              f"Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

    # Plot training curves
    plt.figure()
    plt.plot(range(1, EPOCHS+1), train_losses, label="Loss")
    plt.plot(range(1, EPOCHS+1), train_accs, label="Accuracy")
    plt.title(f"{name} Training Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"{name}_training_curve.png")   # saves graph as image
    plt.show()

    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
    acc = correct / len(test_loader.dataset)
    print(f"[{name}] Test Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"[{name}] Model saved at {save_path}\n")
    # ----------------------
    # Classification Report + Confusion Matrix
    # ----------------------
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print(f"\nClassification Report for {name}:")
    print(classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_loader.dataset.classes,
                yticklabels=test_loader.dataset.classes)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.show()

    return acc


# --------------------------
# RUN CNN
# --------------------------
cnn_model = SimpleCNN(num_classes=len(train_data.classes)).to(device)
cnn_acc = train_model(cnn_model, train_loader, test_loader, "CNN",
                      os.path.join(SAVE_DIR, "cnn_model.pth"))

# --------------------------
# RUN RESNET
# --------------------------
resnet_model = models.resnet18(pretrained=True)
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_features, len(train_data.classes))
resnet_model = resnet_model.to(device)

resnet_acc = train_model(resnet_model, train_loader, test_loader, "ResNet18",
                         os.path.join(SAVE_DIR, "resnet18_model.pth"))

# --------------------------
# FINAL COMPARISON
# --------------------------
print("\n📊 Final Results:")
print(f"CNN Test Accuracy: {cnn_acc:.4f}")
print(f"ResNet18 Test Accuracy: {resnet_acc:.4f}")
