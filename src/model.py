import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from PIL import Image
import os

CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
DATA_DIR = "data/flower_photos"
MODEL_PATH = "models/flowers_model.pth"

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

def build_model(num_classes: int = 5) -> nn.Module:
    """Load EfficientNet-B0, freeze all layers except final classifier."""
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    # Replace final classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def train():
    """Fine-tune EfficientNet-B0 on flowers dataset."""
    train_tf, val_tf = get_transforms()

    dataset = ImageFolder(DATA_DIR, transform=train_tf)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    print(f"Train: {train_size} images | Val: {val_size} images")
    print(f"Classes: {dataset.classes}")

    model = build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

def load_model() -> nn.Module:
    """Load saved model weights."""
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def predict_image(model: nn.Module, image: Image.Image) -> dict:
    """Predict flower class from a PIL image."""
    _, val_tf = get_transforms()
    tensor = val_tf(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        idx     = probs.argmax().item()
    return {
        "class_name": CLASSES[idx],
        "confidence": round(probs[idx].item(), 4)
    }

if __name__ == "__main__":
    train()
