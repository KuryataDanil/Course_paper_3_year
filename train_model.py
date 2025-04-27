import os
import json  # Добавлен импорт json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def train_and_save_model():
    data_path = "./train"
    classes = sorted(os.listdir(data_path))

    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    idx_to_class = {str(idx): class_name for idx, class_name in enumerate(classes)}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Обучающих изображений: {len(train_dataset)}, валидационных: {len(val_dataset)}")
    print(f"Классы: {classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    train_losses, val_losses, train_acc, val_acc = [], [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_acc.append(100 * correct / total)

        model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_losses.append(running_loss / len(val_loader))
        val_acc.append(100 * correct / total)

        print(
            f"Эпоха {epoch + 1}/{epochs}: Потери {train_losses[-1]:.4f}, Вал. потери {val_losses[-1]:.4f}, Точность {train_acc[-1]:.2f}%, Вал. точность {val_acc[-1]:.2f}%")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.title("График потерь")

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title("График точности")

    plt.show()

    output_dir = "./output_images"
    os.makedirs(output_dir, exist_ok=True)

    y_true, y_pred = [], []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred, target_names=classes))

    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title("Матрица ошибок")

    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.show()

    model_dir = "./model"
    os.makedirs(model_dir, exist_ok=True)

    torch.save(model, os.path.join(model_dir, "resnet18_full_model.pt"))

    torch.save(model.state_dict(), os.path.join(model_dir, "resnet18_weights.pt"))

    with open(os.path.join(model_dir, "classes.json"), 'w', encoding='utf-8') as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=4)

    print(f"Модель и файл classes.json успешно сохранены в {model_dir}")


if __name__ == "__main__":
    train_and_save_model()