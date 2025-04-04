import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision

from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import os

DATA_PATH = "./train"
CLASSES = os.listdir(DATA_PATH)  # Получаем названия классов

print(f"Найдено {len(CLASSES)} классов: {CLASSES}")

import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Обучающих изображений: {len(train_dataset)}, валидационных: {len(val_dataset)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASSES))  # Меняем выходной слой под количество классов

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10
train_losses, val_losses, train_acc, val_acc = [], [], [], []

for epoch in range(EPOCHS):
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

    print(f"Эпоха {epoch+1}/{EPOCHS}: Потери {train_losses[-1]:.4f}, Вал. потери {val_losses[-1]:.4f}, Точность {train_acc[-1]:.2f}%, Вал. точность {val_acc[-1]:.2f}%")

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

from sklearn.metrics import classification_report, confusion_matrix
import os

OUTPUT_DIR = "./output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Создаём папку, если её нет


y_true, y_pred = [], []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Вывод отчёта по метрикам
print(classification_report(y_true, y_pred, target_names=CLASSES))

# Матрица ошибок
conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.title("Матрица ошибок")

# Сохраняем график
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
plt.show()




# Путь к папке с тестовыми изображениями
# TEST_PATH = "./test"
#
# # Получение списка файлов
# test_images = [f for f in os.listdir(TEST_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]
#
# # Подготовка трансформаций (такие же, как для обучающих данных, без аугментаций)
# test_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# # Класс для загрузки тестовых изображений
# class TestDataset(Dataset):
#     def __init__(self, image_paths, transform):
#         self.image_paths = image_paths
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, idx):
#         image_path = self.image_paths[idx]
#         image = Image.open(image_path).convert('RGB')
#         image = self.transform(image)
#         return image, os.path.basename(image_path)
#
# # Полные пути к изображениям
# test_image_paths = [os.path.join(TEST_PATH, img) for img in test_images]
# test_dataset = TestDataset(test_image_paths, test_transform)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
# # Предсказания модели
# model.eval()
# predictions = []
#
# with torch.no_grad():
#     for images, filenames in test_loader:
#         images = images.to(device)
#         outputs = model(images)
#         _, preds = torch.max(outputs, 1)
#         preds = preds.cpu().numpy()
#         for filename, pred in zip(filenames, preds):
#             predictions.append((filename, CLASSES[pred]))
#
# # Сохраняем предсказания в CSV
# submission = pd.DataFrame(predictions, columns=["file", "species"])
# submission.to_csv("submission.csv", index=False)
# print("Файл submission.csv успешно создан.")
