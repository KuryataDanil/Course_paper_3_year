import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


def train_model():
    image_size = (224, 224)
    input_shape = (*image_size, 3)
    batch_size = 32
    output_dir = "./output/random_forest/output_images/"

    package_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(package_dir, "../../data", "train")
    classes = sorted(os.listdir(data_path))
    num_classes = len(classes)

    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    idx_to_class = {str(idx): class_name for idx, class_name in enumerate(classes)}

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Обучающих изображений: {len(train_dataset)}, валидационных: {len(val_dataset)}")
    print(f"Классы: {classes}")

    def dataloader_to_numpy(dataloader):
        images, labels = [], []
        for batch in dataloader:
            images.append(batch[0].numpy())
            labels.append(batch[1].numpy())

        images = np.concatenate(images)
        # Преобразуем изображения в плоский формат (N, H*W*C)
        images = images.reshape(images.shape[0], -1)
        labels = np.concatenate(labels)

        return images, labels

    x_train, y_train = dataloader_to_numpy(train_loader)
    x_val, y_val = dataloader_to_numpy(val_loader)

    # Создаем и обучаем модель Random Forest
    print("Обучение Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)

    # Предсказания на валидационном наборе
    y_pred = rf_model.predict(x_val)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=classes))

    conf_matrix = confusion_matrix(y_val, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title("Матрица ошибок (Random Forest)")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrix_rf.png"), dpi=300)
    plt.show()

    # Сохраняем важности признаков (для демонстрации)
    importances = rf_model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances[:50])), importances[:50])  # Показываем только первые 50 для наглядности
    plt.title("Важность признаков (первые 50, Random Forest)")
    plt.savefig(os.path.join(output_dir, "feature_importances.png"), dpi=300)
    plt.show()