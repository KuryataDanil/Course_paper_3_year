import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import torch
import tensorflow as tf
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split




def train_model():
    image_size = (224, 224)
    input_shape = (*image_size, 3)
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    output_dir = "./output/sequential/output_images/"

    package_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(package_dir, "../../data", "train")
    classes = sorted(os.listdir(data_path))
    num_classes = len(classes)

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

    def dataloader_to_keras(dataloader):
        images, labels = [], []
        for batch in dataloader:
            images.append(batch[0].numpy())
            labels.append(batch[1].numpy())

        images = np.concatenate(images)
        images = np.transpose(images, (0, 2, 3, 1))  # Из (N,C,H,W) в (N,H,W,C)
        labels = np.concatenate(labels)
        return images, labels

    x_train, y_train = dataloader_to_keras(train_loader)
    x_val, y_val = dataloader_to_keras(val_loader)

    def create_mlp_model():
        model = keras.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    model = create_mlp_model()
    model.summary()

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Сохранение модели
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'plant_mlp_model.h5'))

    # Оценка модели
    y_pred = np.argmax(model.predict(x_val), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=classes))

    # Визуализация
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Графики обучения
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()

    plt.savefig(os.path.join(output_dir, "training_metrics.png"), dpi=300)
    plt.close()
