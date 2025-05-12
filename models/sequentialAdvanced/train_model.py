import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import torch
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

    def dataloader_to_keras(dataloader):
        images, labels = [], []
        for batch in dataloader:
            images.append(batch[0].numpy())
            labels.append(batch[1].numpy())

        images = np.concatenate(images)
        images = np.transpose(images, (0, 2, 3, 1))  # Из (N,C,H,W) в (N,H,W,C)
        labels = np.concatenate(labels)

        # Convert integer labels to one-hot encoded format
        labels = keras.utils.to_categorical(labels, num_classes=num_classes)

        return images, labels

    x_train, y_train = dataloader_to_keras(train_loader)
    x_val, y_val = dataloader_to_keras(val_loader)

    def create_sequentialAdvanced_model(num_classes, input_shape=(224, 224, 3)):
        model = keras.Sequential([
            # Первый блок сверток
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.2),

            # Второй блок сверток
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),

            # Третий блок сверток
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.4),

            # Четвертый блок сверток
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.4),

            # Полносвязные слои
            layers.Flatten(),
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(num_classes, activation='softmax')
        ])

        # Компиляция модели с дополнительными метриками
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )

        return model


    model = create_sequentialAdvanced_model(num_classes)
    model.summary()


    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    package_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(package_dir, "saved_model")
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'plant_mlp_model.h5'))

    y_true = np.argmax(y_val, axis=1)
    y_pred = np.argmax(model.predict(x_val), axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title("Матрица ошибок")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.show()

    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

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

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "training_metrics.png"), dpi=300)
    plt.show()