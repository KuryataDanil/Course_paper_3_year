import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import json

from torch.utils.data import Dataset

def test_model():
    TEST_PATH = "./test"
    test_images = [f for f in os.listdir(TEST_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class TestDataset(Dataset):
        def __init__(self, image_paths, transform):
            self.image_paths = image_paths
            self.transform = transform

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, os.path.basename(image_path)

    test_image_paths = [os.path.join(TEST_PATH, img) for img in test_images]
    test_dataset = TestDataset(test_image_paths, test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    MODEL_PATH = "./model/resnet18_full_model.pt"
    CLASSES_PATH = "./model/classes.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH)
    model = model.to(device)
    model.eval()

    with open(CLASSES_PATH, "r") as f:
        CLASSES = json.load(f)

    predictions = []

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            for filename, pred in zip(filenames, preds):
                predictions.append((filename, CLASSES[pred]))

    submission = pd.DataFrame(predictions, columns=["file", "species"])
    submission.to_csv("submission.csv", index=False)
    print("Файл submission.csv успешно создан.")
