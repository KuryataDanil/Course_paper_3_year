import json
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.decorators import execution_animation, execution_time


@execution_time()
@execution_animation()
def test_model():
    test_path = "./test"
    test_images = [f for f in os.listdir(test_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

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

    test_image_paths = [os.path.join(test_path, img) for img in test_images]
    test_dataset = TestDataset(test_image_paths, test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    model_path = "./model/resnet18_full_model.pt"
    classes_path = "./model/classes.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, weights_only=False)
    model = model.to(device)
    model.eval()

    with open(classes_path, "r", encoding='utf-8') as f:
        classes = json.load(f)

    predictions = []

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            for filename, pred in zip(filenames, preds):
                predictions.append((filename, classes[str(pred)]))

    submission = pd.DataFrame(predictions, columns=["file", "species"])
    submission.to_csv("./output/submission.csv", index=False)
    print("Файл submission.csv успешно создан.")
