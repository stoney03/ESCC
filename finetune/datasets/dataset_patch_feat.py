import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as transforms


class PatchDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file: CSV file path includes patch path and TME feature path.
            rransform: Optional transform of preprocessed image.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = self.data.iloc[idx, 0]
        clinical_path = self.data.iloc[idx, 1]

        image = Image.open(img_path).convert("RGB")

        clinical_features = pd.read_csv(clinical_path).values
        clinical_features = torch.tensor(clinical_features, dtype=torch.float32).squeeze(0)

        if self.transform:
            image = self.transform(image)

        return image, clinical_features

if __name__ == '__main__':

    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataset = PatchDataset(csv_file='./patch_feat_paths.csv',transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


    for images, clinical_features in dataloader:
        print(images.shape)
        print(clinical_features.shape)
        break
