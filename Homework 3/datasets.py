# datasets.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FacesDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

        # Get list of unique image names
        self.image_names = self.dataframe['image_name'].unique()

        # Group bounding boxes by image name
        self.groups = self.dataframe.groupby('image_name')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, image_name)
        image = np.array(Image.open(img_path).convert("RGB"))

        # Get bounding boxes
        records = self.groups.get_group(image_name)
        boxes = records[['x0', 'y0', 'x1', 'y1']].values

        # Select the first bounding box (assuming at least one exists)
        boxes = boxes[0]  # Shape: [4]

        # Apply transformations
        sample = {
            'image': image,
            'bboxes': [boxes],  # Albumentations expects a list of bboxes
            'labels': [0]  # Dummy labels for albumentations lib
        }

        if self.transform:
            sample = self.transform(**sample)

        # Convert to tensors
        image = sample['image']
        boxes = torch.tensor(sample['bboxes'][0])  # Shape: [4]

        return image, boxes
