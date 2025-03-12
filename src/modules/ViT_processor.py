import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor

# Define augmentations
augment_transform = A.Compose([
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.8)
])
augmentation_factors = {1: 5, 2: 5, 3: 4, 4: 15, 6: 7}
class ImageClassificationDataset(Dataset):
    def __init__(self, images, labels, processor, augment=False):
        self.images = list(images)
        self.labels = list(labels)
        self.processor = processor
        self.augment = augment
        if augment:
            augmented_images = []
            augmented_labels = []
            for idx in range(len(self.images)):
                image = self.images[idx]
                label = self.labels[idx]
                if label in augmentation_factors:
                    num_augments = augmentation_factors[label]
                    for _ in range(num_augments):
                        augmented_img = augment_transform(image=image)["image"]
                        augmented_images.append(augmented_img)
                        augmented_labels.append(label)
            self.images.extend(augmented_images)
            self.labels.extend(augmented_labels)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        item = self.processor(image, return_tensors="pt")
        item = {key: val.squeeze(0) for key, val in item.items()}
        item['labels'] = torch.tensor(label)
        return item

class VitPreprocessor:
    def preprocess(self, X_train, y_train, X_valid, y_valid, X_test, y_test):
        model_name_or_path = 'google/vit-base-patch16-224-in21k'
        processor = ViTImageProcessor.from_pretrained(model_name_or_path)
        train_dataset = ImageClassificationDataset(X_train, y_train, processor,augment=True)
        valid_dataset = ImageClassificationDataset(X_valid, y_valid, processor,augment=False)
        test_dataset = ImageClassificationDataset(X_test, y_test, processor,augment=False)

        return train_dataset, valid_dataset, test_dataset