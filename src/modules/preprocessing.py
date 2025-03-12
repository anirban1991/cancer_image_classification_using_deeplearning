import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

# Define augmentations
augment_transform = A.Compose([
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.8)
])

augmentation_factors = {1: 5, 2: 5, 3: 4, 4: 15, 6: 7}  # Augment specific classes

class CancerDataset(Dataset):
    def __init__(self, images, labels, augment=False, pretrained=False):
        # Flatten lists if they are nested
        self.images = images
        self.labels = labels
        self.augment = augment
        self.pretrained = pretrained

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Ensure image is a NumPy array
        if isinstance(image, list):
            image = np.array(image, dtype=np.uint8)

        # Apply augmentation if needed
        if self.augment and label in augmentation_factors:
            augmented_images = [augment_transform(image=image)["image"] for _ in range(augmentation_factors[label])]
            augmented_images = [self.normalize(ToTensorV2()(image=a)["image"]) for a in augmented_images]
            return torch.stack(augmented_images), [label] * augmentation_factors[label]

        # Convert image to tensor (HWC → CHW) and normalize
        image = ToTensorV2()(image=image)["image"]
        image = self.normalize(image)

        return image, label

    def normalize(self, image):
        if self.pretrained:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return (image - mean) / std  # Normalize using ImageNet mean & std
        return image / 255.0  # Scale to [0, 1] if training from scratch


def augment_image(image):
    augmented = augment_transform(image=image)["image"]
    return augmented

class Preprocessor:
    def preprocess(self, X_train, y_train, X_valid, y_valid, X_test, y_test, pretrained=False):
        train_dataset = CancerDataset(X_train, y_train, augment=True, pretrained=pretrained)
        valid_dataset = CancerDataset(X_valid, y_valid, augment=False, pretrained=pretrained)
        test_dataset = CancerDataset(X_test, y_test, augment=False, pretrained=pretrained)

        # Use DataLoader for memory-efficient training
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=lambda x: x)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        return train_loader, valid_loader, test_loader




