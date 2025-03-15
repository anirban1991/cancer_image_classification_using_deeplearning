import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from transformers import ViTImageProcessor
from skimage.util import random_noise

# Define augmentations
augment_transform = A.Compose([
    A.RandomBrightnessContrast(p=0.8),
    A.Rotate(limit=20, p=0.5),
    ToTensorV2()
])

# Leukemia
augment_transform_class6 = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.2, -0.1), contrast_limit=(0.1, 0.2), p=0.8),
    A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02, p=0.6),
    A.Sharpen(p=0.5),
    A.ElasticTransform(alpha=1, sigma=50, p=0.5),
    ToTensorV2()
])

# Breast cancer
def augment_breast_cancer(image):
    noisy_image = random_noise(image, mode='speckle', var=0.05)
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.Equalize(p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, p=0.5),
        A.GridDistortion(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.HorizontalFlip(p=0.3),
        ToTensorV2()
    ])
    augmented = transform(image=np.array(noisy_image * 255, dtype=np.uint8))['image']
    return augmented

augment_transform_class4 = augment_breast_cancer

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
                        if label == 4:
                            augment_transform_class4(image=image)
                        elif label ==6:
                            augment_transform(image=image)["image"]
                        else:
                            augmented_img = augment_transform_class6(image=image)["image"]
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