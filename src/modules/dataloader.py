import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import hashlib
import time
import sys

class DataLoader:
    def __init__(self, img_size):
        self.IMG_SIZE = img_size

    def load_data(self):
        # Paths for datasets
        brain_cancer_path = "/kaggle/input/brain-tumor-mri-dataset"
        breast_cancer_path = "/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT"
        melanoma_path = "/kaggle/input/melanoma-skin-cancer-dataset-of-10000-images/melanoma_cancer_dataset"
        leukemia_path = "/kaggle/input/leukemia-classification/C-NMC_Leukemia"

        # Categories mapping
        brain_classes = {
            "glioma": 1,
            "meningioma": 2,
            "pituitary": 3,
            "no_tumor": 0
        }

        breast_classes = {
            "benign": 0,
            "malignant": 4,
            "normal": 0
        }

        melanoma_classes = {
            "malignant": 5,
            "benign": 0
        }

        leukemia_classes = {
            "hem": 0,
            "all": 6
        }

        Hash = hashlib.sha512
        MAX_HASH_PLUS_ONE = 2**(Hash().digest_size * 8)
        def str_to_probability(in_str):
            """Return a reproducible uniformly random float in the interval [0, 1) for the given string."""
            seed = in_str.encode()
            hash_digest = Hash(seed).digest()
            hash_int = int.from_bytes(hash_digest, 'big')  # Uses explicit byteorder for system-agnostic reproducibility
            return hash_int / MAX_HASH_PLUS_ONE  # Float division

        # Function to load and preprocess images
        def load_images_from_folder(folder, label_mapping):
            images, labels = [], []
            i = 0
            for class_folder in os.listdir(folder):
                class_path = os.path.join(folder, class_folder)
                if os.path.isdir(class_path) and class_folder in label_mapping:
                    label = label_mapping[class_folder]
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        img = cv2.imread(img_path)
                        if "Leukemia" in img_path:
                            if img is not None and str_to_probability(img_path) <0.2:
                                i += 1
                                print(i, img_path, end="\r")
                                time.sleep(0.005)
                                img = cv2.resize(img, self.IMG_SIZE)
                                images.append(img)
                                labels.append(label)
                        else:
                            i += 1
                            print(i, end="\r")
                            time.sleep(0.005)
                            img = cv2.resize(img, self.IMG_SIZE)
                            images.append(img)
                            labels.append(label)
            return images, labels
        # Function to load and preprocess images

        # Load training and testing data
        X_train, y_train, X_valid, y_valid, X_test, y_test = [], [], [], [], [], []

        # 1. **Brain Cancer Data**
        train_brain_path = os.path.join(brain_cancer_path, "Training")
        test_brain_path = os.path.join(brain_cancer_path, "Testing")

        X_train_brain, y_train_brain = load_images_from_folder(train_brain_path, brain_classes)
        X_test_brain, y_test_brain = load_images_from_folder(test_brain_path, brain_classes)
        X_train_brain, X_valid_brain, y_train_brain, y_valid_brain = train_test_split(
            X_train_brain, y_train_brain, test_size=0.1, random_state=42, stratify=y_train_brain)

        X_train.extend(X_train_brain)
        y_train.extend(y_train_brain)
        X_valid.extend(X_valid_brain)
        y_valid.extend(y_valid_brain)
        X_test.extend(X_test_brain)
        y_test.extend(y_test_brain)

        print("Brain Data, Done")

        # 2. **Breast Cancer Data (Needs Splitting)**
        X_breast, y_breast = load_images_from_folder(breast_cancer_path, breast_classes)
        X_train_breast, X_test_breast, y_train_breast, y_test_breast = train_test_split(
            X_breast, y_breast, test_size=0.1, random_state=42, stratify=y_breast)
        X_train_breast, X_valid_breast, y_train_breast, y_valid_breast = train_test_split(
            X_train_breast, y_train_breast, test_size=0.1, random_state=42, stratify=y_train_breast)

        X_train.extend(X_train_breast)
        y_train.extend(y_train_breast)
        X_valid.extend(X_valid_breast)
        y_valid.extend(y_valid_breast)
        X_test.extend(X_test_breast)
        y_test.extend(y_test_breast)

        print("Breast Data, Done")

        # 3. **Melanoma Data**
        train_melanoma_path = os.path.join(melanoma_path, "train")
        test_melanoma_path = os.path.join(melanoma_path, "test")

        X_train_melanoma, y_train_melanoma = load_images_from_folder(train_melanoma_path, melanoma_classes)
        X_test_melanoma, y_test_melanoma = load_images_from_folder(test_melanoma_path, melanoma_classes)

        X_train_melanoma, X_valid_melanoma, y_train_melanoma, y_valid_melanoma = train_test_split(
            X_train_melanoma, y_train_melanoma, test_size=0.1, random_state=42, stratify=y_train_melanoma)

        X_train.extend(X_train_melanoma)
        y_train.extend(y_train_melanoma)
        X_valid.extend(X_valid_melanoma)
        y_valid.extend(y_valid_melanoma)
        X_test.extend(X_test_melanoma)
        y_test.extend(y_test_melanoma)

        print("Melanoma Data, Done")

        # 4. **Leukemia Data (Splitting Train 95% / Test 5%)**
        train_leukemia_path = os.path.join(leukemia_path, "training_data")

        X_leukemia, y_leukemia = [], []

        # Load all images from the training folder
        for subfolder in os.listdir(train_leukemia_path):
            subfolder_path = os.path.join(train_leukemia_path, subfolder)
            if "fold_0" in subfolder_path:
                continue
            X_temp, y_temp = load_images_from_folder(subfolder_path, leukemia_classes)
            X_leukemia.extend(X_temp)
            y_leukemia.extend(y_temp)

        # Split into 95% train, 5% test
        X_train_leukemia, X_test_leukemia, y_train_leukemia, y_test_leukemia = train_test_split(
            X_leukemia, y_leukemia, test_size=0.1, random_state=42, stratify=y_leukemia)
        X_train_leukemia, X_valid_leukemia, y_train_leukemia, y_valid_leukemia = train_test_split(
            X_train_leukemia, y_train_leukemia, test_size=0.1, random_state=42, stratify=y_train_leukemia)

        # Append to global train/test lists
        X_train.extend(X_train_leukemia)
        y_train.extend(y_train_leukemia)
        X_valid.extend(X_valid_leukemia)
        y_valid.extend(y_valid_leukemia)
        X_test.extend(X_test_leukemia)
        y_test.extend(y_test_leukemia)

        print("Done loading")
        return X_train, y_train, X_valid, y_valid, X_test, y_test