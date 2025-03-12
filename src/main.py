## For the evaluate module you have to first install it using pip install evaluate

import torch
import torch.optim as optim
import torch.nn as nn
import sys
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

sys.path.insert(1, '/kaggle/input/data-load-prep')
from dataloader import DataLoader
from preprocessing import Preprocessor
from ViT_processor import VitPreprocessor

sys.path.insert(1, '/kaggle/input/models')
from baseline import BaselineModel
from CustomCNN import BaselineCNN
from Resnet import ResnetModel
from Ensemble import EnsembleModel
from vision_transformer import Vit


def get_model(model_name):
    if model_name == "Baseline":
        return BaselineModel(224 * 224 * 3)
    elif model_name == "CNN":
        return BaselineCNN(7)
    elif model_name == "resnet":
        return ResnetModel(7)
    elif model_name == "ensemble":
        MLP = BaselineModel(224 * 224 * 3)
        resnet = ResnetModel(7)
        return EnsembleModel(MLP, resnet, 128, 7)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_mapping = {"No cancer": 0, "Glioma": 1, "Meningioma": 2, "Pituitary": 3, "Breast cancer": 4, "Melanoma": 5,
                 "Leukemia": 6}

loader = DataLoader((224, 224))
# X_train, y_train, X_valid, y_valid, X_test, y_test = loader.load_data()

model_name = "Baseline"

if model_name == "Vit":
    vitprocessor = VitPreprocessor()

    train_dataset, valid_dataset, test_dataset = vitprocessor.preprocess(X_train, y_train, X_valid, y_valid, X_test,
                                                                         y_test)

    model = Vit(train_dataset, valid_dataset, test_dataset)

    model.train()

    logs = model.trainer.state.log_history
    train_losses = []
    eval_losses = []
    for i in range(len(logs) - 1):
        if "eval_loss" in logs[i]:
            train_losses.append(logs[i - 1]["loss"])
            eval_losses.append(logs[i]["eval_loss"])

    plt.plot(train_losses, label="Training Loss")
    plt.plot(eval_losses, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    predictions = model.trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(7), yticklabels=range(7))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    model.test()

else:
    preprocessor = Preprocessor()

    if model_name == "ensemble" or model_name == "resnet":
        pretrained = True
    else:
        pretrained = False

    train_loader, valid_loader, test_loader = preprocessor.preprocess(X_train, y_train, X_valid, y_valid, X_test,
                                                                      y_test, pretrained)

    model = get_model(model_name)
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)

    # Store losses and F1-scores for plotting
    train_losses, valid_losses = [], []
    train_f1_scores, valid_f1_scores = {c: [] for c in class_mapping.values()}, {c: [] for c in class_mapping.values()}

    num_epochs = 50

    best_loss = np.inf
    patience = 3
    patience_lr = 2
    counter = 0
    counter_lr = 0
    decay_factor = 0.5

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        y_true_train, y_pred_train = [], []

        for batch_idx, batch in enumerate(train_loader):
            images, labels = [], []
            for image, label in batch:
                if type(label) == int:
                    images.append(image.view(1, 3, 224, 224))
                    labels.append(torch.tensor([label]))
                else:
                    for i in range(len(label)):
                        images.append(image[i].view(1, 3, 224, 224))
                        labels.append(torch.tensor([label[i]]))

            images, labels = torch.cat(images, dim=0).to(device), torch.cat(labels, dim=0).to(device)

            optimizer.zero_grad()
            if model_name == "ensemble":
                outputs = model(images)
            else:
                outputs, features = model(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            sys.stdout.write(
                f"\rEpoch [{epoch + 1}/{num_epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
            sys.stdout.flush()

            # Store predictions and labels for F1-score
            _, predicted = torch.max(outputs, 1)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Compute class-wise F1-score for training
        f1_train = f1_score(y_true_train, y_pred_train, average=None, labels=list(class_mapping.values()))
        for i, c in enumerate(class_mapping.values()):
            train_f1_scores[c].append(f1_train[i] if i < len(f1_train) else 0)

        model.eval()
        total_valid_loss = 0
        y_true_valid, y_pred_valid = [], []

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                if model_name == "ensemble":
                    outputs = model(images)
                else:
                    outputs, features = model(images)

                loss = loss_function(outputs, labels)

                total_valid_loss += loss.item()

                # Store predictions and labels for F1-score
                _, predicted = torch.max(outputs, 1)
                y_true_valid.extend(labels.cpu().numpy())
                y_pred_valid.extend(predicted.cpu().numpy())

        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)

        # Compute class-wise F1-score for validation
        f1_valid = f1_score(y_true_valid, y_pred_valid, average=None, labels=list(class_mapping.values()))
        for i, c in enumerate(class_mapping.values()):
            valid_f1_scores[c].append(f1_valid[i] if i < len(f1_valid) else 0)

        for param_group in optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']}")

        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            counter = 0
            counter_lr = 0
        else:
            counter += 1
            counter_lr += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
            if counter_lr >= patience_lr:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= decay_factor  # Reduce LR
                print(
                    f"Epoch {epoch}: Validation loss did not improve. Reduced LR to {optimizer.param_groups[0]['lr']}")
                counter_lr = 0
        # Print epoch results
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Valid Loss: {avg_valid_loss:.4f}")
        print(f"Train F1-scores: {f1_train}")
        print(f"Valid F1-scores: {f1_valid}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(valid_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()