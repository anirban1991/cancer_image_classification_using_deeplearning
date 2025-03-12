import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=7, device=None):
        """
        Initializes the CNN model and moves it to the specified device (CPU/GPU).
        """
        super(BaselineCNN, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(num_classes).to(self.device)

    def _build_model(self, num_classes):
        """
        Defines the architecture of the CNN.
        """

        class CNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.feature_extractor = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(2, stride=2),

                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2, stride=2),

                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2, stride=2),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, 128)
                )

                self.classifier = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(128, num_classes),
                    nn.LogSoftmax(dim=1)
                )

            def forward(self, x):
                features = self.feature_extractor(x)
                x = self.classifier(features)
                return x, features

        return CNN(num_classes)

    def forward(self, x):
        return self.model(x)

    def get_model(self):
        """
        Returns the model.
        """
        return self.model
