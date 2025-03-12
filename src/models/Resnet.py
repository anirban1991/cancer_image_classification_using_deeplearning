import torch
import torch.nn as nn
import torchvision.models as models


class ResnetModel(nn.Module):
    def __init__(self, num_classes=7, device=None):
        """
        Initializes the CNN model and moves it to the specified device (CPU/GPU).
        """
        super(ResnetModel, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(num_classes).to(self.device)

    def _build_model(self, num_classes):
        """
        Defines the architecture of the CNN.
        """

        class Classifier(nn.Module):
            def __init__(self, num_classes):
                super(Classifier, self).__init__()
                self.resnet = models.resnet50(pretrained=True)
                self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

                for param in self.resnet.parameters():
                    param.requires_grad = False

                for param in list(self.resnet.parameters())[-5:]:
                    param.requires_grad = True

                self.fc1 = nn.Linear(2048, 128)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(128, num_classes)
                self.dropout = nn.Dropout(0.5)

            def forward(self, x):
                x = self.resnet(x)
                x = torch.flatten(x, start_dim=1)

                features = self.fc1(x)

                x = self.relu(features)
                x = self.dropout(x)

                x = self.fc2(x)
                return x, features

        return Classifier(num_classes)

    def forward(self, x):
        return self.model(x)

    def get_model(self):
        """
        Returns the model.
        """
        return self.model
