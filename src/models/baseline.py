import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, input_size=3 * 250 * 250, num_classes=7, device=None):
        """
        Initializes the model and moves it to the specified device (CPU/GPU).
        """
        super(BaselineModel, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(input_size, num_classes).to(self.device)

    def _build_model(self, input_size, num_classes):
        class MLP(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.feature_extractor = nn.Sequential(
                                    nn.Flatten(),  # Flatten input from (3,250,250) to 3*250*250
                                    nn.Linear(input_size, 256),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.1),
                                    nn.Linear(256, 128)
                                )

                self.classifier = nn.Sequential(
                                    nn.ReLU(),
                                    nn.Dropout(p=0.1),
                                    nn.Linear(128, num_classes),
                                    nn.LogSoftmax(dim=1)
                                )

            def forward(self, x):
                features = self.feature_extractor(x)
                x = self.classifier(features)
                return x, features

        return MLP(num_classes)

    def forward(self, x):
        return self.model(x)

    def get_model(self):
        """
        Returns the model.
        """
        return self.model
