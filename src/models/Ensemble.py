import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, mlp, resnet, feature_size, num_classes):
        super(EnsembleModel, self).__init__()
        self.mlp = mlp
        self.resnet = resnet

        self.final_linear_layer = nn.Sequential(
            nn.Linear(feature_size * 2, 256),  # Concatenate features from three models
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        logits_mlp, features_mlp = self.mlp(x)
        logits_resnet, features_resnet = self.resnet(x)

        # Concatenate feature embeddings from all models
        combined_features = torch.cat((features_mlp, features_resnet), dim=1)

        output = self.final_linear_layer(combined_features)
        return output
