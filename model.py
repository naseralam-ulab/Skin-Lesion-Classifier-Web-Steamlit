import torch.nn as nn
import torch.nn.functional as F

class CustomModel(nn.Module):
    def __init__(self, out_classes):
        super(CustomModel, self).__init__()
        self.out_classes = out_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3,3),
            # 224 -> 74
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            # 74 -> 24
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.BatchNorm2d(256),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            # 24 -> 8
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            # 8 -> 1

        )

        self.classifier = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Linear(256 * 2 * 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, out_classes),
            nn.ReLU(),
            nn.BatchNorm1d(out_classes)
        )
    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier(x)
        return x