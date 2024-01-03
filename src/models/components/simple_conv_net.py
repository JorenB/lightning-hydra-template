import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvNet(nn.Module):
    def __init__(self, img_dim: int, num_classes: int):
        super().__init__()

        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        h = img_dim
        w = img_dim

        self.fc = nn.Linear(128 * h * w // 64, num_classes)

    def forward(self, x):
        out = self.layer1(x)

        out = self.layer2(out)

        out = self.layer3(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc(out)

        return out
