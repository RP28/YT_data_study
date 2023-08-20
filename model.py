import torch
import torch.nn as nn


class BinaryClassifierWithResidual(nn.Module):
    def __init__(self):
        super(BinaryClassifierWithResidual, self).__init__()
        self.fwd = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=264, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        self.fc1 = nn.Sequential(nn.Linear(264 * 4 * 6, 10), nn.Linear(10, 1))

    def forward(self, x):
        x = self.fwd(x)
        x = x.view(-1, 264 * 4 * 6)
        x = torch.sigmoid(self.fc1(x))
        return x
