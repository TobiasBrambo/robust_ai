import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 12 * 12, 128)  # adjust to match the flattened input
        self.fc2 = nn.Linear(128, 10)  # output 10 classes

    def forward(self, x):
        x = F.ReLU(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.ReLU(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


