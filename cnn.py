import torch.nn as nn
import torch.nn.functional as functional


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, 1)
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.fully_connected1 = nn.Linear(53 * 53 * 32, 240)
        self.fully_connected2 = nn.Linear(240, 120)
        self.fully_connected3 = nn.Linear(120, 42)

    def forward(self, data):
        data = functional.relu(self.conv1(data))
        data = functional.max_pool2d(data, 2, 2)

        data = functional.relu(self.conv2(data))
        data = functional.max_pool2d(data, 2, 2)

        data = data.view(-1, 53 * 53 * 32)
        data = functional.relu(self.fully_connected1(data))
        data = functional.relu(self.fully_connected2(data))
        data = self.fully_connected3(data)

        return functional.log_softmax(data, dim=1)
