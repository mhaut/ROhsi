
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, input_width, num_classes):
        super(SimpleCNN, self).__init__()
        self.dim = input_width
        self.conv1 = nn.Conv2d(input_channels, 50, kernel_size=3)
        self.conv2 = nn.Conv2d(50, 100, kernel_size=5)
        self.conv3 = nn.Conv2d(100, 200, kernel_size=5)
        self.conv4 = nn.Conv2d(200, 400, kernel_size=2)
        self.fc1 = nn.Linear(400, 300)
        self.fc2 = nn.Linear(300, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(self.conv4(x), 2)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def simplecnn(**kwargs):
    return SimpleCNN(**kwargs)
