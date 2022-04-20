import torch
from torch import nn

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Softmax(),
        )

    def forward(self, x):
        outputs = self.network(x)
        return outputs


def bellman_loss():
    pass


def my_loss(output, target):
    loss = torch.mean((output - target) ** 2)
    return loss
