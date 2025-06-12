import torch.nn as nn
import torch.nn.functional as F


class DDQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DDQNNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.q_head = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        q_values = self.q_head(x)
        return q_values
