import torch.nn as nn
import torch.nn.functional as F


class DuelingDDQN(nn.Module):
    def __init__(self, state_size, action_size):
        """Build DDQN as mentioned in paper from Lim"""
        super(DuelingDDQN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, padding=1)

        # Calculate the size of the flattened features after convolution
        self.fc1_input_size = 100 * state_size
        self.fc1 = nn.Linear(self.fc1_input_size, 50)  # Use calculated size
        self.fc2 = nn.Linear(50, 50)

        self.value_stream = nn.Linear(50, 1)
        self.advantage_stream = nn.Linear(50, action_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            x = x.squeeze(2)
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
