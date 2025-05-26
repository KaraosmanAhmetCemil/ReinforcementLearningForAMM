import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv1d(1, 100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(100, 100, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(100*state_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.actor = nn.Linear(50, action_size)
        self.critic = nn.Linear(50, 1)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        return F.softmax(self.actor(x), dim=-1), self.critic(x)
