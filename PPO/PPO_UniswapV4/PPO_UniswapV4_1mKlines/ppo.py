import torch.nn as nn
import torch.nn.functional as F


class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, x):
        x = x.squeeze(1) 
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        return self.actor(x), self.critic(x)
        
        