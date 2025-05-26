import numpy as np
import torch


class EpsilonGreedyPolicy:
    """Define Epsiolon Greedy Policy as mentioned in paper from Lim"""
    def __init__(self, start_epsilon=1.0, min_epsilon=0.05, decay=0.995):
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def select_action(self, q_values):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(q_values))  # Explore
        else:
            return torch.argmax(q_values).item()  # Exploit

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
