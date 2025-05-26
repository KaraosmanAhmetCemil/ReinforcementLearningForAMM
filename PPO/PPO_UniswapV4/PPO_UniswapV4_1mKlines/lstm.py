import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # Add dropout in attention branch to regularize the weights
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        # Compute attention weights across the time dimension
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)