import torch
import torch.nn as nn

        
class LSTMModel(nn.Module):
    """Build LSTM as mentioned in paper from Lim"""
    def __init__(self, input_size=7, hidden_size=100, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.batch_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)  # Increased dropout from 0.2 to 0.3
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.batch_norm(lstm_out[:, -1, :])
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        # Constrain output to [0, 1] to match the scaled price range
        output = torch.sigmoid(output)
        return output

