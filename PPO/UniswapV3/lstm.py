import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=100, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.batch_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)  # Now output_size is 1

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.batch_norm(lstm_out[:, -1, :])
        lstm_out = self.dropout(lstm_out)
        return self.fc(lstm_out)

       