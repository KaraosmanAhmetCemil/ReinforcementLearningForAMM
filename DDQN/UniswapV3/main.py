import warnings 
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from get_real_data import generate_trading_data, generate_alternative_data
from lstm import LSTMModel
from ddqn import DuelingDDQN
from training import train_hybrid_model
from testing import evaluation
from visualization import visualization


def main():
    SEED = 42 
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = pd.read_csv("ETH_1H.csv")
    
    data = generate_trading_data(df)

    train_data = data[:6000]
    test_data = data[6000:7000]
    validation_data = data[7000:]

    main_scaler = MinMaxScaler()
    
    train_data_scaled = train_data.copy()
    
    train_data_scaled[['price', 'volume']] = main_scaler.fit_transform(train_data[['price', 'volume']])
        
    # Ensure that both test and validation data are clipped to [0, 1]
    test_data_scaled = test_data.copy()
    test_data_scaled[['price', 'volume']] = np.clip(main_scaler.transform(test_data[['price', 'volume']]), 0, 1)
        
    validation_data_scaled = validation_data.copy()
    validation_data_scaled[['price', 'volume']] = np.clip(main_scaler.transform(validation_data[['price', 'volume']]), 0, 1)

    train_alternative_data = generate_alternative_data(train_data)
    test_alternative_data = generate_alternative_data(test_data)
    validation_alternative_data = generate_alternative_data(validation_data)
    
    alt_scaler = MinMaxScaler()
    train_alternative_data_scaled = alt_scaler.fit_transform(train_alternative_data)
    validation_alternative_data_scaled = alt_scaler.transform(validation_alternative_data)
    test_alternative_data_scaled = alt_scaler.transform(test_alternative_data)

    train_inputs = np.column_stack((train_data[['price', 'volume']].values, train_alternative_data_scaled))
    validation_inputs = np.column_stack((validation_data[['price', 'volume']].values, validation_alternative_data_scaled))
    test_inputs = np.column_stack((test_data[['price', 'volume']].values, test_alternative_data_scaled))

    # Prepare validation data for LSTM
    validation_prices = validation_data_scaled['price'].values
    validation_volumes = validation_data_scaled['volume'].values
    validation_targets = validation_prices[50:]  # Targets are the next prices after the sequence

    validation_sequences = []
    for i in range(len(validation_inputs) - 50):
        validation_sequences.append(validation_inputs[i:i+50])

    validation_inputs = torch.tensor(np.array(validation_sequences), dtype=torch.float32, device=device)
    validation_targets = torch.tensor(validation_targets, dtype=torch.float32, device=device).unsqueeze(1)
    
    # Train the hybrid LSTM-Q-learning model
    replay_buffer = deque(maxlen=20000)
    lstm_model = LSTMModel().to(device)
    q_network = DuelingDDQN(state_size=2, action_size=3).to(device)
    target_q_network = DuelingDDQN(state_size=2, action_size=3).to(device)
    target_q_network.load_state_dict(q_network.state_dict())

    criterion = nn.SmoothL1Loss()
    learning_rate = 0.0001
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    q_optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    epochs = 200
    seq_length = 50

    lstm_model, q_network, train_losses, val_losses = train_hybrid_model(
        lstm_model, q_network, target_q_network, lstm_optimizer, q_optimizer, criterion, replay_buffer, 
        train_data_scaled, train_alternative_data_scaled, validation_inputs, validation_targets, 
        seq_length, device, epochs=epochs
    )
    
    test_predictions_original, liquidity_utilization, divergence_loss, slippage_loss, rewards_over_steps = evaluation(
        test_data, test_data_scaled, test_alternative_data_scaled, lstm_model, q_network, 
        main_scaler, alt_scaler, seq_length, device
    )
    
    visualization(test_data, test_predictions_original, test_data['price'].values, liquidity_utilization, 
                  divergence_loss, slippage_loss, seq_length, train_losses, val_losses, rewards_over_steps)

if __name__ == '__main__':
    main()