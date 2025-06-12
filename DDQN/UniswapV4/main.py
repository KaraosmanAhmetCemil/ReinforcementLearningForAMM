import warnings 
warnings.filterwarnings('ignore')
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from get_real_data import generate_trading_data, generate_alternative_data
from training import train_ddqn_model
from testing import evaluation
from ddqn import DDQNNetwork
from lstm import AttentionLSTMModel
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
    
    train_data = data[:5000]
    test_data = data[5000:6000]
    validation_data = data[6000:7000]
    

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

    validation_prices = validation_data_scaled['price'].values
    validation_volumes = validation_data_scaled['volume'].values
    validation_targets = validation_prices[50:]
    validation_sequences = []
    for i in range(len(validation_inputs) - 50):
        validation_sequences.append(validation_inputs[i:i+50])
    validation_inputs_tensor = torch.tensor(np.array(validation_sequences), dtype=torch.float32, device=device)
    validation_targets_tensor = torch.tensor(validation_targets, dtype=torch.float32, device=device).unsqueeze(1)
    
    # Initialize LSTM model
    lstm_model = AttentionLSTMModel().to(device)
    # Instantiate the DDQN network
    ddqn_network = DDQNNetwork(state_size=4, action_size=3).to(device)
    # Create a target network and copy parameters from the online DDQN
    target_network = DDQNNetwork(state_size=4, action_size=3).to(device)
    target_network.load_state_dict(ddqn_network.state_dict())
    ddqn_optimizer = optim.Adam(ddqn_network.parameters(), lr=0.0004)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.0004)

    seq_length = 50

    # Train the models using the DDQN training function and capture training metrics
    lstm_model, ddqn_network, training_metrics = train_ddqn_model(
        lstm_model=lstm_model,
        ddqn_network=ddqn_network,
        target_network=target_network,
        ddqn_optimizer=ddqn_optimizer,
        lstm_optimizer=lstm_optimizer,
        criterion=nn.SmoothL1Loss(),
        train_data=train_data_scaled,
        train_alternative_data=train_alternative_data_scaled,
        validation_inputs=validation_inputs_tensor,
        validation_targets=validation_targets_tensor,
        epochs=40,
        batch_size=32,
        seq_length=seq_length,
        device=device,
        main_scaler=main_scaler
    )

    viz_data = evaluation(
        test_data,
        test_data_scaled,
        test_alternative_data_scaled,
        lstm_model,
        ddqn_network,
        main_scaler,
        alt_scaler,
        seq_length=seq_length,
        device=device
    )
    
    visualization(viz_data, training_metrics)
    
if __name__ == '__main__':
    main()