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
from ppo import ActorCritic
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
    
    # Initialize components
    lstm_model = LSTMModel().to(device)
    actor_critic = ActorCritic(state_size=2, action_size=3).to(device)
    ppo_optimizer = optim.Adam(actor_critic.parameters(), lr=0.001)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    seq_length = 50

    # Train the model
    lstm_model, actor_critic, train_losses, validation_losses = train_hybrid_model(
        lstm_model=lstm_model,
        actor_critic=actor_critic,
        lstm_optimizer=lstm_optimizer,
        ppo_optimizer=ppo_optimizer,
        criterion=nn.SmoothL1Loss(),
        replay_buffer=[],
        train_data=train_data_scaled,
        train_alternative_data=train_alternative_data_scaled,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        epochs=50,
        batch_size=32,
        seq_length=seq_length,
        device=device
    )

    # Test the model
    test_predictions_original, liquidity_utilization, divergence_loss, slippage_loss, step_rewards = evaluation(
        test_data,
        test_data_scaled,
        test_alternative_data_scaled,
        lstm_model,
        actor_critic,
        main_scaler,
        alt_scaler,
        seq_length=seq_length,
        device=device
    )
    
    # draw the visualzations
    visualization(test_data, test_predictions_original, test_data['price'].values, liquidity_utilization, divergence_loss, slippage_loss, seq_length, train_losses, validation_losses, step_rewards)

    
if __name__ == '__main__':
    main()