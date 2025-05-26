import numpy as np
import torch
from utility import q_learning_loss, compute_expected_reward
from epsilon_greedy import EpsilonGreedyPolicy
import random


def train_q_network(q_network, target_q_network, optimizer, replay_buffer, device, batch_size=32, gamma=0.98):
    """
    Trains the Q-network using the DD-DQN algorithm.
    """
    if len(replay_buffer) < batch_size:
        return  # Wait until we have enough experience

    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.bool, device=device)

    q_values = q_network(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # Get Q-value for chosen action

    with torch.no_grad():
        next_q_values = target_q_network(next_states)
        next_q_values = next_q_values.max(1)[0]
        target_q_values = rewards + gamma * (next_q_values * (~dones) + np.random.normal(0, 0.05))  # Adding Gaussian noise

    loss = q_learning_loss(q_values, target_q_values)  # Use the correct loss function

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_hybrid_model(lstm_model, q_network, target_q_network, lstm_optimizer, q_optimizer, criterion, replay_buffer, 
                       train_data, train_alternative_data, validation_inputs, validation_targets, seq_length, 
                       device, epochs=50, batch_size=32, gamma=0.98):
    """
    Trains the hybrid LSTM-Q-learning model with validation.
    """
    train_prices = train_data['price'].values
    train_volumes = train_data['volume'].values

    if train_alternative_data.shape[0] != len(train_prices):
        raise ValueError(f"Alternative data size mismatch! Expected {len(train_prices)}, got {train_alternative_data.shape[0]}")

    train_inputs = np.column_stack((train_prices, train_volumes, train_alternative_data))

    sequences, targets = [], []
    for i in range(len(train_inputs) - seq_length):
        sequences.append(train_inputs[i:i+seq_length])
        targets.append([train_prices[i+seq_length]])  # Target is next price

    train_inputs = torch.tensor(np.array(sequences), dtype=torch.float32, device=device)
    train_targets = torch.tensor(targets, dtype=torch.float32, device=device)

    policy = EpsilonGreedyPolicy()

    # Lists to track losses for plotting
    train_loss_history = []
    val_loss_history = []
    
    patience = 20  # number of epochs to wait before stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(1, epochs+1):
        # Train LSTM
        lstm_model.train()
        lstm_optimizer.zero_grad()
        predictions = lstm_model(train_inputs)
        predictions = predictions[:, 0].unsqueeze(1)  # Ensure correct shape
        lstm_loss = criterion(predictions, train_targets)
        lstm_loss.backward()
        lstm_optimizer.step()

        # Generate Q-network training data from LSTM predictions
        lstm_predicted_states = np.column_stack((predictions[:, 0].detach().cpu().numpy(), 
                                                  np.ones_like(predictions[:, 0].detach().cpu().numpy())))

        for i in range(len(lstm_predicted_states) - 1):
            state = lstm_predicted_states[i]
            next_state = lstm_predicted_states[i+1]

            liquidity_utilization = np.random.uniform(0.4, 0.9) 
            reward = compute_expected_reward(state[0], train_prices[seq_length + i], liquidity_utilization)

            # Select action using epsilon-greedy strategy
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) 
            q_values = q_network(state_tensor)
            action = policy.select_action(q_values)

            replay_buffer.append((state, action, reward, next_state, False))

        # Train Q-network using LSTM-generated states
        train_q_network(q_network, target_q_network, q_optimizer, replay_buffer, device, batch_size=batch_size, gamma=gamma)
        policy.decay_epsilon()

        # Evaluate on validation data
        lstm_model.eval()
        with torch.no_grad():
            validation_predictions = lstm_model(validation_inputs)
            validation_loss = criterion(validation_predictions, validation_targets)

        train_loss_history.append(lstm_loss.item())
        val_loss_history.append(validation_loss.item())

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train LSTM Loss: {lstm_loss.item()}, Validation LSTM Loss: {validation_loss.item()}')
            
        # Early stopping logic:
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    return lstm_model, q_network, train_loss_history, val_loss_history