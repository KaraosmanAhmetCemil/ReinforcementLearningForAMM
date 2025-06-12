import numpy as np
import torch
from utility import select_action_ddqn, compute_expected_reward
from epsilon_greedy import EpsilonGreedyPolicy
import random
from collections import deque
from amm import AutomatedMarketMaker, DynamicFeeHook, RebalanceHook
import torch.nn.functional as F


def train_ddqn_model(lstm_model, ddqn_network, target_network, ddqn_optimizer, lstm_optimizer, criterion, 
                     train_data, train_alternative_data, validation_inputs, validation_targets,
                     epochs=50, batch_size=32, gamma=0.98, seq_length=50, device=None,
                     epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995, target_update_freq=10, main_scaler=None):
    """
    Trains the LSTM for price prediction and the DDQN agent for action selection.
    Also records training metrics for later visualization.
    """
    # Replay buffer for DDQN transitions
    replay_buffer = deque(maxlen=10000)
    epsilon = epsilon_start

    amm = AutomatedMarketMaker(
        token_x_reserve=1000.0,
        token_y_reserve=1000.0,
        fee_tier=0.01,
        price_range=(0.1, 10.0)
    )
    amm.register_hook(DynamicFeeHook())
    amm.register_hook(RebalanceHook())

    # Lists to record training and validation losses, and DDQN loss
    train_losses = []
    val_losses = []
    ddqn_losses = []

    for epoch in range(1, epochs+1):
        
        # LSTM Training Phase
        
        lstm_model.train()
        lstm_optimizer.zero_grad()
        train_prices = train_data['price'].values
        train_volumes = train_data['volume'].values
        train_inputs_array = np.column_stack((train_prices, train_volumes, train_alternative_data))
        sequences, targets = [], []
        for i in range(len(train_inputs_array) - seq_length):
            sequences.append(train_inputs_array[i:i+seq_length])
            targets.append([train_prices[i+seq_length]])
        train_inputs_tensor = torch.tensor(np.array(sequences), dtype=torch.float32, device=device)
        train_targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
        predictions = lstm_model(train_inputs_tensor)
        lstm_loss = criterion(predictions, train_targets_tensor)
        lstm_loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 0.5)
        lstm_optimizer.step()

        # DDQN Agent Phase

        lstm_model.eval()
        predicted_prices = predictions.squeeze().detach().cpu().numpy()
        ddqn_states = []
        for i in range(len(predicted_prices)):
            volatility = train_alternative_data[i+seq_length, 0]
            fee_tier = min(0.01, 0.003 * (1 + volatility * 5))
            state_vector = [
                predicted_prices[i],
                train_volumes[i+seq_length],
                volatility,
                fee_tier
            ]
            ddqn_states.append(state_vector)

        episode_transitions = []
        for i in range(len(ddqn_states) - 1):
            state = ddqn_states[i]
            next_state = ddqn_states[i+1]
            action = select_action_ddqn(state, ddqn_network, epsilon, device)
            
            scaled_price = train_prices[seq_length + i] 
            scaled_volume = train_volumes[seq_length + i]

            # Inverse-transform to get the real price
            real_price = main_scaler.inverse_transform([[scaled_price, scaled_volume]])[0][0]
            
            # Keep AMM price in sync with real data
            #real_price = train_prices[seq_length + i]
            amm.update_market_price(real_price)
            
            # Record current AMM metrics before executing the trade
            prev_metrics = {
                'fees': amm.total_fees,
                'divergence': amm.lvr,
                'slippage': amm.slip_loss_total,
                'gas': amm.gas_cost,
                'hook_rev': amm.hook_revenue,
                'mev': amm.mev_risk
            }
            
            if action == 1 and amm.token_y_reserve > 1e-8:
                trade_result = amm.swap_y_for_x()
            elif action == 2 and amm.token_x_reserve > 1e-8:
                trade_result = amm.swap_x_for_y()
            else:
                trade_result = (0, 0, 0, 0)
            
            metadata = {
                'fees': amm.total_fees - prev_metrics['fees'],
                'divergence': amm.lvr - prev_metrics['divergence'],
                'slippage': amm.slip_loss_total - prev_metrics['slippage'],
                'gas_cost': amm.gas_cost - prev_metrics['gas'],
                'hook_revenue': amm.hook_revenue - prev_metrics['hook_rev'],
                'mev_risk': amm.mev_risk - prev_metrics['mev']
            }
            
            reward = compute_expected_reward(
                state[0], 
                train_prices[seq_length + i],
                metadata
            )
            
            done = (i == len(ddqn_states) - 2)
            episode_transitions.append((state, action, reward, next_state, done))
            
        for trans in episode_transitions:
            replay_buffer.append(trans)

        current_ddqn_loss = 0.0
        if len(replay_buffer) >= batch_size:
            minibatch = random.sample(replay_buffer, batch_size)
            states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb = zip(*minibatch)
            states_mb = torch.tensor(states_mb, dtype=torch.float32, device=device).unsqueeze(1)
            actions_mb = torch.tensor(actions_mb, dtype=torch.long, device=device)
            rewards_mb = torch.tensor(rewards_mb, dtype=torch.float32, device=device)
            next_states_mb = torch.tensor(next_states_mb, dtype=torch.float32, device=device).unsqueeze(1)
            dones_mb = torch.tensor(dones_mb, dtype=torch.float32, device=device)

            q_values = ddqn_network(states_mb)
            current_q = q_values.gather(1, actions_mb.unsqueeze(1)).squeeze()

            with torch.no_grad():
                next_q_values_online = ddqn_network(next_states_mb)
                best_actions = torch.argmax(next_q_values_online, dim=1, keepdim=True)
                next_q_values_target = target_network(next_states_mb)
                next_q = next_q_values_target.gather(1, best_actions).squeeze()
                target_q = rewards_mb + gamma * next_q * (1 - dones_mb)

            loss = F.mse_loss(current_q, target_q)
            current_ddqn_loss = loss.item()
            ddqn_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddqn_network.parameters(), 0.5)
            ddqn_optimizer.step()

        # Decay epsilon and update target network periodically
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if epoch % target_update_freq == 0:
            target_network.load_state_dict(ddqn_network.state_dict())

        # Validation Step
        
        lstm_model.eval()
        with torch.no_grad():
            validation_predictions = lstm_model(validation_inputs)
            validation_loss = criterion(validation_predictions, validation_targets)

        train_losses.append(lstm_loss.item())
        val_losses.append(validation_loss.item())
        ddqn_losses.append(current_ddqn_loss)

        if epoch % 5 == 0:
            print(f'Epoch {epoch}: LSTM Loss: {lstm_loss.item():.6f} | Val Loss: {validation_loss.item():.6f} | '
                  f'DDQN Loss: {current_ddqn_loss:.6f} | Epsilon: {epsilon:.6f}')

    training_metrics = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'ddqn_loss': ddqn_losses
    }
    return lstm_model, ddqn_network, training_metrics
