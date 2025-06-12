import numpy as np
import torch
from utility import select_action_ppo, compute_expected_reward
import random
from collections import deque
from amm import AutomatedMarketMaker, DynamicFeeHook, RebalanceHook
import torch.nn.functional as F


def train_ppo_model(lstm_model, ppo_network, ppo_optimizer, lstm_optimizer, criterion, 
                    train_data, train_alternative_data, validation_inputs, validation_targets,
                    epochs=50, gamma=0.98, seq_length=50, device=None,
                    clip_epsilon=0.2, ppo_epochs=4, main_scaler=None):

    training_metrics = {'train_loss': [], 'val_loss': [], 'ppo_loss': []}

    # Initialize AMM environment
    amm = AutomatedMarketMaker(
        token_x_reserve=1000.0,
        token_y_reserve=1000.0,
        fee_tier=0.01,
        price_range=(0.1, 10.0)
    )
    amm.register_hook(DynamicFeeHook())
    amm.register_hook(RebalanceHook())

    # Prepare data
    train_prices = train_data['price'].values
    train_volumes = train_data['volume'].values
    train_inputs_array = np.column_stack((train_prices, train_volumes, train_alternative_data))

    # Phase 1: Pre-train LSTM
    for pre_epoch in range(5):
        lstm_model.train()
        lstm_optimizer.zero_grad()
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

    # Training loop
    for epoch in range(1, epochs+1):
        # LSTM Training
        lstm_model.train()
        lstm_optimizer.zero_grad()
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

        # Generate PPO states using LSTM predictions
        lstm_model.eval()
        with torch.no_grad():
            predicted_prices = predictions.squeeze().cpu().numpy()
        
        ppo_states = []
        for i in range(len(predicted_prices)):
            volatility = train_alternative_data[i+seq_length, 0]
            fee_tier = min(0.01, 0.003 * (1 + volatility * 5))
            state = [
                predicted_prices[i],
                train_volumes[i+seq_length],
                volatility,
                fee_tier
            ]
            ppo_states.append(state)

        # Collect trajectory with TD targets
        episode_transitions = []
        next_states = []
        for i in range(len(ppo_states) - 1):
            state = ppo_states[i]
            next_state = ppo_states[i+1]
            
            # Action selection
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                logits, value = ppo_network(state_tensor)
                probs = F.softmax(logits, dim=-1)
                m = torch.distributions.Categorical(probs)
                action = m.sample()
                log_prob = m.log_prob(action).item()
                action = action.item()
                value = value.item()

            # Environment interaction
            scaled_price = train_prices[seq_length + i]
            scaled_volume = train_volumes[seq_length + i]
            real_price = main_scaler.inverse_transform([[scaled_price, scaled_volume]])[0][0]
            amm.update_market_price(real_price)
            
            prev_metrics = {
                'fees': amm.total_fees,
                'divergence': amm.lvr,
                'slippage': amm.slip_loss_total,
                'gas': amm.gas_cost,
                'hook_rev': amm.hook_revenue,
                'mev': amm.mev_risk
            }
            
            if action == 1 and amm.token_y_reserve > 1e-8:
                amm.swap_y_for_x()
            elif action == 2 and amm.token_x_reserve > 1e-8:
                amm.swap_x_for_y()
            
            metadata = {
                'fees': amm.total_fees - prev_metrics['fees'],
                'divergence': amm.lvr - prev_metrics['divergence'],
                'slippage': amm.slip_loss_total - prev_metrics['slippage'],
                'gas_cost': amm.gas_cost - prev_metrics['gas'],
                'hook_revenue': amm.hook_revenue - prev_metrics['hook_rev'],
                'mev_risk': amm.mev_risk - prev_metrics['mev']
            }
            
            reward = compute_expected_reward(state[0], train_prices[seq_length + i], metadata)
            done = (i == len(ppo_states) - 2)
            
            episode_transitions.append((state, action, reward, next_state, done, log_prob, value))
            next_states.append(next_state)

        # Compute TD targets and advantages
        states, actions, rewards, next_states, dones, old_log_probs, values = zip(*episode_transitions)
        
        # Get next state values
        with torch.no_grad():
            next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=device).unsqueeze(1)
            _, next_values = ppo_network(next_states_tensor)
            next_values = next_values.squeeze().cpu().numpy()
        
        # Convert to numpy arrays
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)
        
        # Calculate TD targets and advantages
        td_targets = rewards + gamma * next_values * (1 - dones)
        advantages = td_targets - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Normalize states
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        state_mean = states_tensor.mean(dim=0)
        state_std = states_tensor.std(dim=0)
        states_tensor = (states_tensor - state_mean) / (state_std + 1e-8)
        states_tensor = states_tensor.unsqueeze(1)  # Add sequence dimension

        # Convert to tensors
        returns = torch.tensor(td_targets, dtype=torch.float32, device=device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)

        # Minibatch PPO updates
        batch_size = len(states_tensor)
        minibatch_size = 64
        indices = np.arange(batch_size)
        ppo_loss_total = 0.0

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                idx = indices[start:end]

                mb_states = states_tensor[idx]
                mb_actions = actions_tensor[idx]
                mb_old_log_probs = old_log_probs_tensor[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]

                # Forward pass
                logits, values_pred = ppo_network(mb_states)
                probs = F.softmax(logits, dim=-1)
                m = torch.distributions.Categorical(probs)
                new_log_probs = m.log_prob(mb_actions)
                entropy = m.entropy().mean()

                # Loss calculations
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred.squeeze(), mb_returns)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                # Optimization step
                ppo_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ppo_network.parameters(), 0.5)
                ppo_optimizer.step()
                ppo_loss_total += loss.item()

        # Calculate average PPO loss
        avg_ppo_loss = ppo_loss_total / (ppo_epochs * (batch_size // minibatch_size))
        training_metrics['ppo_loss'].append(avg_ppo_loss)

        # Validation
        lstm_model.eval()
        with torch.no_grad():
            validation_predictions = lstm_model(validation_inputs)
            validation_loss = criterion(validation_predictions, validation_targets)
        
        # Record metrics
        training_metrics['train_loss'].append(lstm_loss.item())
        training_metrics['val_loss'].append(validation_loss.item())
        
        if epoch % 5 == 0:
            print(f'Epoch {epoch}:')
            print(f'  LSTM Loss: {lstm_loss.item():.6f} | Val Loss: {validation_loss.item():.6f}')
            print(f'  PPO Loss: {avg_ppo_loss:.6f} |  Total Reward: {rewards.sum():.2f}')

    return lstm_model, ppo_network, training_metrics
