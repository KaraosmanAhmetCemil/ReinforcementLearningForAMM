import numpy as np
import torch
import torch.nn.functional as F
from utility import compute_expected_reward
import random


def train_hybrid_model(lstm_model, actor_critic, lstm_optimizer, ppo_optimizer, criterion, 
                      replay_buffer, train_data, train_alternative_data, validation_inputs, 
                      validation_targets, epochs=50, batch_size=32, gamma=0.98, clip_epsilon=0.2,
                      ppo_epochs=4, seq_length=50, device=None):
    """
    Trains the hybrid LSTM-PPO model with validation and records training metrics.
    """
    def compute_advantages(rewards, values, dones, next_values):
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - dones[t]
                next_return = values[t+1] if t+1 < len(values) else 0
            else:
                nextnonterminal = 1.0
                next_return = values[t+1]
            delta = rewards[t] + gamma * next_return * nextnonterminal - values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * nextnonterminal * last_advantage
        return advantages

    def make_minibatches(states, actions, log_probs, returns, advantages, batch_size):
        buffer_size = len(states)
        indices = np.arange(buffer_size)
        np.random.shuffle(indices)
        for start in range(0, buffer_size, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            yield (
                states[batch_indices],
                actions[batch_indices],
                log_probs[batch_indices],
                returns[batch_indices],
                advantages[batch_indices]
            )
            
    # Hyperparameters for PPO
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_coef = 0.5

    # Prepare training data for LSTM
    train_prices = train_data['price'].values
    train_volumes = train_data['volume'].values
    train_inputs = np.column_stack((train_prices, train_volumes, train_alternative_data))

    sequences, targets = [], []
    for i in range(len(train_inputs) - seq_length):
        sequences.append(train_inputs[i:i+seq_length])
        targets.append([train_prices[i+seq_length]])

    train_inputs = torch.tensor(np.array(sequences), dtype=torch.float32, device=device)
    train_targets = torch.tensor(targets, dtype=torch.float32, device=device)

    # Lists to record training and validation losses
    train_losses = []
    validation_losses = []

    for epoch in range(1, epochs+1):
        # Train LSTM
        lstm_model.train()
        lstm_optimizer.zero_grad()
        predictions = lstm_model(train_inputs)
        lstm_loss = criterion(predictions, train_targets)
        lstm_loss.backward()
        lstm_optimizer.step()

        # Record train loss
        train_losses.append(lstm_loss.item())

        # Generate states from LSTM predictions
        lstm_predicted_states = np.column_stack((
            predictions[:, 0].detach().cpu().numpy(),
            np.ones_like(predictions[:, 0].detach().cpu().numpy())
        ))
        
        # Collect PPO experiences (for policy update only)
        episode_states = []
        episode_actions = []
        episode_log_probs = []
        episode_rewards = []
        episode_values = []
        episode_dones = []

        for i in range(len(lstm_predicted_states) - 1):
            state = lstm_predicted_states[i]
            next_state = lstm_predicted_states[i+1]
            
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            # Select action using PPO
            with torch.no_grad():
                action_probs, value = actor_critic(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Compute reward (used only for PPO update here)
            liquidity_utilization = np.random.uniform(0.4, 0.9)
            reward = compute_expected_reward(state[0], train_prices[seq_length + i], liquidity_utilization)
            
            # Store transition
            episode_states.append(state)
            episode_actions.append(action.item())
            episode_log_probs.append(log_prob.item())
            episode_rewards.append(reward)
            episode_values.append(value.item())
            episode_dones.append(False)

        # Compute advantages and returns for PPO update
        with torch.no_grad():
            next_value = actor_critic(
                torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            )[1].item()
        episode_values.append(next_value)
        
        advantages = compute_advantages(
            np.array(episode_rewards),
            np.array(episode_values),
            np.array(episode_dones),
            next_value
        )
        returns = advantages + np.array(episode_values[:-1])

        # Convert to tensors
        states_tensor = torch.tensor(np.array(episode_states), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(episode_actions, dtype=torch.long, device=device)
        old_log_probs_tensor = torch.tensor(episode_log_probs, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO updates
        for _ in range(ppo_epochs):
            for batch in make_minibatches(states_tensor, actions_tensor, old_log_probs_tensor,
                                        returns_tensor, advantages_tensor, batch_size):
                batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages = batch
                
                # Get new policy
                new_probs, new_values = actor_critic(batch_states.unsqueeze(1))
                dist = torch.distributions.Categorical(new_probs)
                new_log_probs = dist.log_prob(batch_actions)
                
                # Policy loss
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)
                
                # Entropy loss
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
                
                # Optimize
                ppo_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
                ppo_optimizer.step()

        # Validation
        lstm_model.eval()
        with torch.no_grad():
            validation_predictions = lstm_model(validation_inputs)
            validation_loss = criterion(validation_predictions, validation_targets)
        validation_losses.append(validation_loss.item())

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train LSTM Loss: {lstm_loss.item():.4f}, '
                  f'Validation LSTM Loss: {validation_loss.item():.4f}, '
                  f'PPO Loss: {total_loss.item():.4f}')

    return lstm_model, actor_critic, train_losses, validation_losses