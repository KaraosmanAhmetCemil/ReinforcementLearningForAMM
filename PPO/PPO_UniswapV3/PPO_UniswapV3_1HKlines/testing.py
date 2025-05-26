from amm import AutomatedMarketMaker
from utility import select_action, detect_price_event, compute_expected_reward
import numpy as np
import torch


def get_predictions(lstm_model, actor_critic, test_data, test_alternative_data, device=None, seq_length=50):
    """Tests the hybrid LSTM-PPO model and returns predictions and actions."""
    lstm_model.eval()
    actor_critic.eval()
    
    with torch.no_grad():
        test_prices = test_data['price'].values
        test_volumes = test_data['volume'].values
        test_inputs = np.column_stack((test_prices, test_volumes, test_alternative_data))

        test_sequences = []
        for i in range(len(test_inputs) - seq_length):
            test_sequences.append(test_inputs[i:i+seq_length])

        test_inputs = torch.tensor(np.array(test_sequences), dtype=torch.float32, device=device)
        test_predictions = lstm_model(test_inputs).squeeze().cpu().numpy()

        # Prepare states for PPO
        test_states = np.column_stack((test_predictions, test_volumes[seq_length:]))
        test_states = torch.tensor(test_states, dtype=torch.float32, device=device).unsqueeze(1)

        # Get actions from PPO
        action_probs, _ = actor_critic(test_states)
        actions = action_probs.argmax(dim=-1).cpu().numpy()

        # Apply event-driven filtering
        final_actions = []
        for i in range(len(actions)):
            if i > 0:
                price_change = abs(test_predictions[i] - test_predictions[i-1])
                volatility = np.std([test_predictions[i], test_predictions[i-1]])
                dynamic_threshold = 0.05 * (1 + volatility*10)
                
                if price_change > dynamic_threshold:
                    final_actions.append(actions[i])
                else:
                    final_actions.append(0)  # Hold
            else:
                final_actions.append(actions[i])

        return test_predictions, np.array(final_actions)

def evaluation(test_data, test_data_scaled, test_alternative_data_scaled, lstm_model, actor_critic, main_scaler, alt_scaler, seq_length, device=None):
    """
    Evaluates the model on test data, performs market updates,
    and computes reward at each step.
    """
    test_predictions, actions = get_predictions(lstm_model, actor_critic, test_data_scaled, test_alternative_data_scaled, device, seq_length)
    test_prices = test_data['price'].values
    
    price_min = main_scaler.data_min_[0]
    price_range = main_scaler.data_range_[0]
    test_predictions_original = test_predictions * price_range + price_min

    # Initialize Automated Market Maker (AMM)
    amm = AutomatedMarketMaker(token_x_reserve=1000, token_y_reserve=1000)

    # Event-driven market update with pseudo arbitrage and liquidity recalibration
    previous_price = test_prices[seq_length]

    # This list will store reward per step (calculated in evaluation)
    step_rewards = []
    liquidity_utilizations = []
    amm_balances = []

    for i in range(len(test_predictions_original)):
        predicted_v = test_predictions_original[i]
        observed_v = test_prices[seq_length + i]

        # Get current AMM reserves (for potential use)
        token_x_reserve, token_y_reserve = amm.get_reserves()

        # Define proxy for trade volume (using test data volume)
        trade_volume = test_data['volume'].iloc[seq_length + i]

        # Calculate dynamic liquidity utilization
        current_liquidity_utilization = amm.calculate_liquidity_utilization(trade_volume)
        liquidity_utilizations.append(current_liquidity_utilization)

        # Compute reward for this step
        reward = compute_expected_reward(predicted_v, observed_v, liquidity_utilization=current_liquidity_utilization)
        step_rewards.append(reward)

        # Execute PPO actions based on predicted actions
        action = actions[i]
        if action == 1:  # Swap Y for X
            amm.swap_y_for_x()
        elif action == 2:  # Swap X for Y
            amm.swap_x_for_y()

        # Record reserves regardless of trade success
        current_reserves = amm.get_reserves()
        amm_balances.append(current_reserves)

        # Update previous price for event detection
        previous_price = observed_v

    liquidity_utilization = np.mean(liquidity_utilizations) * 100

    # Ensure test_predictions_original is properly shaped before computing losses
    if test_predictions_original.ndim > 1:
        test_predictions_original = test_predictions_original[:, 0]

    price_slice = test_prices[seq_length:seq_length + len(test_predictions_original)]
    divergence_loss = np.mean(np.abs((test_predictions_original - price_slice) / price_slice)) * 100  # percentage error
    
    slippage_loss = np.abs(test_predictions_original - test_prices[seq_length:seq_length+len(test_predictions_original)]) / test_prices[seq_length:seq_length+len(test_predictions_original)]
    slippage_loss = slippage_loss.mean()

    print(f'\nLiquidity Utilization: {liquidity_utilization}')
    print(f'Average Divergence Loss: {divergence_loss}')
    print(f'Average Slippage Loss: {slippage_loss}')

    return test_predictions_original, liquidity_utilization, divergence_loss, slippage_loss, step_rewards

