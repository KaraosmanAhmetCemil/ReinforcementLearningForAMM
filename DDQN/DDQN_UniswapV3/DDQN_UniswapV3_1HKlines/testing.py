from amm import AutomatedMarketMaker
from utility import select_action, detect_price_event, compute_expected_reward
import numpy as np
import torch


def get_predictions(test_data_scaled, test_alternative_data_scaled, lstm_model, q_network, seq_length, device):
    """Testing LSTM Model & Using Predictions in AMM with Q-Network."""
    with torch.no_grad():
        test_prices = test_data_scaled['price'].values
        test_volumes = test_data_scaled['volume'].values
        test_inputs = np.column_stack((test_prices, test_volumes, test_alternative_data_scaled))

        test_sequences = []
        for i in range(len(test_inputs) - seq_length):
            test_sequences.append(test_inputs[i:i+seq_length])

        test_inputs = torch.tensor(np.array(test_sequences), dtype=torch.float32, device=device)
        test_predictions = lstm_model(test_inputs).squeeze().cpu().numpy()

        # Convert LSTM predictions to a format suitable for DuelingDDQN
        test_states = np.column_stack((test_predictions, test_volumes[seq_length:]))
        test_states = torch.tensor(test_states, dtype=torch.float32, device=device)
        test_states = test_states.unsqueeze(1)  # Reshape to match Conv1D input format

        q_values = q_network(test_states)
        actions = [select_action(q_values[i], threshold=0.1, beta=0.1) for i in range(len(q_values))]
        
    return test_predictions, actions

# def evaluation(test_data, test_data_scaled, test_alternative_data_scaled, lstm_model, q_network, main_scaler, alt_scaler, seq_length, device):
    # test_predictions, actions = get_predictions(test_data_scaled, test_alternative_data_scaled, lstm_model, q_network, seq_length, device)
    # test_prices = test_data['price'].values

    # dummy_volume = np.zeros_like(test_predictions)
    # scaled_data = np.column_stack((test_predictions, dummy_volume))
    # inverse_scaled = main_scaler.inverse_transform(scaled_data)
    # test_predictions_original = inverse_scaled[:, 0]
    
    # # Initialize Automated Market Maker (AMM)
    # amm = AutomatedMarketMaker(token_x_reserve=1000, token_y_reserve=1000)
    # previous_price = test_prices[seq_length]

    # # Event-driven market update with pseudo arbitrage and liquidity recalibration
    # for i in range(1, len(test_predictions_original)):
        # current_price = test_prices[seq_length + i]
        # predicted_v = test_predictions_original[i]  # Use LSTM predictions

        # if detect_price_event(previous_price, current_price, threshold=0.5):
            # amm.pseudo_arbitrage(current_price)
            # amm.update_market_price(current_price, predicted_v)

        # # Execute Q-learning actions
        # action = actions[i]
        # if action == 1:
            # amm.swap_y_for_x()
        # elif action == 2:
            # amm.swap_x_for_y() 

        # previous_price = current_price

    # amm_balances = []
    # rewards_over_steps = []      # To track rewards over each step
    # liquidity_utilizations = []  # To track per-step liquidity utilization

    # for i in range(len(test_predictions_original)):
        # predicted_v = test_predictions_original[i]
        # observed_v = test_prices[seq_length + i]

        # token_x_reserve, token_y_reserve = amm.get_reserves()
        # trade_volume = test_data['volume'].iloc[seq_length + i]
        # # Use the AMM's built-in liquidity utilization function which caps at 1.0.
        # current_liquidity_utilization = amm.calculate_liquidity_utilization(trade_volume)
        # liquidity_utilizations.append(current_liquidity_utilization)
        
        # reward = compute_expected_reward(predicted_v, observed_v, liquidity_utilization=current_liquidity_utilization)
        # rewards_over_steps.append(reward)

        # action = actions[i]
        # if action == 0:
            # trade_result = 0
        # elif action == 1:
            # trade_result = amm.swap_y_for_x()
        # elif action == 2:
            # trade_result = amm.swap_x_for_y()
        # else:
            # trade_result = 0

        # if trade_result > 0:
            # amm_balances.append(amm.get_reserves())
        # else:
            # amm_balances.append(amm_balances[-1] if amm_balances else (amm.token_x_reserve, amm.token_y_reserve))

    # # Convert AMM balance history to arrays for visualization
    # amm_x_balances, amm_y_balances = zip(*amm_balances)
    # amm_x_balances, amm_y_balances = map(list, zip(*amm_balances))

    # # Calculate average liquidity utilization as percentage:
    # liquidity_utilization = np.mean(liquidity_utilizations) * 100
    
    # if test_predictions_original.ndim > 1:
        # test_predictions_original = test_predictions_original[:, 0]

    # price_slice = test_prices[seq_length:seq_length + len(test_predictions_original)]
    # divergence_loss = np.mean(np.abs((test_predictions_original - price_slice) / price_slice)) * 100  # percentage error

    # slippage_loss = np.abs(test_predictions_original - test_prices[seq_length:len(test_predictions_original) + seq_length]) / test_prices[seq_length:len(test_predictions_original) + seq_length]
    # slippage_loss = slippage_loss.mean()

    # print(f'\nLiquidity Utilization: {liquidity_utilization}')
    # print(f'Average Divergence Loss: {divergence_loss}')
    # print(f'Average Slippage Loss: {slippage_loss}')

    # return test_predictions_original, liquidity_utilization, divergence_loss, slippage_loss, rewards_over_steps
    
def evaluation(test_data, test_data_scaled, test_alternative_data_scaled, lstm_model, q_network, main_scaler, alt_scaler, seq_length, device):
    test_predictions, actions = get_predictions(test_data_scaled, test_alternative_data_scaled, lstm_model, q_network, seq_length, device)
    test_prices = test_data['price'].values

    dummy_volume = np.zeros_like(test_predictions)
    scaled_data = np.column_stack((test_predictions, dummy_volume))
    inverse_scaled = main_scaler.inverse_transform(scaled_data)
    test_predictions_original = inverse_scaled[:, 0]
    
    # Initialize Automated Market Maker (AMM)
    amm = AutomatedMarketMaker(token_x_reserve=1000, token_y_reserve=1000)
    previous_price = test_prices[seq_length]

    # Event-driven market update with pseudo arbitrage and liquidity recalibration
    for i in range(1, len(test_predictions_original)):
        current_price = test_prices[seq_length + i]
        predicted_v = test_predictions_original[i]  # Use LSTM predictions

        if detect_price_event(previous_price, current_price, threshold=0.5):
            amm.pseudo_arbitrage(current_price)
            amm.update_market_price(current_price, predicted_v)

        # Execute Q-learning actions as before
        action = actions[i]
        if action == 1:
            amm.swap_y_for_x()
        elif action == 2:
            amm.swap_x_for_y() 

        previous_price = current_price

    amm_balances = []
    rewards_over_steps = []      # Track rewards over each step
    liquidity_utilizations = []  # Track per-step liquidity utilization

    for i in range(len(test_predictions_original)):
        predicted_v = test_predictions_original[i]
        observed_v = test_prices[seq_length + i]

        # Instead of using external volume, execute the trade action to get the actual trade delta.
        action = actions[i]
        trade_volume_executed = 0
        if action == 1:
            result = amm.swap_y_for_x()
            if isinstance(result, tuple):
                trade_volume_executed = abs(result[0])
        elif action == 2:
            result = amm.swap_x_for_y()
            if isinstance(result, tuple):
                trade_volume_executed = abs(result[0])
        # For action 0 (or any undefined action) no trade is executed and trade_volume_executed remains 0.

        # Use the executed trade volume to compute liquidity utilization
        current_liquidity_utilization = amm.calculate_liquidity_utilization(trade_volume_executed)
        liquidity_utilizations.append(current_liquidity_utilization)
        
        reward = compute_expected_reward(predicted_v, observed_v, liquidity_utilization=current_liquidity_utilization)
        rewards_over_steps.append(reward)

        # Record current AMM reserves after the trade execution
        amm_balances.append(amm.get_reserves())

    # Convert AMM balance history to arrays for visualization (if desired)
    amm_x_balances, amm_y_balances = zip(*amm_balances)
    amm_x_balances, amm_y_balances = map(list, zip(*amm_balances))

    # Calculate average liquidity utilization as percentage:
    liquidity_utilization = np.mean(liquidity_utilizations) * 100
    
    if test_predictions_original.ndim > 1:
        test_predictions_original = test_predictions_original[:, 0]

    price_slice = test_prices[seq_length:seq_length + len(test_predictions_original)]
    divergence_loss = np.mean(np.abs((test_predictions_original - price_slice) / price_slice)) * 100  # percentage error

    slippage_loss = np.abs(test_predictions_original - test_prices[seq_length:len(test_predictions_original) + seq_length])
    slippage_loss = (slippage_loss / test_prices[seq_length:len(test_predictions_original) + seq_length]).mean()

    print(f'\nLiquidity Utilization: {liquidity_utilization}')
    print(f'Average Divergence Loss: {divergence_loss}')
    print(f'Average Slippage Loss: {slippage_loss}')

    return test_predictions_original, liquidity_utilization, divergence_loss, slippage_loss, rewards_over_steps


