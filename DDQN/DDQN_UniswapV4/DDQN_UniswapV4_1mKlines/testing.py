import numpy as np
import torch
from amm import AutomatedMarketMaker, DynamicFeeHook, RebalanceHook
from utility import select_action_ddqn, detect_price_event, compute_expected_reward
import pandas as pd


def get_predictions(lstm_model, ddqn_network, test_data, test_alternative_data, device=None, seq_length=50, amm=None):
    """
    Generate predictions from the LSTM and select actions using the DDQN network.
    """
    lstm_model.eval()
    ddqn_network.eval()
    
    with torch.no_grad():
        test_prices = test_data['price'].values
        test_volumes = test_data['volume'].values
        test_inputs_array = np.column_stack((test_prices, test_volumes, test_alternative_data))
        test_sequences = []
        for i in range(len(test_inputs_array) - seq_length):
            test_sequences.append(test_inputs_array[i:i+seq_length])
        test_inputs_tensor = torch.tensor(np.array(test_sequences), dtype=torch.float32, device=device)
        test_predictions = lstm_model(test_inputs_tensor).squeeze().cpu().numpy()

        test_states = []
        actions = []
        for i in range(len(test_predictions)):
            actual_index = i + seq_length
            if actual_index >= len(test_alternative_data):
                actual_index = len(test_alternative_data) - 1
            volatility = test_alternative_data[actual_index, 0]
            fee_tier = min(0.03, 0.003 * (1 + volatility * 3))
            state = [
                test_predictions[i],
                test_volumes[actual_index],
                volatility,
                fee_tier
            ]
            test_states.append(state)
            action = select_action_ddqn(state, ddqn_network, epsilon=0.05, device=device)
            actions.append(action)
        test_states_tensor = torch.tensor(test_states, dtype=torch.float32, device=device).unsqueeze(1)
                
        return test_predictions, np.array(actions), test_states_tensor


def evaluation(test_data, test_data_scaled, test_alternative_data_scaled, 
               lstm_model, ddqn_network, main_scaler, alt_scaler, seq_length, device=None):
                   
    ALPHA = 2.0  # fee weight
    BETA = 0.02  # gas penalty
    GAMMA = 1.0  # hook incentive
    DELTA = 0.1  # MEV risk penalty    
    
    amm = AutomatedMarketMaker(
        token_x_reserve=1000.0,
        token_y_reserve=1000.0,
        fee_tier=0.01,
        price_range=(0.1, 10.0)
    )
    amm.register_hook(DynamicFeeHook())
    amm.register_hook(RebalanceHook())

    flash_window = 3
    swap_counter = 0
    in_flash_batch = False

    test_predictions, actions, test_states = get_predictions(
        lstm_model, ddqn_network, 
        test_data_scaled, test_alternative_data_scaled,
        device, seq_length, amm
    )

    dummy_volume = np.zeros_like(test_predictions)
    scaled_data = np.column_stack((test_predictions, dummy_volume))
    inverse_scaled = main_scaler.inverse_transform(scaled_data)
    test_predictions = inverse_scaled[:, 0]

    test_prices = np.clip(test_data['price'].values, 1e-8, 1e8)
    timestamps = test_data['timestamp'].values[seq_length:]
    price_impacts = []
    step_metrics = []
    total_reward = 0.0
    previous_price = amm.get_price()
    initial_value = (amm.token_x_reserve * previous_price) + amm.token_y_reserve

    for i in range(len(test_predictions)):
        prev_state = {
            'fees': amm.total_fees,
            'divergence': amm.lvr,
            'slippage': amm.slip_loss_total,
            'hook_rev': amm.hook_revenue,
            'gas': amm.gas_cost,
            'mev': amm.mev_risk,
            'reserves': amm.get_reserves()
        }
        
        current_price = test_prices[seq_length + i]
        predicted_price = np.clip(test_predictions[i], 1e-8, 1e8)
        action = actions[i]
        
        if action in [1, 2] and not in_flash_batch:
            amm.begin_flash_transaction()
            in_flash_batch = True
        
        trade_result = (0, 0, 0, 0)
        input_delta = 0
        
        if action == 1 and amm.token_y_reserve > 1e-8:
            trade_result = amm.swap_y_for_x()
            input_delta = amm.delta_y
            swap_counter += 1
        elif action == 2 and amm.token_x_reserve > 1e-8:
            trade_result = amm.swap_x_for_y()
            input_delta = amm.delta_x
            swap_counter += 1
        
        liquidity_utilization = amm.calculate_liquidity_utilization(input_delta)
        
        if in_flash_batch and (swap_counter >= flash_window or action == 0):
            amm.end_flash_transaction()
            in_flash_batch = False
            swap_counter = 0
        
        if detect_price_event(previous_price, current_price):
            amm.pseudo_arbitrage(np.clip(current_price, 0.1, 10.0))
            amm.update_market_price(np.clip(current_price, 0.1, 10.0), np.clip(predicted_price, 0.1, 10.0))
        
        current_value = (amm.token_x_reserve * current_price) + amm.token_y_reserve
        price_impact = abs(current_price - previous_price) / previous_price if previous_price > 1e-8 else 0
        price_impacts.append(price_impact)
        
        step_metrics.append({
            'step': i,
            'action': action,
            'price': current_price,
            'predicted': predicted_price,
            'fees': amm.total_fees - prev_state['fees'],
            'divergence': amm.lvr - prev_state['divergence'],
            'slippage': amm.slip_loss_total - prev_state['slippage'],
            'hook_revenue': amm.hook_revenue - prev_state['hook_rev'],
            'gas_cost': amm.gas_cost - prev_state['gas'],
            'mev_risk': amm.mev_risk - prev_state['mev'],
            'dx': amm.token_x_reserve - prev_state['reserves'][0],
            'dy': amm.token_y_reserve - prev_state['reserves'][1],
            'value_change': current_value - (prev_state['reserves'][0] * previous_price + prev_state['reserves'][1]),
            'price_impact': price_impact,
            'liquidity_utilization': liquidity_utilization,
        })
        
        step_reward = (
            ALPHA * (step_metrics[-1]['fees'] - step_metrics[-1]['divergence']) 
            - BETA * step_metrics[-1]['gas_cost'] 
            + GAMMA * step_metrics[-1]['hook_revenue'] 
            - DELTA * step_metrics[-1]['mev_risk']
        )
        
        step_metrics[-1]['reward'] = step_reward
        total_reward += step_reward
        previous_price = current_price

    final_value = (amm.token_x_reserve * current_price) + amm.token_y_reserve
    liquidity_change = (final_value - initial_value) / initial_value
    metrics_df = pd.DataFrame(step_metrics)

    viz_data = {
        'timestamps': timestamps[:len(test_predictions)],
        'actual_prices': test_prices[seq_length:seq_length+len(test_predictions)],
        'predicted_prices': test_predictions,
        'price_impacts': price_impacts,
        'actions': actions,
        'metrics': {
            'total_reward': total_reward,
            'total_fees': amm.total_fees,
            'total_divergence': amm.lvr,
            'total_slippage': amm.slip_loss_total,
            'total_hook_revenue': amm.hook_revenue,
            'total_mev_risk': amm.mev_risk,
            'total_gas_cost': amm.gas_cost,
            'liquidity_change': liquidity_change,
            'price_impact_avg': np.mean(price_impacts) if price_impacts else 0,
            'num_trades': sum(1 for a in actions if a != 0),
            'final_reserves': amm.get_reserves(),
            'final_value': final_value,
            'average_liquidity_utilization': metrics_df['liquidity_utilization'].mean()
        },
        'step_metrics': metrics_df
    }

    print("\nV4 Evaluation Metrics:")
    print(f"Total Reward: {viz_data['metrics']['total_reward']:.2f}")
    print(f"Total Fees: {viz_data['metrics']['total_fees']:.4f}")
    print(f"Hook Revenue: {viz_data['metrics']['total_hook_revenue']:.4f}")
    print(f"Divergence Loss: {viz_data['metrics']['total_divergence']:.4f}")
    print(f"Slippage Loss: {viz_data['metrics']['total_slippage']:.4f}")
    print(f"MEV Risk: {viz_data['metrics']['total_mev_risk']:.4f}")
    print(f"Gas Costs: {viz_data['metrics']['total_gas_cost']}")
    print(f"Average Liquidity Utilization: {viz_data['metrics']['average_liquidity_utilization']:.2%}")
    
    return viz_data
