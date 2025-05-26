import numpy as np
import torch
import torch.nn.functional as F


# Updating event detection to account for equilibrium shifts
def detect_price_event(previous_price, current_price, threshold=0.05):
    """More sensitive event detection with volatility-adjusted threshold"""
    volatility = np.std([previous_price, current_price])
    dynamic_threshold = threshold * (1 + volatility*10)
    return abs(current_price - previous_price) > dynamic_threshold

def select_action(state, actor_critic):
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        probs, value = actor_critic(state_tensor)
    dist = Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action), value.item()

def compute_expected_reward(predicted_v, observed_v, liquidity_utilization):

    prediction_slippage = abs(predicted_v - observed_v)
    
    # Original divergence loss computation can result in very large numbers.
    divergence_loss = predicted_v * liquidity_utilization - observed_v * liquidity_utilization
    slippage_loss = (observed_v - predicted_v) ** 2 / max(observed_v, 1e-6)
    expected_load = divergence_loss * slippage_loss
    
    # Reward is negative loss (for maximization) with a bonus for liquidity utilization.
    reward = - (prediction_slippage + expected_load) + liquidity_utilization
    
    # Scale reward down to reduce magnitude
    return reward / 100.0
