import numpy as np
import torch
import torch.nn.functional as F


# Updating event detection to account for equilibrium shifts
def detect_price_event(previous_price, current_price, threshold=0.05):
    """More sensitive event detection with volatility-adjusted threshold"""
    volatility = np.std([previous_price, current_price])
    dynamic_threshold = threshold * (1 + volatility*10)
    return abs(current_price - previous_price) > dynamic_threshold

def select_action(q_values, threshold=0.1, beta=0.1): # Using adjusted parameters
    """
    Selects an action based on event-driven price change detection.
    """
    q_values = q_values.squeeze()
    if q_values.dim() == 1 and len(q_values) > 1:
        price_change = abs(q_values[-1] - q_values[-2])
    else:
        price_change = 0  # Default to no significant change

    action = 0 # Default to hold
    if price_change > threshold * beta:
        action = torch.argmax(q_values.squeeze()).item()
    else:
        action = 0  # Hold position
    return action
    
def adaptive_beta(price_history, window=50):
    """
    Dynamically adjusts beta based on market volatility.
    """
    if len(price_history) < window:
        return 0.05 

    volatility = np.std(price_history[-window:])
    return max(0.01, min(0.1, volatility / 10)) 

def compute_expected_reward(predicted_v, observed_v, liquidity_utilization):

    prediction_slippage = abs(predicted_v - observed_v)

    # Compute divergence loss
    divergence_loss = predicted_v * liquidity_utilization - observed_v * liquidity_utilization

    # Compute slippage loss
    slippage_loss = (observed_v - predicted_v) ** 2 / max(observed_v, 1e-6)

    # Compute expected load
    expected_load = divergence_loss * slippage_loss

    # Expected reward function (negative loss for maximization)
    reward = - (prediction_slippage + expected_load) + liquidity_utilization
    return reward

def q_learning_loss(predicted_q, target_q):
    """
    Implements the loss function.
    """
    return F.smooth_l1_loss(predicted_q, target_q)