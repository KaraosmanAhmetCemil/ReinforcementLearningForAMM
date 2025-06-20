import numpy as np
import torch
import torch.nn.functional as F
    

def detect_price_event(previous_price, current_price, threshold=0.005):
    """Triggers a price event if the relative change exceeds the threshold (default 0.5%)."""
    if previous_price <= 0:
        return False
    relative_change = abs(current_price - previous_price) / previous_price
    return relative_change > threshold

def select_action_ddqn(state, ddqn_network, epsilon, device):
    if np.random.rand() < epsilon:
        return np.random.randint(0, ddqn_network.q_head.out_features)
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_values = ddqn_network(state_tensor)
        return torch.argmax(q_values, dim=1).item()

def compute_expected_reward(predicted_v, observed_v, metadata):
    """Reward calculation using the formula:
    Reward = α*(Fees−LVR) − β*GasCost + γ*HookRevenue − δ*MEVRisk
    """
    
    ALPHA = 2.0  # fee weight
    BETA = 0.02  # gas penalty
    GAMMA = 1.0  # hook incentive
    DELTA = 0.1  # MEV risk penalty
    EPSILON = 0.5 # constant for slippage
    
    if isinstance(metadata, dict):
        fees = min(metadata.get('fees', 0), 1e6)  # Prevent overflow
        lvr = max(metadata.get('lvr', 0), -1e6)
        slippage = max(metadata.get('slippage', 0), -1e6)
        gas_cost = metadata.get('gas_cost', 0)
        hook_revenue = metadata.get('hook_revenue', 0.0)
        mev_risk = metadata.get('mev_risk', 0.0)
    elif isinstance(metadata, np.ndarray):
        fees = min(metadata[0] if len(metadata) > 0 else 0.0, 1e6)
        lvr = max(metadata[1] if len(metadata) > 1 else 0.0, -1e6)
        slippage = max(metadata[2] if len(metadata) > 2 else 0.0, -1e6)
        gas_cost = metadata[3] if len(metadata) > 3 else 0
        hook_revenue = metadata[4] if len(metadata) > 4 else 0.0
        mev_risk = metadata[5] if len(metadata) > 5 else 0.0
    else:
        fees = lvr = slippage = gas_cost = hook_revenue = mev_risk = 0.0

    reward = (ALPHA * (fees - lvr) 
              - BETA * gas_cost 
              + GAMMA * hook_revenue 
              - DELTA * mev_risk
              - EPSILON * slippage)
    
    return float(reward)