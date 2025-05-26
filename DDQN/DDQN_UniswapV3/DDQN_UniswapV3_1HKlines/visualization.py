import os
import matplotlib.pyplot as plt


def visualization(test_data, test_predictions_original, test_prices, liquidity_utilization, divergence_loss, slippage_loss, 
                  seq_length, train_losses, val_losses, rewards_over_steps):
    """Performs visualization of metrics."""
    if not os.path.exists('graphs'):
        os.mkdir('graphs')

    # Actual vs Predicted Prices
    actual_prices = test_prices[seq_length:len(test_predictions_original)+seq_length]
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Prices')
    plt.plot(test_predictions_original, label='Predicted Prices')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Prices – DDQN (1-Hour Klines)')
    plt.legend()
    plt.savefig('graphs/actual_vs_predicted.png')
    plt.show()

    # Train and Validation Losses
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Losses – DDQN (1-Hour Klines)')
    plt.legend()
    plt.savefig('graphs/train_val_loss.png')
    plt.show()

    # Reward Over Steps
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_over_steps, label='Reward Over Steps – DDQN (1-Hour Klines)')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward Over Steps')
    plt.legend()
    plt.savefig('graphs/reward_over_steps.png')
    plt.show()

    baseline_liquidity_utilization = 56 
    baseline_divergence_loss = 1.465
    baseline_slippage_loss = 0.4779

    # Liquidity Utilization Comparison
    plt.figure(figsize=(12, 6))
    plt.bar(['Baseline AMM', 'Proposed AMM'],
            [baseline_liquidity_utilization, liquidity_utilization],
            color=['red', 'blue'])
    plt.xlabel('AMM Structure')
    plt.ylabel('Liquidity Utilization (%)')
    plt.title('Liquidity Utilization Comparison – DDQN (1-Hour Klines)')
    plt.ylim(0, 100)
    plt.savefig('graphs/liquidity_utilization.png')
    plt.show()

    # Divergence Loss Comparison
    plt.figure(figsize=(12, 6))
    plt.bar(['Baseline AMM', 'Proposed AMM'], 
            [baseline_divergence_loss, divergence_loss], 
            color=['blue', 'green'])
    plt.xlabel('AMM Structure')
    plt.ylabel('Average Divergence Loss')
    plt.title('Comparison of Average Divergence Losses – DDQN (1-Hour Klines)')
    plt.savefig('graphs/divergence_loss.png')
    plt.show()

    # Slippage Loss Comparison
    plt.figure(figsize=(12, 6))
    plt.bar(['Baseline AMM', 'Proposed AMM'], 
            [baseline_slippage_loss, slippage_loss], 
            color=['red', 'green'])
    plt.xlabel('AMM Structure')
    plt.ylabel('Average Slippage') 
    plt.title('Comparison of Average Slippage – DDQN (1-Hour Klines)')
    plt.savefig('graphs/slippage_loss.png')
    plt.show()