import os
import matplotlib.pyplot as plt


def visualization(test_data, test_predictions_original, test_prices, liquidity_utilization, divergence_loss, slippage_loss, seq_length,
                  train_losses, validation_losses, step_rewards):

    if not os.path.exists('graphs'):
        os.mkdir('graphs')

    # Liquidity Utilization Comparison
    plt.figure(figsize=(12, 6))
    baseline_liquidity_utilization = 56  # Uniswap V3 baseline
    plt.bar(['Baseline AMM', 'Proposed AMM'],
            [baseline_liquidity_utilization, liquidity_utilization],
            color=['red', 'blue'])
    plt.xlabel('AMM Structure')
    plt.ylabel('Liquidity Utilization (%)')
    plt.title('Liquidity Utilization Comparison – PPO (1-Minute Klines)')
    plt.ylim(0, 100)
    plt.savefig('graphs/liquidity_utilization.png')
    plt.show()

    # Divergence Loss Comparison
    plt.figure(figsize=(12, 6))
    baseline_divergence_loss = 1.465
    plt.bar(['Baseline AMM', 'Proposed AMM'], 
            [baseline_divergence_loss, divergence_loss], 
            color=['blue', 'green'])
    plt.xlabel('AMM Structure')
    plt.ylabel('Average Divergence Loss')
    plt.title('Comparison of Average Divergence Losses – PPO (1-Minute Klines)')
    plt.savefig('graphs/divergence_loss.png')
    plt.show()

    # Slippage Loss Comparison
    plt.figure(figsize=(12, 6))
    baseline_slippage_loss = 0.4779
    plt.bar(['Baseline AMM', 'Proposed AMM'], 
            [baseline_slippage_loss, slippage_loss], 
            color=['red', 'green'])
    plt.xlabel('AMM Structure')
    plt.ylabel('Average Slippage')
    plt.title('Comparison of Average Slippage – PPO (1-Minute Klines)')
    plt.savefig('graphs/slippage_loss.png')
    plt.show()

    # 1. Actual vs Predicted Prices
    plt.figure(figsize=(12, 6))
    # Extract actual prices corresponding to the predictions
    actual_prices = test_prices[seq_length:seq_length+len(test_predictions_original)]
    plt.plot(actual_prices, label="Actual Prices")
    plt.plot(test_predictions_original, label="Predicted Prices")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.title("Actual vs Predicted Prices – PPO (1-Minute Klines)")
    plt.legend()
    plt.savefig("graphs/actual_vs_predicted.png")
    plt.show()

    # 2. Train vs Validation Loss over Epochs
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss – PPO (1-Minute Klines)")
    plt.legend()
    plt.savefig("graphs/train_validation_loss.png")
    plt.show()

    # 3. Reward over Steps
    plt.figure(figsize=(12, 6))
    steps = range(1, len(step_rewards) + 1)
    plt.plot(steps, step_rewards, label="Reward per Step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward over Steps – PPO (1-Minute Klines)")
    plt.legend()
    plt.savefig("graphs/reward_over_steps.png")
    plt.show()