import os
import matplotlib.pyplot as plt


def visualization(viz_data, training_metrics=None):

    if not os.path.exists('graphs'):
        os.mkdir('graphs')

    plt.figure(figsize=(12, 6))
    baseline_liquidity_utilization = 56 
    plt.bar(['Baseline AMM', 'Proposed Uniswap V4 Based AMM'],
            [baseline_liquidity_utilization, viz_data['metrics']['average_liquidity_utilization'] * 100],
            color=['red', 'blue'])
    plt.xlabel('AMM Structure')
    plt.ylabel('Liquidity Utilization (%)')
    plt.title('Liquidity Utilization Comparison – DDQN with Hooks and updated reward function (1-Hour Klines)')
    plt.ylim(0, 100)
    plt.savefig('graphs/liquidity_utilization_ddqn_v4_1h.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    baseline_divergence_loss = 1.465
    plt.bar(['Baseline AMM', 'Proposed Uniswap V4 Based AMM'], 
            [baseline_divergence_loss, viz_data['metrics']['total_divergence']], 
            color=['blue', 'green'])
    plt.xlabel('AMM Structure')
    plt.ylabel('Average Divergence Loss')
    plt.yscale('log')
    plt.title('Comparison of Average Divergence Losses – DDQN with Hooks and updated reward function (1-Hour Klines)')
    plt.savefig('graphs/divergence_loss_ddqn_v4_1h.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    baseline_slippage_loss = 0.4779
    plt.bar(['Baseline AMM', 'Proposed Uniswap V4 Based AMM'], 
            [baseline_slippage_loss, viz_data['metrics']['total_slippage']], 
            color=['red', 'green'])
    plt.xlabel('AMM Structure')
    plt.ylabel('Average Slippage') 
    plt.title('Comparison of Average Slippage – DDQN with Hooks and updated reward function (1-Hour Klines)')
    plt.savefig('graphs/slippage_loss_ddqn_v4_1h.png')
    plt.show()

    # LSTM Training and Validation Loss Curve
    if training_metrics is not None:
        epochs = range(1, len(training_metrics['train_loss']) + 1)
        plt.figure(figsize=(10,6))
        plt.plot(epochs, training_metrics['train_loss'], label='Train Loss')
        plt.plot(epochs, training_metrics['val_loss'], label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("LSTM Training and Validation Loss – DDQN with Hooks and updated reward function (1-Hour Klines)")
        plt.legend()
        plt.savefig('graphs/lstm_loss_curve_ddqn_v4_1h.png')

    # Reward Curve over Steps
    rewards = viz_data['step_metrics']['reward'].values
    steps = viz_data['step_metrics']['step'].values
    plt.figure(figsize=(10,6))
    plt.plot(steps, rewards, label='Reward per Step')
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward Curve over Steps – DDQN with Hooks and updated reward function (1-Hour Klines)")
    plt.legend()
    plt.savefig('graphs/reward_curve_ddqn_v4_1h.png')

    # Predicted vs. Actual Prices
    plt.figure(figsize=(10,6))
    plt.plot(viz_data['timestamps'], viz_data['actual_prices'], label='Actual Price')
    plt.plot(viz_data['timestamps'], viz_data['predicted_prices'], label='Predicted Price')
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.title("Predicted vs Actual Prices – DDQN with Hooks and updated reward function (1-Hour Klines)")
    plt.legend()
    plt.savefig('graphs/predicted_vs_actual_ddqn_v4_1h.png')

    plt.show()