import numpy as np

class AutomatedMarketMaker:
    def __init__(self, token_x_reserve, token_y_reserve, fee_tier=0.003, price_range=None):
        self.token_x_reserve = token_x_reserve
        self.token_y_reserve = token_y_reserve
        self.fee_tier = fee_tier
        self.k = token_x_reserve * token_y_reserve
        self.price_range = price_range if price_range else (0, float('inf'))
        self.price_history = []  # Add price tracking for volatility calculation

    def get_price(self):
        """
        Returns the current price of token X in terms of token Y.
        """
        return self.token_y_reserve / self.token_x_reserve

    def get_dynamic_trade_size(self, volatility=0.01):
        """More responsive trade sizing based on current reserves"""
        impact_threshold = 0.01  # Max 1% price impact
        liquidity_depth = (self.token_x_reserve + self.token_y_reserve) / 2
        
        # Dynamic scaling based on both volatility and liquidity depth
        max_trade = liquidity_depth * impact_threshold * (1 - min(volatility/0.05, 1))
        min_trade = max(liquidity_depth * 0.001, 1)  # At least 0.1% of liquidity
        return np.clip(np.random.normal(max_trade/2, max_trade/4), min_trade, max_trade)

    def swap_x_for_y(self):
        """
        Swaps token X for token Y using the Uniswap V3 model with concentrated liquidity.
        """
        if self.token_x_reserve == 0 or self.token_y_reserve == 0:
            raise ValueError("Insufficient liquidity for trade.")

        current_price = self.get_price()
        if not (self.price_range[0] <= current_price <= self.price_range[1]):
            return 0  # Return zero if price is out of bounds instead of error

        amount_x = self.get_dynamic_trade_size()  # Use adaptive trade size
        amount_x_with_fee = amount_x * (1 - self.fee_tier)
        new_x_reserve = self.token_x_reserve + amount_x_with_fee
        new_y_reserve = self.k / new_x_reserve if new_x_reserve > 0 else self.token_y_reserve
        amount_y_out = max(self.token_y_reserve - new_y_reserve, 0)  # Prevent negative outputs

        # Update reserves only if valid trade
        if amount_y_out > 0:
            self.token_x_reserve = new_x_reserve
            self.token_y_reserve = new_y_reserve

        return amount_y_out

    def swap_y_for_x(self):
        """
        Swaps token Y for token X using the Uniswap V3 model with concentrated liquidity.
        """
        if self.token_x_reserve == 0 or self.token_y_reserve == 0:
            raise ValueError("Insufficient liquidity for trade.")

        current_price = self.get_price()
        if not (self.price_range[0] <= current_price <= self.price_range[1]):
            return 0  # Return zero if price is out of bounds instead of error

        amount_y = self.get_dynamic_trade_size()  # Use adaptive trade size
        amount_y_with_fee = amount_y * (1 - self.fee_tier)
        new_y_reserve = self.token_y_reserve + amount_y_with_fee
        new_x_reserve = self.k / new_y_reserve if new_y_reserve > 0 else self.token_x_reserve
        amount_x_out = max(self.token_x_reserve - new_x_reserve, 0)  # Prevent negative outputs

        # Update reserves only if valid trade
        if amount_x_out > 0:
            self.token_x_reserve = new_x_reserve
            self.token_y_reserve = new_y_reserve

        return amount_x_out

    def get_reserves(self):
        """
        Returns the current reserves of token X and token Y.
        """
        return self.token_x_reserve, self.token_y_reserve

    def set_price_range(self, lower_bound, upper_bound):
        """
        Sets the price range for concentrated liquidity.
        """
        self.price_range = (lower_bound, upper_bound)
        
    def calculate_liquidity_utilization(self, trade_volume):
        liquidity_depth = (self.token_x_reserve + self.token_y_reserve) / 2
        # Removed volatility factor and apply a multiplier to raise the utilization level.
        liquidity_utilization = (trade_volume * 50) / liquidity_depth
        return min(liquidity_utilization, 1.0)

    def calculate_divergence_loss(self, initial_funds, final_funds):
        """
        Computes divergence loss as difference between initial and final funds.
        """
        return abs(initial_funds - final_funds)

    def calculate_slippage_loss(self, expected_price, actual_price):
        """
        Computes slippage loss as difference between expected and actual execution price.
        """
        return abs(expected_price - actual_price) / expected_price

    def update_market_price(self, new_market_price, predicted_v=None):
        current_price = self.get_price()
        self.price_history.append(current_price)
        
        # Use dampened price adjustment (80% weight to current price)
        target_price = predicted_v if predicted_v is not None else new_market_price
        equilibrium_price = current_price * 0.8 + target_price * 0.2
        
        # More conservative adjustment with volatility damping
        price_ratio = equilibrium_price / current_price
        adjustment_factor = np.clip(price_ratio, 0.9, 1.1)  # Max ±10% adjustment
        
        # Add small incentive based on price momentum
        incentive = self.compute_liquidity_incentive()
        new_x_reserve = np.clip(
            self.token_x_reserve * (adjustment_factor + incentive),
            min(self.token_x_reserve*0.5, 1e4),  # Dynamic lower bound
            max(self.token_x_reserve*2, 1e4)     # Dynamic upper bound
        )
        
        # Maintain constant product and update reserves
        new_y_reserve = self.k / new_x_reserve
        self.token_x_reserve, self.token_y_reserve = new_x_reserve, new_y_reserve
        self.k = self.token_x_reserve * self.token_y_reserve

    def pseudo_arbitrage(self, new_market_price):
        current_price = self.get_price()
        if current_price == 0:
            return
        
        # Gradual price convergence (max 5% adjustment per call)
        target_ratio = np.clip(new_market_price / current_price, 0.95, 1.05)
        new_x_reserve = np.clip(
            self.token_x_reserve * target_ratio,
            self.token_x_reserve*0.9,
            self.token_x_reserve*1.1
        )
        new_y_reserve = self.k / new_x_reserve
        self.token_x_reserve, self.token_y_reserve = new_x_reserve, new_y_reserve

    def compute_liquidity_incentive(self, std_dev=0.005):
        """Generate normalized liquidity incentive based on price momentum"""
        if len(self.price_history) < 2:
            return 0
            
        # Calculate price momentum as percentage change
        momentum = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
        # Generate incentive proportional to momentum with noise
        return np.clip(
            np.random.normal(momentum*0.1, std_dev),
            -0.01,
            0.01
        )
