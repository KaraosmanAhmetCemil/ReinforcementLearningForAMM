import numpy as np

class AutomatedMarketMaker:
    def __init__(self, token_x_reserve, token_y_reserve, fee_tier=0.003, price_range=None):
        self.token_x_reserve = token_x_reserve
        self.token_y_reserve = token_y_reserve
        self.fee_tier = fee_tier
        self.k = token_x_reserve * token_y_reserve
        self.price_range = price_range if price_range else (0, float('inf'))
        self.volume_scale = 100 
        self.price_history = [] 
        self.equilibrium_states = []
        self.trade_history = []

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
        current_price = self.get_price()
        if not (self.price_range[0] <= current_price <= self.price_range[1]):
            return 0, 0, 0  # Return (0, 0, 0) for failed trades
    
        # Track pre-trade state
        initial_x = self.token_x_reserve
        delta_x = self.get_dynamic_trade_size()
        
        # Execute trade
        amount_x_with_fee = delta_x * (1 - self.fee_tier)
        new_x = initial_x + amount_x_with_fee
        new_y = self.k / new_x
        delta_y = self.token_y_reserve - new_y
    
        if delta_y > 0:
            # Calculate losses using paper's formulas
            div_loss = self.calculate_divergence_loss(initial_x, delta_x)
            slip_loss = self.calculate_slippage_loss(initial_x, delta_x)
            
            # Update reserves
            self.token_x_reserve = new_x
            self.token_y_reserve = new_y
            return delta_y, div_loss, slip_loss
            
        return 0, 0, 0  # Return (0, 0, 0) for failed trades

    def swap_y_for_x(self):
        """Swaps Y for X with proper loss tracking using paper's equations"""
        current_price = self.get_price()
        if not (self.price_range[0] <= current_price <= self.price_range[1]):
            return 0, 0, 0  # Return (0, 0, 0) for out-of-range trades
    
        # Track pre-trade state
        initial_y = self.token_y_reserve
        delta_y = self.get_dynamic_trade_size()  # Amount of Y being deposited
        
        # Execute trade
        amount_y_with_fee = delta_y * (1 - self.fee_tier)
        new_y = initial_y + amount_y_with_fee
        new_x = self.k / new_y if new_y > 0 else self.token_x_reserve
        delta_x = self.token_x_reserve - new_x
    
        if delta_x > 0:
            x_at_swap = self.k / initial_y 
            equivalent_delta_x = x_at_swap - (self.k / new_y) 
            
            # Calculate losses using paper's formulas with X terms
            div_loss = self.calculate_divergence_loss(x_at_swap, equivalent_delta_x)
            slip_loss = self.calculate_slippage_loss(x_at_swap, equivalent_delta_x)
            
            # Update reserves
            self.token_x_reserve = new_x
            self.token_y_reserve = new_y
            return delta_x, div_loss, slip_loss
        
        return 0, 0, 0  # Return (0, 0, 0) for failed trades

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
        """
        Computes liquidity utilization with dynamic adjustments.
        The trade_volume is scaled to match the AMM's liquidity scale.
        """
        volatility = np.std(self.price_history[-50:]) if len(self.price_history) >= 50 else 0.0
        liquidity_depth = (self.token_x_reserve + self.token_y_reserve) / 2
        
        # Scale the trade volume from data to match liquidity
        scaled_trade_volume = trade_volume * self.volume_scale

        # Use a lower denominator factor if volatility is low
        if volatility < 0.005:
            liquidity_utilization = scaled_trade_volume / (liquidity_depth * 0.8)
        else:
            liquidity_utilization = scaled_trade_volume / (liquidity_depth * 1.2)

        return min(liquidity_utilization, 1.0)  # Cap at 100%

    def calculate_divergence_loss(self, x, delta_x):
        """Computes CPMM divergence loss per paper's equation"""
        numerator = delta_x ** 2
        denominator = 2 * delta_x * (x ** 2) + x ** 3 + (delta_x ** 2) * x + x
        return abs(numerator / denominator) if denominator != 0 else 0

    def calculate_slippage_loss(self, x, delta_x):
        numerator = -delta_x**2 * (delta_x + x)
        denominator = x**2 * (delta_x**2 + x**2 + 2*delta_x*x + 1)
        return abs(numerator / denominator) if denominator != 0 else 0

    def update_market_price(self, new_market_price, predicted_v=None):
        prev_state = (self.token_x_reserve, 
                      self.token_y_reserve,
                      self.get_price())
        self.equilibrium_states.append(prev_state)
        
        current_price = self.get_price()
        self.price_history.append(current_price)
        
        # Increase weight on the predicted price for faster convergence
        target_price = predicted_v if predicted_v is not None else new_market_price
        equilibrium_price = current_price * 0.3 + target_price * 0.7  # shifted weights
        
        price_ratio = equilibrium_price / current_price
        # Allow a wider range of adjustment
        adjustment_factor = np.clip(price_ratio, 0.8, 1.2)
        
        incentive = self.compute_liquidity_incentive()
        new_x_reserve = self.token_x_reserve * (adjustment_factor + incentive)
        # Relax clipping to allow a 20% change instead of 10%
        new_x_reserve = np.clip(new_x_reserve, self.token_x_reserve * 0.8, self.token_x_reserve * 1.2)
        
        new_y_reserve = self.k / new_x_reserve
        self.token_x_reserve, self.token_y_reserve = new_x_reserve, new_y_reserve
        self.k = self.token_x_reserve * self.token_y_reserve

    def pseudo_arbitrage(self, new_market_price):
        current_price = self.get_price()
        if current_price == 0:
            return
        
        # Allow a broader adjustment range
        target_ratio = np.clip(new_market_price / current_price, 0.8, 1.2)
        new_x_reserve = np.clip(
            self.token_x_reserve * target_ratio,
            self.token_x_reserve * 0.8,
            self.token_x_reserve * 1.2
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

