import numpy as np


class Hook:
    """Base class for Uniswap V4 hooks"""
    def before_swap(self, amm, swap_type):
        return {}
    
    def after_swap(self, amm, swap_type, delta_in, delta_out):
        return {}


class DynamicFeeHook(Hook):
    def before_swap(self, amm, swap_type):
        if len(amm.price_history) >= 2:
            volatility = np.std(amm.price_history[-10:]) if len(amm.price_history) >= 10 else np.std(amm.price_history)
        else:
            volatility = 0.1
        return {'fee': amm.fee_tier * (1 + volatility * 5)}


class RebalanceHook(Hook):
    def after_swap(self, amm, swap_type, delta_in, delta_out):
        current_price = amm.get_price()
        
        if np.isnan(current_price):
            print("RebalanceHook: Detected NaN price. Resetting AMM reserves and price range.")
            amm.reset()
            return {'rebalance': True}
        
        # Compute volatility based on recent price history
        volatility = np.std(amm.price_history[-10:]) if len(amm.price_history) >= 10 else 0.1
        # Compute a dynamic range factor: narrow (5%) in low volatility, wider (up to 10%) in high volatility
        range_factor = max(0.05, min(0.1, volatility * 5))
        new_lower = current_price * (1 - range_factor)
        new_upper = current_price * (1 + range_factor)
        
        current_lower, current_upper = amm.price_range
        current_range = current_upper - current_lower
        new_range = new_upper - new_lower
        
        # Update range if the current price is outside the current range
        # or if the current range is significantly wider than the new dynamic range.
        if (current_price < current_lower or 
            current_price > current_upper or 
            current_range > new_range * 1.1):
            amm.set_price_range(new_lower, new_upper)
            return {'rebalance': True}
        
        return {}


class AutomatedMarketMaker:
    def __init__(self, token_x_reserve, token_y_reserve, fee_tier=0.003, price_range=None):
        self.token_x_reserve = token_x_reserve
        self.token_y_reserve = token_y_reserve
        self.fee_tier = fee_tier
        self.k = token_x_reserve * token_y_reserve
        self.price_range = price_range if price_range else (0, float('inf'))
        
        self.price_history = []
        self.equilibrium_states = []
        self.trade_history = []
        self.hooks = []

        # Metrics
        self.total_fees = 0.0
        self.hook_revenue = 0.0
        self.gas_cost = 0
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.lvr = 0.0          # Cumulative divergence loss
        self.slip_loss_total = 0.0
        self.mev_risk = 0.0
        
        self.in_flash_transaction = False
        self.queued_swaps = []
        self.MIN_RESERVE = 1e-6
        
    def force_rebalance_if_out_of_range(self, swap_type):
        """
        If current_price is outside [price_range[0], price_range[1]], 
        forcibly call after_swap with zero deltas so RebalanceHook can trigger.
        """
        current_price = self.get_price()
        if current_price < self.price_range[0] or current_price > self.price_range[1]:
            # Call after_swap with zero in/out so the RebalanceHook can fix the range
            for hook in self.hooks:
                hook.after_swap(self, swap_type, 0, 0)

    def reset(self):
        """
        Reset AMM reserves, k, and price range to a safe default. 
        Clears price_history so the environment can continue safely.
        Adjust as needed if you prefer partial resets.
        """
        print("[AMM Reset] Reinitializing reserves & range to defaults.")
        self.token_x_reserve = 1000.0
        self.token_y_reserve = 1000.0
        self.k = self.token_x_reserve * self.token_y_reserve
        self.set_price_range(0.1, 10.0)
        
    def get_micro_steps(self, volatility):
        """
        Determine the number of micro-trades based on market volatility.
        A higher volatility will increase the number of micro-trades (up to a max of 50),
        reducing the size of each step and thus mitigating slippage loss.
        """
        return int(max(20, min(50, 20 + int(200 * volatility))))

    def begin_flash_transaction(self):
        """Start a flash loan transaction"""
        self.in_flash_transaction = True
        
    def end_flash_transaction(self):
        """Execute queued swaps with reduced gas cost"""
        self.in_flash_transaction = False
        for swap_fn in self.queued_swaps:
            swap_fn()
        self.gas_cost += 1  # Single gas charge for batch
        self.queued_swaps = []

    def register_hook(self, hook):
        self.hooks.append(hook)

    def swap_x_for_y(self):
        # Return (0,0,0,0) if too small to trade
        if self.token_y_reserve < self.MIN_RESERVE * 10:
            return 0, 0, 0, 0

        if self.in_flash_transaction:
            # Defer the actual swap until end_flash_transaction
            self.queued_swaps.append(lambda: self._swap_x_for_y_impl())
            return 0, 0, 0, 0
        else:
            return self._swap_x_for_y_impl()

    def _swap_x_for_y_impl(self):
        """
        Executes a swap of token X for token Y by splitting the trade into many micro-trades.
        The micro-trade approach reduces instantaneous impact by updating reserves incrementally.
        """
        self.force_rebalance_if_out_of_range('x_for_y')
        current_price = self.get_price()
        if not (self.price_range[0] <= current_price <= self.price_range[1]):
            return 0, 0, 0, 0

        self.token_x_reserve = max(self.token_x_reserve, self.MIN_RESERVE)
        self.token_y_reserve = max(self.token_y_reserve, self.MIN_RESERVE)

        original_fee = self.fee_tier
        fee = original_fee
        for hook in self.hooks:
            params = hook.before_swap(self, 'x_for_y')
            fee = params.get('fee', fee)
        fee = np.clip(fee, 0.0001, 0.05)
        fee_diff = fee - original_fee

        initial_x = self.token_x_reserve
        initial_y = self.token_y_reserve

        vol = np.std(self.price_history[-10:]) if len(self.price_history) >= 10 else 0.1
        total_delta_x = self.get_dynamic_trade_size(volatility=vol)
        fee_amount = total_delta_x * fee
        delta_x_after_fee = total_delta_x - fee_amount

        # Use dynamic micro-trade steps based on volatility
        micro_steps = self.get_micro_steps(vol)
        micro_delta_x = delta_x_after_fee / micro_steps

        aggregated_delta_y = 0.0
        curr_x = initial_x
        curr_y = initial_y

        for _ in range(micro_steps):
            new_x = curr_x + micro_delta_x
            new_y = (curr_x * curr_y) / new_x
            delta_y_i = curr_y - new_y
            aggregated_delta_y += delta_y_i
            curr_x, curr_y = new_x, new_y

        self.delta_x = total_delta_x
        self.delta_y = aggregated_delta_y
        self.token_x_reserve = curr_x
        self.token_y_reserve = curr_y
        self.k = self.token_x_reserve * self.token_y_reserve

        div_loss = self.calculate_divergence_loss(initial_x, initial_y)
        expected_delta_y = delta_x_after_fee * (initial_y / initial_x)
        slip_loss = max((expected_delta_y - aggregated_delta_y) / (expected_delta_y + 1e-8), 0.0)

        self.lvr += div_loss
        self.slip_loss_total += slip_loss
        self.total_fees += fee_amount
        self.hook_revenue += total_delta_x * fee_diff

        self.price_history.append(self.get_price())
        for hook in self.hooks:
            hook.after_swap(self, 'x_for_y', total_delta_x, aggregated_delta_y)

        return (min(aggregated_delta_y, self.token_y_reserve), div_loss, slip_loss, fee_amount)

    def swap_y_for_x(self):
        if self.token_x_reserve < self.MIN_RESERVE * 10:
            return 0, 0, 0, 0

        if self.in_flash_transaction:
            self.queued_swaps.append(lambda: self._swap_y_for_x_impl())
            return 0, 0, 0, 0
        else:
            return self._swap_y_for_x_impl()

    def _swap_y_for_x_impl(self):
        """
        Executes a swap of token Y for token X by splitting the trade into many micro-trades.
        The incremental execution minimizes price impact and lowers slippage.
        """
        self.force_rebalance_if_out_of_range('y_for_x')
        current_price = self.get_price()
        if not (self.price_range[0] <= current_price <= self.price_range[1]):
            return 0, 0, 0, 0

        self.token_x_reserve = max(self.token_x_reserve, self.MIN_RESERVE)
        self.token_y_reserve = max(self.token_y_reserve, self.MIN_RESERVE)

        original_fee = self.fee_tier
        fee = original_fee
        for hook in self.hooks:
            params = hook.before_swap(self, 'y_for_x')
            fee = params.get('fee', fee)
        fee = np.clip(fee, 0.0001, 0.05)
        fee_diff = fee - original_fee

        initial_y = self.token_y_reserve
        initial_x = self.token_x_reserve

        vol = np.std(self.price_history[-10:]) if len(self.price_history) >= 10 else 0.1
        total_delta_y = self.get_dynamic_trade_size(volatility=vol)
        fee_amount = total_delta_y * fee
        delta_y_after_fee = total_delta_y - fee_amount

        micro_steps = self.get_micro_steps(vol)
        micro_delta_y = delta_y_after_fee / micro_steps

        aggregated_delta_x = 0.0
        curr_y = initial_y
        curr_x = initial_x

        for _ in range(micro_steps):
            new_y = curr_y + micro_delta_y
            new_x = (curr_x * curr_y) / new_y
            delta_x_i = curr_x - new_x
            aggregated_delta_x += delta_x_i
            curr_y, curr_x = new_y, new_x

        self.delta_y = total_delta_y
        self.delta_x = aggregated_delta_x
        self.token_y_reserve = curr_y
        self.token_x_reserve = curr_x
        self.k = self.token_x_reserve * self.token_y_reserve

        div_loss = self.calculate_divergence_loss(initial_x, initial_y)
        expected_delta_x = delta_y_after_fee * (initial_x / initial_y)
        slip_loss = max((expected_delta_x - aggregated_delta_x) / (expected_delta_x + 1e-8), 0.0)

        self.lvr += div_loss
        self.slip_loss_total += slip_loss
        self.total_fees += fee_amount
        self.hook_revenue += total_delta_y * fee_diff

        self.price_history.append(self.get_price())
        for hook in self.hooks:
            hook.after_swap(self, 'y_for_x', total_delta_y, aggregated_delta_x)

        return (min(aggregated_delta_x, self.token_x_reserve), div_loss, slip_loss, fee_amount)

    def get_price(self):
        x = max(self.token_x_reserve, self.MIN_RESERVE)
        y = max(self.token_y_reserve, self.MIN_RESERVE)
        return y / x
        
    def get_dynamic_trade_size(self, volatility=0.01):
        """
        Adaptive dynamic trade size:
        - Computes a base trade size as 0.5% of the liquidity depth.
        - Scales the trade size down based on market volatility and recent trade utilization.
        """
        liquidity_depth = (self.token_x_reserve + self.token_y_reserve) / 2.0
        base_trade = liquidity_depth * 0.005  # Reduced base fraction from 1% to 0.5%

        recent_trade_ratio = 0.0
        if self.price_history:
            min_reserve = min(self.token_x_reserve, self.token_y_reserve)
            recent_trade_ratio = max(self.delta_x, self.delta_y) / (min_reserve + 1e-8)

        sensitivity = 1 + volatility * 40 + recent_trade_ratio * 15  # Increased sensitivity multipliers
        dynamic_trade = base_trade / sensitivity

        trade_size = np.clip(
            np.random.normal(dynamic_trade, dynamic_trade * 0.1),
            1e-8,
            dynamic_trade
        )
        return trade_size

    def get_reserves(self):
        return (self.token_x_reserve, self.token_y_reserve)

    def set_price_range(self, lower_bound, upper_bound):
        self.price_range = (lower_bound, upper_bound)

    def calculate_liquidity_utilization(self, trade_volume):
        """
        Computes liquidity utilization metric with dynamic adjustments.
        """
        volatility = np.std(self.price_history[-50:]) if len(self.price_history) >= 50 else 0.01
        liquidity_depth = (self.token_x_reserve + self.token_y_reserve) / 2
        
        # Lower the scaling factor => more aggressive utilization measurement
        if volatility < 0.005:
            factor = 0.001
        else:
            factor = 0.002
        
        liquidity_utilization = trade_volume / (liquidity_depth * factor)
        return min(liquidity_utilization, 1.0)

    def calculate_divergence_loss(self, initial_x, initial_y):
        current_price = self.get_price()
        if current_price <= 0:
            return 0.0
        
        value_held = (initial_x * current_price) + initial_y
        value_pool = (self.token_x_reserve * current_price) + self.token_y_reserve
        
        if value_held <= 0:
            return 0.0
        
        loss = (value_held - value_pool) / value_held
        loss = round(loss, 4)  # four-decimal precision
        return max(loss, 0.0)

    def calculate_slippage_loss(self, y, delta_y_after_fee, x):
        """
        Calculate slippage loss using a simple price impact formula.
        """
        if delta_y_after_fee <= 0 or y <= 1e-8 or x <= 1e-8:
            return 0.0
        try:
            price_before = x / y
            new_y = y + delta_y_after_fee
            new_x = (x * y) / new_y
            delta_x = x - new_x
            executed_price = delta_x / delta_y_after_fee
            slippage = abs((price_before - executed_price) / (price_before + 1e-8))
            return max(slippage, 0.0)
        except Exception as e:
            print("[calculate_slippage_loss] error:", e)
            return 0.0

    def update_market_price(self, new_market_price, predicted_v=None):
        """
        More defensive update that clamps or resets on invalid values.
        """
        # Validate incoming price
        if new_market_price is None or not np.isfinite(new_market_price) or new_market_price <= 0:
            print("[update_market_price] Invalid new_market_price:", new_market_price)
            self.reset()
            return

        # Current price must also be valid
        current_price = self.get_price()
        if not np.isfinite(current_price) or current_price <= 0:
            print("[update_market_price] Invalid current_price:", current_price)
            self.reset()
            return

        # Compute ratio and clamp to [0.1, 10]
        price_ratio = np.clip(new_market_price / current_price, 0.1, 10.0)

        # Log-scale the ratio, clamp to [-0.05, 0.05] => ~5% step
        adjustment_factor = np.log(price_ratio)
        adjustment_factor = np.clip(adjustment_factor, -0.05, 0.05)

        # Update x_reserve
        new_x_reserve = self.token_x_reserve * np.exp(adjustment_factor)
        new_x_reserve = np.clip(new_x_reserve, 1e-8, 1e12)

        # If k is invalid, reset
        if not np.isfinite(self.k) or self.k <= 0:
            print("[update_market_price] Invalid k:", self.k)
            self.reset()
            return

        # new_y_reserve = k / new_x_reserve
        new_y_reserve = self.k / new_x_reserve
        if not np.isfinite(new_y_reserve) or new_y_reserve <= 0:
            print("[update_market_price] new_y_reserve is invalid:", new_y_reserve)
            self.reset()
            return
        new_y_reserve = np.clip(new_y_reserve, 1e-8, 1e12)

        # Finalize
        self.token_x_reserve = new_x_reserve
        self.token_y_reserve = new_y_reserve
        self.k = self.token_x_reserve * self.token_y_reserve

        # If k became non-finite, reset
        if not np.isfinite(self.k):
            print("[update_market_price] k became non-finite:", self.k)
            self.reset()

    def pseudo_arbitrage(self, new_market_price):
        current_price = self.get_price()
        if current_price == new_market_price:
            return
        
        x = self.token_x_reserve
        y = self.token_y_reserve
        value_before = (x * current_price) + y
        
        # Use a less aggressive damping factor (alpha = 0.5)
        if new_market_price > current_price:
            new_x = np.sqrt((x * y) / new_market_price)
            delta_x = new_x - x
            delta_y = y - (x * y) / new_x
            alpha = 0.5
            self.token_x_reserve = (1 - alpha) * x + alpha * new_x
            self.token_y_reserve = (1 - alpha) * y + alpha * (y - delta_y)
        else:
            new_y = np.sqrt((x * y) * new_market_price)
            delta_y = new_y - y
            delta_x = x - (x * y) / new_y
            alpha = 0.5
            self.token_y_reserve = (1 - alpha) * y + alpha * new_y
            self.token_x_reserve = (1 - alpha) * x + alpha * (x - delta_x)
        
        self.k = self.token_x_reserve * self.token_y_reserve
        value_after = (self.token_x_reserve * new_market_price) + self.token_y_reserve
        if value_before > 0:
            mev_loss = max((value_before - value_after) / value_before, 0.0)
            self.mev_risk += mev_loss

