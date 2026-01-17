# region imports
from AlgorithmImports import *
# endregion

class RiskManagedStrangleStrategy(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2025, 1, 1)
        self.set_cash(100000)
        self.settings.seed_initial_prices = True
        
        # Add SPY equity and options
        self._spy = self.add_equity("NVDA", Resolution.HOUR).symbol
        self._spy_option = self.add_option("NVDA", Resolution.HOUR).symbol
        self.securities[self._spy_option].set_filter(lambda u: u.strikes(-60, 60).expiration(25, 60))
        
        # Add VIX for volatility filter
        self._vix = self.add_data(CBOE, "VIX", Resolution.DAILY).symbol

        # ---- NVDA-specific volatility proxy (PRIMARY) ----
        self._atr = self.atr(self._spy, 14, MovingAverageType.WILDERS, Resolution.DAILY)
        self.set_warm_up(30, Resolution.DAILY)

        # ATR% rules (tune these)
        self._max_atr_pct_entry = 0.055   # skip new trades if ATR% > 5.5%
        self._atr_low_pct = 0.030         # <3.0% = calm
        self._atr_high_pct = 0.045        # >4.5% = elevated
        
        # Improved risk management parameters
        self._total_stop_loss_multiplier = 2.5  # Close if total position loses 2.5x initial credit
        self._partial_profit_target = 0.25  # Close half at 25% profit
        self._full_profit_target = 0.50  # Close remainder at 50% profit
        self._max_vix = 20  # Lower VIX threshold for better entries
        self._max_delta_exit = 0.30  # Close if delta exceeds 0.30 (position getting tested)
        self._max_positions = 4  # Allow up to 4 concurrent positions
        
        # Dynamic position sizing based on VIX
        self._base_position_size_pct = 0.15  # Base 15% of portfolio per position
        self._vix_low_threshold = 12  # Below this, increase size
        self._vix_high_threshold = 18  # Above this, decrease size
        
        # Delta targeting - tighter tolerance
        self._target_delta = 0.10
        self._delta_tolerance = 0.03  # Accept 0.07-0.13 delta
        
        # Track open positions
        self._open_positions = {}
        self._position_counter = 0
        
        # Schedule weekly strangle opening (every Monday)
        self.schedule.on(self.date_rules.every(DayOfWeek.MONDAY), 
                        self.time_rules.after_market_open(self._spy, 30), 
                        self._open_strangle)
        
        # Check positions daily for management
        self.schedule.on(self.date_rules.every_day(self._spy),
                        self.time_rules.after_market_open(self._spy, 35),
                        self._manage_positions)

    def _open_strangle(self):
        """Open a new 10 delta strangle"""
        # Get option chain - need to wait for next data slice
        self._should_open_strangle = True
    
    def on_data(self, data: Slice):
        """Process data and open/manage positions"""
        if hasattr(self, '_should_open_strangle') and self._should_open_strangle:
            self._should_open_strangle = False
            self._execute_strangle_open(data)
    
    def _execute_strangle_open(self, data: Slice):
        """Execute the strangle opening with data slice"""
        # Check if we've reached max positions
        if len(self._open_positions) >= self._max_positions:
            return
        
        # VIX filter - don't open in high volatility
        if self._vix in data and data[self._vix]:
            current_vix = data[self._vix].value
            if current_vix > self._max_vix:
                self.debug(f"Skipping entry - VIX too high: {current_vix:.2f}")
                return
         
        # ---- NVDA-specific volatility proxy: ATR% ----
        atr_pct = self._get_atr_pct()
        if atr_pct is None:
            self.debug("ATR% not ready yet")
            return

        # Entry skip in extreme NVDA volatility regimes
        if atr_pct > self._max_atr_pct_entry:
            self.debug(f"Skipping entry - NVDA ATR% too high: {atr_pct*100:.2f}%")
            return

        # Dynamic sizing based on NVDA ATR%
        # lower ATR% => slightly bigger size; higher ATR% => smaller size
        if atr_pct < self._atr_low_pct:
            size_multiplier = 1.25
        elif atr_pct > self._atr_high_pct:
            size_multiplier = 0.65
        else:
            size_multiplier = 1.0

        adjusted_size_pct = self._base_position_size_pct * size_multiplier
            
        # Get option chain
        chain = data.option_chains.get(self._spy_option)
        if not chain:
            return

        # ---- Expiry-first selection (30-45 DTE) ----
        target_dte_min, target_dte_max = 30, 45

        expiries = sorted({
            x.expiry for x in chain
            if target_dte_min <= (x.expiry - self.time).days <= target_dte_max
            and x.greeks and x.greeks.delta is not None
        })

        if not expiries:
            self.debug("No expiries in 30-45 DTE with greeks")
            return

        # Choose the nearest expiry in the window (most liquid typically)
        expiry = expiries[0]
            
        # Filter for options expiring in 30-45 days with valid greeks
        calls = [x for x in chain
                if x.expiry == expiry and x.right == OptionRight.CALL
                and x.greeks and x.greeks.delta is not None
                and x.bid_price > 0 and x.ask_price > 0]

        puts = [x for x in chain
                if x.expiry == expiry and x.right == OptionRight.PUT
                and x.greeks and x.greeks.delta is not None
                and x.bid_price > 0 and x.ask_price > 0]

        if not calls or not puts:
            self.debug("No calls/puts for chosen expiry with valid bid/ask + greeks")
            return
        
        # Find 10 delta options with tighter tolerance
        target_delta = self._target_delta
        tol = self._delta_tolerance
        
        # Sort calls by how close they are to target delta
        call_contract = min(calls, key=lambda x: abs(abs(x.greeks.delta) - target_delta))
        
        # Sort puts by how close they are to target delta  
        put_contract = min(puts, key=lambda x: abs(abs(x.greeks.delta) - target_delta))
        
        self.debug(f"Best call delta={abs(call_contract.greeks.delta):.3f}, best put delta={abs(put_contract.greeks.delta):.3f}, count={len(chain)}")

        # Ensure same expiration
        if call_contract.expiry != put_contract.expiry:
            self.debug("Could not find matching expiration dates")
            return
        
        # Verify deltas are reasonable (between 0.05 and 0.15)
        # if abs(call_contract.greeks.delta) < 0.05 or abs(call_contract.greeks.delta) > 0.15:
        #     return
        # if abs(put_contract.greeks.delta) < 0.05 or abs(put_contract.greeks.delta) > 0.15:
        #     return

        if abs(abs(call_contract.greeks.delta) - target_delta) > tol:
            self.debug(f"No suitable CALL delta. Best={abs(call_contract.greeks.delta):.3f}")
            return

        if abs(abs(put_contract.greeks.delta) - target_delta) > tol:
            self.debug(f"No suitable PUT delta. Best={abs(put_contract.greeks.delta):.3f}")
            return

        # Spread sanity check (skip wide markets)
        def spread_ok(c):
            mid = 0.5 * (c.bid_price + c.ask_price)
            if mid <= 0:
                return False
            return (c.ask_price - c.bid_price) <= mid * 0.08   # <= 8% of mid
        
        if not spread_ok(call_contract) or not spread_ok(put_contract):
            self.debug("Skipping entry - spreads too wide")
            return

        # Dynamic position sizing based on VIX
        current_vix = data[self._vix].value if self._vix in data and data[self._vix] else 15
        
        # Adjust size based on VIX: lower VIX = larger position
        if current_vix < self._vix_low_threshold:
            size_multiplier = 1.3  # 30% larger in low VIX
        elif current_vix > self._vix_high_threshold:
            size_multiplier = 0.7  # 30% smaller in elevated VIX
        else:
            size_multiplier = 1.0
        
        adjusted_size_pct = self._base_position_size_pct * size_multiplier
        
        # Calculate contracts based on portfolio value and premium
        option_premium = (call_contract.bid_price + put_contract.bid_price) * 100
        max_contracts = 1
        # max(1, int(self.portfolio.total_portfolio_value * adjusted_size_pct / option_premium))
        
        # --- Place limit orders near mid (better fills than market orders) ---
        def mid_price(c):
            return round(0.5 * (c.bid_price + c.ask_price), 2)

        call_mid = mid_price(call_contract)
        put_mid  = mid_price(put_contract)

        # Sell contracts
        call_ticket = self.limit_order(call_contract.symbol, -max_contracts, call_mid)
        put_ticket  = self.limit_order(put_contract.symbol,  -max_contracts, put_mid)
                
        # Use mid as expected fill if not filled yet (more realistic than assuming best-case)
        call_fill_price = call_ticket.average_fill_price if call_ticket.average_fill_price else call_mid
        put_fill_price  = put_ticket.average_fill_price if put_ticket.average_fill_price else put_mid

        total_credit = (call_fill_price + put_fill_price) * 100 * max_contracts
        
        # Store position details with unique key
        self._position_counter += 1
        position_key = f"pos_{self._position_counter}_{call_contract.expiry.date()}"
        self._open_positions[position_key] = {
            'call': call_contract.symbol,
            'put': put_contract.symbol,
            'expiry': call_contract.expiry,
            'opened': self.time,
            'quantity': max_contracts,
            'initial_quantity': max_contracts,
            'initial_credit': total_credit,
            'call_open_price': call_fill_price,
            'put_open_price': put_fill_price,
            'partial_profit_taken': False,
            'call_open_delta': abs(call_contract.greeks.delta),
            'put_open_delta': abs(put_contract.greeks.delta)
        }
        
        self.debug(f"Opened strangle {position_key}: {max_contracts} contracts, Call {call_contract.symbol} (delta: {call_contract.greeks.delta:.3f}), Put {put_contract.symbol} (delta: {put_contract.greeks.delta:.3f}), Credit: ${total_credit:.2f}, VIX: {current_vix:.2f}")
    
    def _manage_positions(self):
        """Manage positions with improved exit rules and partial profit taking"""
        positions_to_close = []
        
        for key, position in self._open_positions.items():
            days_to_expiry = (position['expiry'] - self.time).days
            
            call_symbol = position['call']
            put_symbol = position['put']
            quantity = position['quantity']
            
            # Get current holdings
            call_invested = self.portfolio[call_symbol].invested
            put_invested = self.portfolio[put_symbol].invested
            
            if not call_invested and not put_invested:
                positions_to_close.append(key)
                continue
            
            # For short options, realistic close cost is the ASK (you pay ask to buy-to-close)
            call_sec = self.securities[call_symbol]
            put_sec  = self.securities[put_symbol]

            call_price = call_sec.ask_price if call_invested and call_sec.ask_price > 0 else call_sec.price
            put_price  = put_sec.ask_price  if put_invested and put_sec.ask_price > 0 else put_sec.price

            # Calculate current P&L
            call_pnl = (position['call_open_price'] - call_price) * 100 * quantity if call_invested else 0
            put_pnl = (position['put_open_price'] - put_price) * 100 * quantity if put_invested else 0
            total_pnl = call_pnl + put_pnl
            
            close_reason = None
            partial_close = False
            
            # Rule 1: Time-based exit (21 days before expiration)
            if days_to_expiry <= 21:
                close_reason = f"Time exit - {days_to_expiry} days to expiry"
            
            # Rule 2: Partial profit taking at 25% (if not done yet)
            elif not position['partial_profit_taken'] and total_pnl >= position['initial_credit'] * self._partial_profit_target:
                partial_close = True
                close_reason = f"Partial profit - P&L: ${total_pnl:.2f}"
            
            # Rule 3: Full profit target at 50%
            elif total_pnl >= position['initial_credit'] * self._full_profit_target:
                close_reason = f"Full profit target - P&L: ${total_pnl:.2f}"
            
            # Rule 4: Total position stop loss (2.5x initial credit)
            elif total_pnl < -position['initial_credit'] * self._total_stop_loss_multiplier:
                close_reason = f"Total stop loss - P&L: ${total_pnl:.2f}"
            
            # Rule 5: Delta-based exit (position getting tested)
            elif call_invested:
                call_contract = self.securities[call_symbol]
                if hasattr(call_contract, 'greeks') and call_contract.greeks and call_contract.greeks.delta:
                    if abs(call_contract.greeks.delta) > self._max_delta_exit:
                        close_reason = f"Call delta breach - Delta: {abs(call_contract.greeks.delta):.3f}"
            
            if not close_reason and put_invested:
                put_contract = self.securities[put_symbol]
                if hasattr(put_contract, 'greeks') and put_contract.greeks and put_contract.greeks.delta:
                    if abs(put_contract.greeks.delta) > self._max_delta_exit:
                        close_reason = f"Put delta breach - Delta: {abs(put_contract.greeks.delta):.3f}"
            
            # Execute closes
            if close_reason:
                if partial_close:
                    # Close half the position
                    contracts_to_close = max(1, quantity // 2)
                    if call_invested:
                        self.market_order(call_symbol, contracts_to_close)
                    if put_invested:
                        self.market_order(put_symbol, contracts_to_close)
                    
                    position['quantity'] = quantity - contracts_to_close
                    position['partial_profit_taken'] = True
                    self.debug(f"Partial close {key}: {close_reason}, closed {contracts_to_close} of {quantity} contracts")
                else:
                    # Close entire position
                    if call_invested:
                        self.liquidate(call_symbol)
                    if put_invested:
                        self.liquidate(put_symbol)
                    
                    positions_to_close.append(key)
                    self.debug(f"Full close {key}: {close_reason}")
        
        # Remove fully closed positions
        for key in positions_to_close:
            del self._open_positions[key]

    def _get_atr_pct(self) -> float:
        price = self.securities[self._spy].price
        if self.is_warming_up or not self._atr.is_ready or price <= 0:
            return None
        return float(self._atr.current.value) / float(price)

