# region imports
from AlgorithmImports import *
import math
# endregion

class RiskManagedStrangleStrategy(QCAlgorithm):
    """
    NVDA short strangle + NVDA-specific volatility proxy (ATR%) + realistic fills/PnL
    + wheel-on-put-assignment (covered calls)

    Key upgrades included:
      ✅ NVDA ATR% regime filter + sizing multiplier
      ✅ Expiry-first 30–45 DTE selection (prevents mismatch/no-trade)
      ✅ Entry/exit with LIMIT @ mid (more realistic than market orders)
      ✅ PnL uses ASK for buy-to-close (realistic)
      ✅ Wheel it on PUT assignment (sell covered calls; no new strangles while wheeling)
    """

    def initialize(self):
        self.set_start_date(2025, 1, 1)
        self.set_cash(100000)
        self.settings.seed_initial_prices = True

        # -----------------------------
        # Underlying + options
        # -----------------------------
        self.nvda = self.add_equity("NVDA", Resolution.HOUR).symbol
        self.nvda_opt = self.add_option("NVDA", Resolution.HOUR).symbol
        self.securities[self.nvda_opt].set_filter(lambda u: u.strikes(-80, 80).expiration(25, 60))

        # Optional macro filter (kept; not primary for NVDA)
        self.vix = self.add_data(CBOE, "VIX", Resolution.DAILY).symbol
        self.max_vix = 20

        # -----------------------------
        # NVDA-specific volatility proxy: ATR%
        # -----------------------------
        self._atr = self.atr(self.nvda, 14, MovingAverageType.WILDERS, Resolution.DAILY)
        self.set_warm_up(30, Resolution.DAILY)

        # ATR% thresholds (tune)
        self.max_atr_pct_entry = 0.055   # skip new trades if ATR% > 5.5%
        self.atr_low_pct = 0.030         # <3.0% = calm
        self.atr_high_pct = 0.045        # >4.5% = elevated

        # -----------------------------
        # Strangle parameters
        # -----------------------------
        self.total_stop_loss_multiplier = 2.5   # close if total loss < -2.5x credit
        self.partial_profit_target = 0.25       # take half off at 25% of credit
        self.full_profit_target = 0.50          # close remainder at 50% of credit
        self.max_positions = 4

        self.target_delta = 0.10
        self.delta_tolerance = 0.03             # accept 0.07–0.13 delta

        # Sizing (naked NVDA: keep conservative)
        self.base_position_size_pct = 0.10      # used to compute contracts (capped)
        self.max_contracts_cap = 2              # hard cap for safety; raise only after confidence

        # Management
        self.time_exit_days = 21                # exit 21 DTE

        # Spread quality gate
        self.max_spread_pct_of_mid = 0.08       # 8% of mid

        # -----------------------------
        # Wheel state
        # -----------------------------
        self.wheel_active = False
        self.wheel_entry_price = None
        self.wheel_entry_date = None
        self.wheel_cc_symbol = None             # current covered call symbol (if any)
        self.max_wheel_days = 45                # exit wheel after 45 days if stuck

        # -----------------------------
        # Tracking
        # -----------------------------
        self.open_positions = {}                # key -> position dict
        self.position_counter = 0

        # -----------------------------
        # Tracking for rebalance (simpler than flags)
        # -----------------------------
        self.last_strangle_check = None
        self.last_cc_check = None

        # Manage daily
        self.schedule.on(self.date_rules.every_day(self.nvda),
                         self.time_rules.after_market_open(self.nvda, 35),
                         self._manage_positions)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _mid(self, bid: float, ask: float) -> float:
        if bid <= 0 or ask <= 0:
            return 0.0
        return round(0.5 * (bid + ask), 2)

    def _spread_ok(self, bid: float, ask: float) -> bool:
        mid = self._mid(bid, ask)
        if mid <= 0:
            return False
        return (ask - bid) <= mid * self.max_spread_pct_of_mid

    def _get_atr_pct(self):
        price = self.securities[self.nvda].price
        if self.is_warming_up or (not self._atr.is_ready) or price <= 0:
            return None
        return float(self._atr.current.value) / float(price)



    # -----------------------------
    # Main data loop
    # -----------------------------
    def on_data(self, data: Slice):
        # Check for option chain data
        if self.nvda_opt not in data.option_chains:
            return
            
        # Try to open strangle on Mondays
        if self.time.weekday() == 0:  # Monday
            if self.last_strangle_check != self.time.date():
                self.last_strangle_check = self.time.date()
                self._execute_strangle_open(data)
        
        # Try to sell covered call DAILY when wheeling (not just Mondays)
        # This ensures we don't miss opportunities due to data gaps
        if self.wheel_active:
            if self.last_cc_check != self.time.date():
                self.last_cc_check = self.time.date()
                self._execute_sell_covered_call(data)

    # -----------------------------
    # Entry: short strangle
    # -----------------------------
    def _execute_strangle_open(self, data: Slice):
        if self.is_warming_up:
            return

        # Don't open new strangles while in wheel mode
        if self.wheel_active:
            self.debug("Skipping strangle entry — wheel active")
            return

        if len(self.open_positions) >= self.max_positions:
            return
        
        # CRITICAL: Check buying power before opening new positions
        # Require at least 30% buffer to avoid margin calls
        buying_power = self.portfolio.total_portfolio_value - abs(self.portfolio.total_margin_used)
        if buying_power < self.portfolio.total_portfolio_value * 0.30:
            self.debug(f"Insufficient buying power: ${buying_power:.2f} (need 30% buffer)")
            return

        # Optional macro filter
        if self.vix in data and data[self.vix]:
            vix = float(data[self.vix].value)
            if vix > self.max_vix:
                self.debug(f"Skipping entry — VIX too high: {vix:.2f}")
                return

        # NVDA-specific volatility filter (ATR%)
        atr_pct = self._get_atr_pct()
        if atr_pct is None:
            self.debug("Skipping entry — ATR% not ready")
            return

        if atr_pct > self.max_atr_pct_entry:
            self.debug(f"Skipping entry — NVDA ATR% too high: {atr_pct*100:.2f}%")
            return

        # ATR%-based sizing multiplier
        if atr_pct < self.atr_low_pct:
            size_mult = 1.25
        elif atr_pct > self.atr_high_pct:
            size_mult = 0.65
        else:
            size_mult = 1.0

        adjusted_size_pct = self.base_position_size_pct * size_mult

        chain = data.option_chains.get(self.nvda_opt)
        if not chain:
            self.debug("No NVDA option chain in slice")
            return

        # ---- Expiry-first selection (30–45 DTE) ----
        dte_min, dte_max = 30, 45
        expiries = sorted({
            x.expiry for x in chain
            if dte_min <= (x.expiry - self.time).days <= dte_max
            and x.greeks and x.greeks.delta is not None
        })
        if not expiries:
            self.debug("No expiries 30–45 DTE with greeks")
            return

        expiry = expiries[0]

        calls = [x for x in chain
                 if x.expiry == expiry and x.right == OptionRight.CALL
                 and x.greeks and x.greeks.delta is not None
                 and x.bid_price > 0 and x.ask_price > 0]

        puts = [x for x in chain
                if x.expiry == expiry and x.right == OptionRight.PUT
                and x.greeks and x.greeks.delta is not None
                and x.bid_price > 0 and x.ask_price > 0]

        if not calls or not puts:
            self.debug("No calls/puts for selected expiry with valid bid/ask")
            return

        # Select closest to target delta within tolerance
        tgt = self.target_delta
        tol = self.delta_tolerance

        call = min(calls, key=lambda x: abs(abs(x.greeks.delta) - tgt))
        put  = min(puts,  key=lambda x: abs(abs(x.greeks.delta) - tgt))

        best_call = abs(call.greeks.delta)
        best_put  = abs(put.greeks.delta)

        self.debug(f"Entry scan exp={expiry.date()} callΔ={best_call:.3f} putΔ={best_put:.3f} ATR%={atr_pct*100:.2f}%")

        if abs(best_call - tgt) > tol:
            self.debug(f"No suitable CALL delta. Best={best_call:.3f}")
            return
        if abs(best_put - tgt) > tol:
            self.debug(f"No suitable PUT delta. Best={best_put:.3f}")
            return

        # Spread quality gate
        if not self._spread_ok(call.bid_price, call.ask_price) or not self._spread_ok(put.bid_price, put.ask_price):
            self.debug("Skipping entry — spreads too wide")
            return

        # Contracts sizing (conservative; premium proxy + hard cap)
        call_mid = self._mid(call.bid_price, call.ask_price)
        put_mid = self._mid(put.bid_price, put.ask_price)
        premium_mid = (call_mid + put_mid) * 100
        if premium_mid <= 0:
            return

        # Additional safety: estimate margin requirement (use strikes as proxy)
        estimated_margin_per_contract = max(call.strike, put.strike) * 100 * 0.20  # ~20% margin
        available_margin = self.portfolio.total_portfolio_value - abs(self.portfolio.total_margin_used)
        
        # Calculate max contracts based on margin AND budget
        budget = self.portfolio.total_portfolio_value * adjusted_size_pct
        qty_from_budget = max(1, int(budget / premium_mid))
        qty_from_margin = max(1, int(available_margin / estimated_margin_per_contract)) if estimated_margin_per_contract > 0 else 1
        
        qty = min(qty_from_budget, qty_from_margin, self.max_contracts_cap)
        
        self.debug(f"Position sizing: budget_qty={qty_from_budget}, margin_qty={qty_from_margin}, final_qty={qty}")

        # Enter with LIMIT @ mid (sell to open => -qty)
        call_limit = self._mid(call.bid_price, call.ask_price)
        put_limit  = self._mid(put.bid_price, put.ask_price)

        call_ticket = self.limit_order(call.symbol, -qty, call_limit)
        put_ticket  = self.limit_order(put.symbol,  -qty, put_limit)

        # Use limit price as entry estimate if average fill not available immediately
        call_entry = call_ticket.average_fill_price if call_ticket.average_fill_price else call_limit
        put_entry  = put_ticket.average_fill_price  if put_ticket.average_fill_price  else put_limit

        credit = (call_entry + put_entry) * 100 * qty

        # Track position
        self.position_counter += 1
        key = f"pos_{self.position_counter}_{expiry.date()}"

        self.open_positions[key] = {
            "call": call.symbol,
            "put": put.symbol,
            "expiry": expiry,
            "opened": self.time,
            "quantity": qty,
            "initial_quantity": qty,
            "initial_credit": credit,
            "call_open_price": call_entry,
            "put_open_price": put_entry,
            "partial_profit_taken": False,
            "call_strike": call.strike,
            "put_strike": put.strike
        }

        self.debug(f"OPEN {key}: qty={qty} credit=${credit:.2f} call={call.symbol} put={put.symbol}")

    # -----------------------------
    # Daily management: exits + assignment detection + wheel reset
    # -----------------------------
    def _manage_positions(self):
        # Always check wheel exit first (before assignment detection)
        self._check_wheel_exit()

        # Detect assignment / wheel transitions
        self._check_for_put_assignment()
        
        # DAILY covered-call cleanup (WHEEL)
        if self.wheel_active and self.wheel_cc_symbol:
            if not self.portfolio[self.wheel_cc_symbol].invested:
                self.debug(f"DAILY CC cleanup: {self.wheel_cc_symbol} expired/closed -> cleared")
                self.wheel_cc_symbol = None
       

        # If wheeling, do NOT manage strangles as if they still exist (but clean up if needed)
        # We'll still try to remove any dead tracked positions.
        positions_to_remove = []

        for key, pos in self.open_positions.items():
            call_sym = pos["call"]
            put_sym = pos["put"]

            call_inv = self.portfolio[call_sym].invested
            put_inv = self.portfolio[put_sym].invested

            # If both legs are gone, stop tracking
            if not call_inv and not put_inv:
                positions_to_remove.append(key)
                continue

            qty = pos["quantity"]
            dte = (pos["expiry"] - self.time).days

            # Realistic close cost for short options = ASK
            call_sec = self.securities[call_sym]
            put_sec  = self.securities[put_sym]

            call_close = call_sec.ask_price if (call_inv and call_sec.ask_price > 0) else call_sec.price
            put_close  = put_sec.ask_price  if (put_inv and put_sec.ask_price  > 0) else put_sec.price

            # PnL (credit received - cost to close)
            call_pnl = (pos["call_open_price"] - call_close) * 100 * qty if call_inv else 0
            put_pnl  = (pos["put_open_price"]  - put_close)  * 100 * qty if put_inv else 0
            total_pnl = call_pnl + put_pnl

            close_reason = None
            partial_close = False

            # Rule 1: Time exit
            if dte <= self.time_exit_days:
                close_reason = f"Time exit — {dte} DTE"

            # Rule 2: Partial profit
            elif (not pos["partial_profit_taken"]) and total_pnl >= pos["initial_credit"] * self.partial_profit_target:
                partial_close = True
                close_reason = f"Partial profit — PnL ${total_pnl:.2f}"

            # Rule 3: Full profit
            elif total_pnl >= pos["initial_credit"] * self.full_profit_target:
                close_reason = f"Full profit — PnL ${total_pnl:.2f}"

            # Rule 4: Total stop loss
            elif total_pnl < -pos["initial_credit"] * self.total_stop_loss_multiplier:
                close_reason = f"Stop loss — PnL ${total_pnl:.2f}"

            # Execute closes
            if close_reason:
                if partial_close:
                    close_qty = max(1, qty // 2)

                    # Buy to close with LIMIT @ mid (more realistic than market)
                    if call_inv:
                        mid_call = self._mid(call_sec.bid_price, call_sec.ask_price) or call_close
                        self.limit_order(call_sym, close_qty, mid_call)
                    if put_inv:
                        mid_put = self._mid(put_sec.bid_price, put_sec.ask_price) or put_close
                        self.limit_order(put_sym, close_qty, mid_put)

                    pos["quantity"] = qty - close_qty
                    pos["partial_profit_taken"] = True
                    self.debug(f"PARTIAL CLOSE {key}: {close_reason} closed={close_qty}/{qty}")

                else:
                    # Full close
                    if call_inv:
                        self.liquidate(call_sym)
                    if put_inv:
                        self.liquidate(put_sym)

                    positions_to_remove.append(key)
                    self.debug(f"FULL CLOSE {key}: {close_reason}")

        for k in positions_to_remove:
            if k in self.open_positions:
                del self.open_positions[k]

    # -----------------------------
    # Wheel logic
    # -----------------------------
    def _check_for_put_assignment(self):
        """
        If a short put is assigned, we'll see NVDA shares appear.
        We then:
          - close any remaining short call legs from strangles (simplify risk)
          - enter wheel mode
        
        CRITICAL: Also handle SHORT stock scenarios (negative shares from call assignments)
        """
        nvda_shares = int(self.portfolio[self.nvda].quantity)

        # Log current state
        self.debug(f"CHECK ASSIGNMENT - NVDA shares: {nvda_shares}, wheel_active: {self.wheel_active}")

        # EMERGENCY: Handle short stock position (call assigned without owning shares)
        if nvda_shares < 0:
            self.debug(f"EMERGENCY: SHORT STOCK DETECTED! shares={nvda_shares}")
            
            # Immediately close ALL short calls to prevent further assignments
            for key, pos in list(self.open_positions.items()):
                call_sym = pos["call"]
                if self.portfolio[call_sym].invested:
                    self.liquidate(call_sym)
                    self.debug(f"Closed short call {call_sym} due to short stock")
            
            # Close covered call if active
            if self.wheel_cc_symbol and self.portfolio[self.wheel_cc_symbol].invested:
                self.liquidate(self.wheel_cc_symbol)
                self.debug(f"Closed wheel CC {self.wheel_cc_symbol} due to short stock")
                self.wheel_cc_symbol = None
            
            # Buy back the short stock immediately
            self.market_order(self.nvda, abs(nvda_shares))
            self.debug(f"Bought back {abs(nvda_shares)} shares to cover short")
            
            # Exit wheel mode and clear all positions
            self.wheel_active = False
            self.wheel_entry_price = None
            self.wheel_entry_date = None
            self.open_positions.clear()
            return

        # Only trigger wheel when shares are actually owned
        if nvda_shares >= 100 and not self.wheel_active:
            self.debug(f"PUT ASSIGNED (or acquired shares): NVDA shares={nvda_shares}")

            # Close remaining short calls from tracked strangles and stop tracking those positions
            for key, pos in list(self.open_positions.items()):
                call_sym = pos["call"]
                if self.portfolio[call_sym].invested:
                    self.liquidate(call_sym)
                # We intentionally do NOT liquidate the put here (it may already be assigned/closed)
                del self.open_positions[key]

            self.wheel_active = True
            self.wheel_entry_price = self.securities[self.nvda].price
            self.wheel_entry_date = self.time
            self.wheel_cc_symbol = None

            self.debug(f"WHEEL ON: shares={nvda_shares} entry~{self.wheel_entry_price:.2f} date={self.time.date()}")

    def _check_wheel_exit(self):
        """
        Exit wheel mode when:
        1. Shares = 0 (covered call assigned)
        2. Shares < 100 (partial assignment)
        3. Stuck in wheel > max_wheel_days (emergency exit)
        4. Stop loss hit (down >20% from entry)
        """
        if not self.wheel_active:
            return

        nvda_shares = int(self.portfolio[self.nvda].quantity)
        current_price = self.securities[self.nvda].price
        
        if nvda_shares == 0:
            self.debug("WHEEL OFF: shares called away (or sold). Resetting wheel state.")
            self.wheel_active = False
            self.wheel_entry_price = None
            self.wheel_entry_date = None
            self.wheel_cc_symbol = None
            return
            
        if nvda_shares < 100:
            self.debug(f"WHEEL PARTIAL SHARES: {nvda_shares} < 100, liquidating and exiting wheel")
            self.liquidate(self.nvda)
            self.wheel_active = False
            self.wheel_entry_price = None
            self.wheel_entry_date = None
            self.wheel_cc_symbol = None
            return
        
        # Emergency exit if stuck too long
        days_in_wheel = (self.time - self.wheel_entry_date).days
        if days_in_wheel > self.max_wheel_days:
            self.debug(f"WHEEL TIMEOUT: {days_in_wheel} days > {self.max_wheel_days}, force exiting")
            if self.wheel_cc_symbol and self.portfolio[self.wheel_cc_symbol].invested:
                self.liquidate(self.wheel_cc_symbol)
            self.liquidate(self.nvda)
            self.wheel_active = False
            self.wheel_entry_price = None
            self.wheel_entry_date = None
            self.wheel_cc_symbol = None
            return
            
        # Stop loss on shares (down >20%)
        if self.wheel_entry_price and current_price < self.wheel_entry_price * 0.80:
            self.debug(f"WHEEL STOP LOSS: price ${current_price:.2f} < entry ${self.wheel_entry_price:.2f} * 0.80")
            if self.wheel_cc_symbol and self.portfolio[self.wheel_cc_symbol].invested:
                self.liquidate(self.wheel_cc_symbol)
            self.liquidate(self.nvda)
            self.wheel_active = False
            self.wheel_entry_price = None
            self.wheel_entry_date = None
            self.wheel_cc_symbol = None
            return

    # ✅ Add this small cleanup block near the top of _execute_sell_covered_call()
    # (right after the nvda_shares checks, BEFORE you return for an existing CC)
    def _execute_sell_covered_call(self, data: Slice):
        """
        Sell covered calls when wheeling.
        Rules:
        - DTE 7–21
        - Delta ~ 0.25
        - Limit @ mid
        - One CC at a time
        - ✅ Cleanup expired/closed CC so we can sell the next one
        - Only sell CCs for shares we actually own (avoid naked calls)
        """
        if not self.wheel_active:
            return

        nvda_shares = int(self.portfolio[self.nvda].quantity)
        
        # Safety: don't sell calls if we don't have enough shares or are short
        if nvda_shares <= 0:
            self.debug(f"Cannot sell CC: shares={nvda_shares}")
            return
            
        if nvda_shares < 100:
            self.debug(f"Insufficient shares for CC: {nvda_shares} < 100")
            return

        # ✅ CC-expired / closed cleanup:
        # If we have a stored CC symbol but it's no longer invested (expired, assigned, or manually closed),
        # clear it so the algo can sell the next covered call.
        if self.wheel_cc_symbol and not self.portfolio[self.wheel_cc_symbol].invested:
            self.debug(f"CC cleanup: {self.wheel_cc_symbol} not invested anymore -> clearing")
            self.wheel_cc_symbol = None

        # If we already have a CC open, do nothing
        if self.wheel_cc_symbol and self.portfolio[self.wheel_cc_symbol].invested:
            return

        chain = data.option_chains.get(self.nvda_opt)
        if not chain:
            self.debug("No NVDA option chain for covered call")
            return

        # Covered call selection - use wider DTE range for more flexibility
        calls = [
            x for x in chain
            if x.right == OptionRight.CALL
            and 7 <= (x.expiry - self.time).days <= 30  # Widened from 21 to 30
            and x.greeks and x.greeks.delta is not None
            and x.bid_price > 0 and x.ask_price > 0
        ]
        if not calls:
            self.debug(f"No suitable calls found for CC. Chain has {len(list(chain))} contracts")
            return

        spot = float(self.securities[self.nvda].price)
        
        # Prefer OTM strikes above current price
        otm_calls = [c for c in calls if c.strike >= spot]
        
        if otm_calls:
            # Choose ~0.25 delta call from OTM options
            call = min(otm_calls, key=lambda x: abs(x.greeks.delta - 0.25))
            self.debug(f"Selected OTM CC candidate: strike={call.strike}, delta={call.greeks.delta:.3f}, DTE={((call.expiry - self.time).days)}")
        else:
            # If no OTM options, use closest to 0.25 delta (emergency)
            call = min(calls, key=lambda x: abs(x.greeks.delta - 0.25))
            self.debug(f"WARNING: No OTM calls, using ATM/ITM: strike={call.strike}, spot={spot:.2f}, delta={call.greeks.delta:.3f}")

        if not self._spread_ok(call.bid_price, call.ask_price):
            spread_pct = ((call.ask_price - call.bid_price) / self._mid(call.bid_price, call.ask_price)) * 100
            self.debug(f"Skipping covered call — spread too wide: {spread_pct:.1f}% (max {self.max_spread_pct_of_mid*100:.1f}%)")
            return

        # Sell covered calls for ALL assigned shares
        contracts = nvda_shares // 100
        
        if contracts < 1:
            self.debug(f"Not enough shares for CC: {nvda_shares} shares = {contracts} contracts")
            return
        
        limit_px = self._mid(call.bid_price, call.ask_price)
        if limit_px <= 0:
            self.debug("Skipping covered call — invalid limit price")
            return

        self.limit_order(call.symbol, -contracts, limit_px)
        self.wheel_cc_symbol = call.symbol

        self.debug(f"SELL CC: {call.symbol} qty={contracts} (ALL shares) Δ={call.greeks.delta:.2f} strike={call.strike} spot={spot:.2f} DTE={((call.expiry - self.time).days)} premium=${limit_px:.2f}")
