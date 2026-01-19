# region imports
from AlgorithmImports import *
from datetime import timedelta
import math

# endregion


class RiskManagedStrangleStrategy(QCAlgorithm):
    """
    NVDA short strangle + ATR% regime filter + wheel-on-put-assignment (covered calls)
    + lightweight logging + CSV/ObjectStore P&L ledgers.

    Logging philosophy (reduced noise):
    - Only log *state changes* and *important events* (OPEN/CLOSE/ASSIGN/WHEEL/CC).
    - Only log one DAILY summary line (once per day).
    - Keep per-position snapshots out of Debug by default (they still go to CSV).
    - Keep ObjectStore flush logs quiet (only log errors or end-of-algorithm).

    CSV outputs (ObjectStore):
      - pnl_daily_ledger.csv   : daily snapshot
      - pnl_trade_ledger.csv   : event ledger (opens/closes/wheel/cc)
    """

    # -----------------------------
    # Init
    # -----------------------------
    def initialize(self):
        self.set_start_date(2025, 1, 1)
        self.set_cash(10000)
        self.settings.seed_initial_prices = True

        # === Log switches (flip to True when actively debugging) ===
        self.log_fills = False           # fill-by-fill logs (noisy)
        self.log_daily_summary = True    # one line per day
        self.log_per_position_debug = False  # per-strangle debug snapshots (noisy)

        # -----------------------------
        # Underlying + options
        # -----------------------------
        self.nvda = self.add_equity("NVDA", Resolution.HOUR).symbol
        self.nvda_opt = self.add_option("NVDA", Resolution.HOUR).symbol
        self.securities[self.nvda_opt].set_filter(lambda u: u.strikes(-80, 80).expiration(25, 60))

        # Optional macro filter (VIX is a blunt tool for NVDA, but kept as safety gate)
        self.vix = self.add_data(CBOE, "VIX", Resolution.DAILY).symbol
        self.max_vix = 20

        # -----------------------------
        # NVDA volatility proxy: ATR%
        # -----------------------------
        self._atr = self.atr(self.nvda, 14, MovingAverageType.WILDERS, Resolution.DAILY)
        self.set_warm_up(30, Resolution.DAILY)

        self.max_atr_pct_entry = 0.055   # block entries if ATR% > 5.5%
        self.atr_low_pct = 0.030         # calm regime
        self.atr_high_pct = 0.045        # elevated regime (reduce size)

        # -----------------------------
        # Strangle parameters (naked = tail risk; keep conservative)
        # -----------------------------
        self.total_stop_loss_multiplier = 2.5   # loss threshold as multiple of credit
        self.partial_profit_target = 0.25       # close half at 25% of credit
        self.full_profit_target = 0.50          # close rest at 50% of credit
        self.max_positions = 4                  # overall cap

        self.target_delta = 0.10
        self.delta_tolerance = 0.03             # accept 0.07–0.13

        self.base_position_size_pct = 0.10
        self.max_contracts_cap = 2              # hard cap for safety

        self.time_exit_days = 21                # exit around 21 DTE
        self.max_spread_pct_of_mid = 0.08       # bid-ask quality gate

        # -----------------------------
        # Wheel state (shares acquired via put assignment)
        # -----------------------------
        self.wheel_active = False
        self.wheel_entry_price = None
        self.wheel_entry_date = None
        self.wheel_cc_symbol = None
        self.max_wheel_days = 45                # emergency timeout

        # -----------------------------
        # Position tracking
        # -----------------------------
        self.open_positions = {}                # pos_key -> dict
        self.position_counter = 0
        # --- Order tagging + fill attribution ---
        self._order_context = {}     # order_id -> dict(meta)
        self._last_holdings_profit = {}  # symbol -> last seen holdings.profit (to estimate realized at close)


        # Throttle entry attempts
        self.last_strangle_check = None
        self.last_cc_check = None

        # -----------------------------
        # Lightweight P&L buckets (for quick debugging)
        # -----------------------------
        self.pnl = {"options_realized": 0.0, "stock_realized": 0.0}
        self.stock_state = {"last_profit": 0.0}   # to bucket deltas in holdings.profit

        # Tags for options (used only if log_fills=True)
        self.option_tags = {}  # contract Symbol -> "STRANGLE_CALL" / "STRANGLE_PUT" / "COVERED_CALL"

        # -----------------------------
        # Daily snapshot throttling
        # -----------------------------
        self._log_last_daily_date = None

        # -----------------------------
        # ObjectStore CSV ledgers
        # -----------------------------
        self.os_daily_key = "pnl_daily_ledger.csv"
        self.os_trade_key = "pnl_trade_ledger.csv"

        self._daily_rows = []
        self._trade_rows = []

        self._daily_header = (
            "date,time,equity,cash,margin_used,spot,shares,avg_price,"
            "opt_unrl,stk_unrl,opt_real,stk_real,open_strangles,wheel_active,wheel_cc\n"
        )
        self._trade_header = (
            "date,time,event,key,detail,qty,price,credit_debit,est_pnl,equity,spot,shares\n"
        )

        # --- Prevent-only call assignment protection ---
        self.prevent_assignment = True
        self.call_delta_threat = 0.65       # close when call delta >= 0.65
        self.itm_buffer_pct = 0.01          # close when spot >= strike*(1+1%)
        self.assignment_risk_dte = 7        # close if ITM and <= 7 DTE
        self.cooldown_days_after_threat = 3 # optional cooldown after forced threat-exit
        self._cooldown_until = None


        self._ensure_objectstore_csv(self.os_daily_key, self._daily_header)
        self._ensure_objectstore_csv(self.os_trade_key, self._trade_header)

        # -----------------------------
        # Scheduling
        # -----------------------------
        # Daily management (exits/assignment/wheel/cleanup) + daily snapshot row
        self.schedule.on(self.date_rules.every_day(self.nvda),
                         self.time_rules.after_market_open(self.nvda, 35),
                         self._manage_positions)

        # Flush CSV buffers at EOD
        self.schedule.on(self.date_rules.every_day(self.nvda),
                         self.time_rules.before_market_close(self.nvda, 1),
                         self._flush_ledgers_to_objectstore)

    # -----------------------------
    # Helpers (pricing + filters)
    # -----------------------------
    def _mid(self, bid: float, ask: float) -> float:
        """Mid price with simple rounding; returns 0 if invalid."""
        if bid <= 0 or ask <= 0:
            return 0.0
        return round(0.5 * (bid + ask), 2)

    def _spread_ok(self, bid: float, ask: float) -> bool:
        """Reject contracts with a bid-ask spread too large relative to mid."""
        mid = self._mid(bid, ask)
        if mid <= 0:
            return False
        return (ask - bid) <= mid * self.max_spread_pct_of_mid

    def _get_atr_pct(self):
        """ATR% = ATR(14) / price. Returns None during warmup or if ATR not ready."""
        price = self.securities[self.nvda].price
        if self.is_warming_up or (not self._atr.is_ready) or price <= 0:
            return None
        return float(self._atr.current.value) / float(price)

    def _fmt_money(self, x: float) -> str:
        return f"${x:,.2f}"

    # -----------------------------
    # P&L computations
    # -----------------------------
    def _compute_options_unrealized(self) -> float:
        """Sum QC unrealized P&L for all invested option holdings."""
        total = 0.0
        for kvp in self.portfolio:
            sec = kvp.key
            hold = kvp.value
            if hold.invested and sec.security_type == SecurityType.OPTION:
                total += float(hold.unrealized_profit)
        return total

    def _compute_stock_unrealized(self) -> float:
        """QC unrealized P&L for NVDA shares (0 if no shares)."""
        hold = self.portfolio[self.nvda]
        return float(hold.unrealized_profit) if hold.invested else 0.0

    def _in_cooldown(self) -> bool:
        return self._cooldown_until is not None and self.time.date() < self._cooldown_until

    def _make_tag(self, *, action: str, reason: str, key: str = "", leg: str = "", extra: str = "") -> str:
        """
        Compact tag for QC order tickets (shows in Orders tab + logs).
        Keep it short but information-dense.
        """
        parts = [action]
        if reason: parts.append(reason)
        if key: parts.append(key)
        if leg: parts.append(leg)
        if extra: parts.append(extra)
        # Use | separators (easy to parse)
        return "|".join(parts)[:180]  # QC tags have practical length limits

    def _track_ticket(self, ticket, *, action: str, reason: str, key: str, leg: str,
                    est_pnl: float = 0.0, note: str = ""):
        """
        Persist metadata so OnOrderEvent can attribute fills + log to CSV.
        """
        if ticket is None:
            return
        self._order_context[int(ticket.order_id)] = {
            "action": action,
            "reason": reason,
            "key": key,
            "leg": leg,
            "est_pnl": float(est_pnl),
            "note": note
        }

    def _place_limit(self, symbol, qty, limit_price, *, action: str, reason: str, key: str, leg: str,
                    est_pnl: float = 0.0, note: str = ""):
        """
        Wrapper around LimitOrder with a rich tag + context tracking.
        """
        tag = self._make_tag(action=action, reason=reason, key=key, leg=leg, extra=note)
        ticket = self.limit_order(symbol, qty, limit_price, tag=tag)
        self._track_ticket(ticket, action=action, reason=reason, key=key, leg=leg, est_pnl=est_pnl, note=note)
        return ticket

    def _place_market(self, symbol, qty, *, action: str, reason: str, key: str, leg: str,
                    est_pnl: float = 0.0, note: str = ""):
        """
        Wrapper around MarketOrder with a rich tag + context tracking.
        """
        tag = self._make_tag(action=action, reason=reason, key=key, leg=leg, extra=note)
        ticket = self.market_order(symbol, qty, tag=tag)
        self._track_ticket(ticket, action=action, reason=reason, key=key, leg=leg, est_pnl=est_pnl, note=note)
        return ticket

    def _place_liquidate(self, symbol, *, action: str, reason: str, key: str, leg: str,
                        est_pnl: float = 0.0, note: str = ""):
        """
        Liquidate does not let us tag the underlying market order directly.
        We'll still call Liquidate and record a 'synthetic' event in the ledger.
        """
        self._append_trade_row(
            event="LIQUIDATE_INTENT",
            key=key,
            detail=f"{action}|{reason}|{leg}|{note}|sym={symbol}",
            qty=0,
            price=0,
            credit_debit=0,
            est_pnl=est_pnl
        )
        self.liquidate(symbol)

    # -----------------------------
    # Daily snapshot (minimal debug + always to CSV)
    # -----------------------------
    def _log_daily_snapshot(self):
        """
        One-per-day snapshot:
        - Debug: one line (optional)
        - CSV: always written
        """
        today = self.time.date()
        if self._log_last_daily_date == today:
            return
        self._log_last_daily_date = today

        equity = float(self.portfolio.total_portfolio_value)
        cash = float(self.portfolio.cash)
        margin_used = float(self.portfolio.total_margin_used)

        opt_unrl = float(self._compute_options_unrealized())
        stk_unrl = float(self._compute_stock_unrealized())

        shares = int(self.portfolio[self.nvda].quantity)
        avg_price = float(self.portfolio[self.nvda].average_price) if shares != 0 else 0.0
        spot = float(self.securities[self.nvda].price)

        if self.log_daily_summary:
            self.debug(
                f"[DAILY] {today} eq={self._fmt_money(equity)} "
                f"optU={self._fmt_money(opt_unrl)} stkU={self._fmt_money(stk_unrl)} "
                f"optR={self._fmt_money(self.pnl['options_realized'])} stkR={self._fmt_money(self.pnl['stock_realized'])} "
                f"spot={spot:.2f} sh={shares} openStr={len(self.open_positions)} wheel={self.wheel_active}"
            )

        self._append_daily_row()

    def _log_position_snapshot(self, key: str, pos: dict, reason: str = "SNAP"):
        """
        Optional debug-only per-position snapshot (no CSV dependency).
        Use only when log_per_position_debug=True to avoid spam.
        """
        if not self.log_per_position_debug:
            return

        call_sym = pos["call"]
        put_sym = pos["put"]
        qty = pos["quantity"]
        expiry = pos["expiry"].date()

        call_inv = self.portfolio[call_sym].invested
        put_inv = self.portfolio[put_sym].invested

        call_sec = self.securities[call_sym]
        put_sec = self.securities[put_sym]

        # For short options, assume buy-to-close happens near ASK (more realistic than last)
        call_close = call_sec.ask_price if (call_inv and call_sec.ask_price > 0) else call_sec.price
        put_close = put_sec.ask_price if (put_inv and put_sec.ask_price > 0) else put_sec.price

        call_pnl = (pos["call_open_price"] - call_close) * 100 * qty if call_inv else 0.0
        put_pnl = (pos["put_open_price"] - put_close) * 100 * qty if put_inv else 0.0
        total = call_pnl + put_pnl

        dte = (pos["expiry"] - self.time).days
        spot = float(self.securities[self.nvda].price)

        self.debug(
            f"[{reason}] {key} exp={expiry} DTE={dte} spot={spot:.2f} "
            f"PnL={self._fmt_money(total)} call={self._fmt_money(call_pnl)} put={self._fmt_money(put_pnl)}"
        )

    # -----------------------------
    # Main data loop (low noise)
    # -----------------------------
    def on_data(self, data: Slice):
        # Need option chain to do anything (entry or CC selection)
        if self.nvda_opt not in data.option_chains:
            return

        # Weekly strangle entry attempt (Monday, once per day)
        if self.time.weekday() == 0 and self.last_strangle_check != self.time.date():
            self.last_strangle_check = self.time.date()
            self._execute_strangle_open(data)

        # Covered call check daily while wheel is active (once per day)
        if self.wheel_active and self.last_cc_check != self.time.date():
            self.last_cc_check = self.time.date()
            self._execute_sell_covered_call(data)

    # -----------------------------
    # Fill logging (reduced)
    # -----------------------------
    def on_order_event(self, order_event: OrderEvent):
        if order_event.status != OrderStatus.FILLED:
            return

        oid = int(order_event.order_id)
        symbol = order_event.symbol
        qty = float(order_event.fill_quantity)
        px = float(order_event.fill_price)

        ctx = self._order_context.get(oid, None)

        # Try to compute "realized delta" when a holding fully closes (best-effort)
        # For options, realized is meaningful when hold.invested becomes False
        realized_delta = 0.0
        hold = self.portfolio[symbol]
        prev_profit = float(self._last_holdings_profit.get(symbol, 0.0))
        cur_profit = float(hold.profit)
        self._last_holdings_profit[symbol] = cur_profit

        # If this fill closes the holding (or reduces it), profit delta is still noisy intraday,
        # but is useful as an "accounting trace" when paired with action/reason.
        realized_delta = cur_profit - prev_profit

        # Light debug (only if enabled)
        if self.log_fills:
            if ctx:
                self.debug(f"[FILL] {ctx['action']} {ctx['reason']} {ctx['key']} {ctx['leg']} "
                        f"{symbol} qty={qty:+.0f} px={px:.2f} profΔ={realized_delta:+.2f}")
            else:
                self.debug(f"[FILL] {symbol} qty={qty:+.0f} px={px:.2f} profΔ={realized_delta:+.2f}")

        # CSV fill row (always)
        if ctx:
            detail = f"{ctx['action']}|{ctx['reason']}|{ctx['leg']}|{ctx.get('note','')}"
            self._append_trade_row(
                event="FILL",
                key=ctx.get("key", ""),
                detail=f"{detail}|oid={oid}|sym={symbol}",
                qty=qty,
                price=px,
                credit_debit=0.0,
                est_pnl=ctx.get("est_pnl", 0.0)
            )
        else:
            self._append_trade_row(
                event="FILL",
                key="",
                detail=f"NOCTX|oid={oid}|sym={symbol}",
                qty=qty,
                price=px,
                credit_debit=0.0,
                est_pnl=0.0
            )

        # Maintain your options_realized bucket (only when leg fully closes)
        if symbol.security_type == SecurityType.OPTION:
            if not hold.invested:
                self.pnl["options_realized"] += float(hold.profit)


    # -----------------------------
    # Entry: short strangle
    # -----------------------------
    def _execute_strangle_open(self, data: Slice):
        if self.is_warming_up:
            return

        if self._in_cooldown():
            self.debug(f"Skipping entry — cooldown until {self._cooldown_until}")
            return


        # No new strangles while wheeling (risk stacking)
        if self.wheel_active:
            return

        # Strategy-level position cap
        if len(self.open_positions) >= self.max_positions:
            return

        # Buying power buffer to reduce margin shocks
        buying_power = self.portfolio.total_portfolio_value - abs(self.portfolio.total_margin_used)
        if buying_power < self.portfolio.total_portfolio_value * 0.30:
            return

        # VIX gate (macro safety)
        if self.vix in data and data[self.vix]:
            vix = float(data[self.vix].value)
            if vix > self.max_vix:
                return

        # ATR% gate (NVDA-specific volatility regime)
        atr_pct = self._get_atr_pct()
        if atr_pct is None or atr_pct > self.max_atr_pct_entry:
            return

        # Size adjustment: larger in calm regime, smaller in elevated regime
        if atr_pct < self.atr_low_pct:
            size_mult = 1.25
        elif atr_pct > self.atr_high_pct:
            size_mult = 0.65
        else:
            size_mult = 1.0
        adjusted_size_pct = self.base_position_size_pct * size_mult

        chain = data.option_chains.get(self.nvda_opt)
        if not chain:
            return

        # Pick the earliest expiry in 30–45 DTE range with available greeks
        dte_min, dte_max = 30, 45
        expiries = sorted({
            x.expiry for x in chain
            if dte_min <= (x.expiry - self.time).days <= dte_max
            and x.greeks and x.greeks.delta is not None
        })
        if not expiries:
            return
        expiry = expiries[0]

        # Candidate contracts (require bid/ask to avoid bogus pricing)
        calls = [x for x in chain if x.expiry == expiry and x.right == OptionRight.CALL
                 and x.greeks and x.greeks.delta is not None and x.bid_price > 0 and x.ask_price > 0]
        puts = [x for x in chain if x.expiry == expiry and x.right == OptionRight.PUT
                and x.greeks and x.greeks.delta is not None and x.bid_price > 0 and x.ask_price > 0]
        if not calls or not puts:
            return

        # Select closest to target delta within tolerance
        tgt, tol = self.target_delta, self.delta_tolerance
        call = min(calls, key=lambda x: abs(abs(x.greeks.delta) - tgt))
        put = min(puts, key=lambda x: abs(abs(x.greeks.delta) - tgt))

        if abs(abs(call.greeks.delta) - tgt) > tol or abs(abs(put.greeks.delta) - tgt) > tol:
            return

        # Spread quality gate
        if not self._spread_ok(call.bid_price, call.ask_price) or not self._spread_ok(put.bid_price, put.ask_price):
            return

        call_mid = self._mid(call.bid_price, call.ask_price)
        put_mid = self._mid(put.bid_price, put.ask_price)
        premium_mid = (call_mid + put_mid) * 100
        if premium_mid <= 0:
            return

        # Simple margin proxy + budget sizing (still conservative due to max_contracts_cap)
        estimated_margin_per_contract = max(call.strike, put.strike) * 100 * 0.20
        available_margin = self.portfolio.total_portfolio_value - abs(self.portfolio.total_margin_used)

        budget = self.portfolio.total_portfolio_value * adjusted_size_pct
        qty_from_budget = max(1, int(budget / premium_mid))
        qty_from_margin = max(1, int(available_margin / estimated_margin_per_contract)) if estimated_margin_per_contract > 0 else 1
        qty = min(qty_from_budget, qty_from_margin, self.max_contracts_cap)

        self.position_counter += 1
        key = f"pos_{self.position_counter}_{expiry.date()}"

        # Place entry with LIMIT @ mid (less optimistic than market fills)
        # call_ticket = self.limit_order(call.symbol, -qty, call_mid)
        # put_ticket = self.limit_order(put.symbol, -qty, put_mid)

        call_ticket = self._place_limit(
            call.symbol, -qty, call_mid,
            action="OPEN", reason="STRANGLE", key=key, leg="STRANGLE_CALL",
            est_pnl=0.0,
            note=f"exp={expiry.date()} strike={call.strike} Δ={call.greeks.delta:.3f}"
        )
        put_ticket = self._place_limit(
            put.symbol, -qty, put_mid,
            action="OPEN", reason="STRANGLE", key=key, leg="STRANGLE_PUT",
            est_pnl=0.0,
            note=f"exp={expiry.date()} strike={put.strike} Δ={put.greeks.delta:.3f}"
        )

        # Tag legs (only used if log_fills=True)
        self.option_tags[call.symbol] = "STRANGLE_CALL"
        self.option_tags[put.symbol] = "STRANGLE_PUT"

        # Use limit price as entry estimate if avg fill not yet available
        call_entry = call_ticket.average_fill_price if call_ticket.average_fill_price else call_mid
        put_entry = put_ticket.average_fill_price if put_ticket.average_fill_price else put_mid
        credit = (call_entry + put_entry) * 100 * qty


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

        # Event ledger + one debug line (not noisy)
        self._append_trade_row(
            event="OPEN_STRANGLE",
            key=key,
            detail=f"exp={expiry.date()} call={call.strike} put={put.strike} Δc={call.greeks.delta:.3f} Δp={put.greeks.delta:.3f} ATR%={atr_pct*100:.2f}",
            qty=qty,
            credit_debit=credit
        )
        self.debug(f"OPEN {key}: qty={qty} credit={self._fmt_money(credit)} exp={expiry.date()}")

        # Optional debug snapshot
        self._log_position_snapshot(key, self.open_positions[key], reason="OPEN")

    # -----------------------------
    # Daily management: exits + assignment + wheel state
    # -----------------------------

    def _manage_positions(self):
        # Always write one daily snapshot row (throttled to once/day)
        self._log_daily_snapshot()

        # Wheel transitions
        self._check_wheel_exit()
        self._check_for_put_assignment()

        # Covered call housekeeping (wheel)
        if self.wheel_active and self.wheel_cc_symbol and not self.portfolio[self.wheel_cc_symbol].invested:
            self.wheel_cc_symbol = None

        to_remove = []

        for key, pos in list(self.open_positions.items()):
            call_sym, put_sym = pos["call"], pos["put"]
            call_inv = self.portfolio[call_sym].invested
            put_inv = self.portfolio[put_sym].invested

            if not call_inv and not put_inv:
                to_remove.append(key)
                continue

            qty = int(pos["quantity"])
            dte = int((pos["expiry"] - self.time).days)

            call_sec = self.securities[call_sym]
            put_sec  = self.securities[put_sym]
            spot = float(self.securities[self.nvda].price)

            # Compute close prices & PnL EARLY so any branch can use it
            call_close = call_sec.ask_price if (call_inv and call_sec.ask_price > 0) else call_sec.price
            put_close  = put_sec.ask_price  if (put_inv  and put_sec.ask_price  > 0) else put_sec.price

            call_pnl = (pos["call_open_price"] - call_close) * 100 * qty if call_inv else 0.0
            put_pnl  = (pos["put_open_price"]  - put_close)  * 100 * qty if put_inv else 0.0
            total_pnl = call_pnl + put_pnl

            # Optional debug snapshot (only if enabled)
            if self.log_per_position_debug:
                self._log_position_snapshot(key, pos, reason="MANAGE")

            # --- Prevent-only: exit before call assignment risk ---
            if self.prevent_assignment and call_inv:
                call_strike = float(pos.get("call_strike", 0.0)) or float(call_sec.symbol.id.strike_price)

                call_delta = None
                try:
                    if call_sec.greeks and call_sec.greeks.delta is not None:
                        call_delta = abs(float(call_sec.greeks.delta))
                except:
                    call_delta = None

                is_itm = (call_strike > 0 and spot > call_strike)
                itm_buffer_hit = (call_strike > 0 and spot >= call_strike * (1.0 + self.itm_buffer_pct))
                delta_threat = (call_delta is not None and call_delta >= self.call_delta_threat)
                expiry_threat = (is_itm and dte <= self.assignment_risk_dte)

                if delta_threat or expiry_threat or itm_buffer_hit:
                    parts = []
                    if delta_threat:
                        parts.append(f"Δ={call_delta:.2f}>= {self.call_delta_threat:.2f}")
                    if expiry_threat:
                        parts.append(f"ITM & DTE={dte}<= {self.assignment_risk_dte}")
                    if itm_buffer_hit:
                        parts.append(f"spot {spot:.2f} >= strike {call_strike:.2f}*(1+{self.itm_buffer_pct:.2%})")
                    close_reason = "ASSIGNMENT_RISK: " + " | ".join(parts)

                    self._close_strangle_explicit(
                        key, pos,
                        action="FORCE_CLOSE",
                        reason="ASSIGNMENT_RISK",
                        est_pnl=total_pnl,
                        note=close_reason
                    )

                    to_remove.append(key)

                    self._cooldown_until = self.time.date() + timedelta(days=self.cooldown_days_after_threat)

                    self._append_trade_row(
                        event="FORCE_CLOSE_ASSIGNMENT_RISK",
                        key=key,
                        detail=close_reason,
                        qty=qty,
                        credit_debit=0.0,
                        est_pnl=total_pnl
                    )
                    self.debug(f"FORCE CLOSE {key}: {close_reason} estPnL={self._fmt_money(total_pnl)} cooldown={self._cooldown_until}")
                    continue

            # Exit rules (normal)
            close_reason = None
            partial_close = False

            if dte <= self.time_exit_days:
                close_reason = f"TIME_EXIT {dte}DTE"
            elif (not pos["partial_profit_taken"]) and total_pnl >= pos["initial_credit"] * self.partial_profit_target:
                partial_close = True
                close_reason = f"PARTIAL_TP pnl={self._fmt_money(total_pnl)}"
            elif total_pnl >= pos["initial_credit"] * self.full_profit_target:
                close_reason = f"FULL_TP pnl={self._fmt_money(total_pnl)}"
            elif total_pnl < -pos["initial_credit"] * self.total_stop_loss_multiplier:
                close_reason = f"STOP pnl={self._fmt_money(total_pnl)}"

            if not close_reason:
                continue

            if partial_close:
                close_qty = max(1, qty // 2)
                if call_inv:
                    # self.limit_order(call_sym, close_qty, self._mid(call_sec.bid_price, call_sec.ask_price) or call_close)
                    self._place_limit(
                        call_sym, close_qty, self._mid(call_sec.bid_price, call_sec.ask_price) or call_close,
                        action="CLOSE", reason="PARTIAL_TP", key=key, leg="STRANGLE_CALL",
                        est_pnl=total_pnl
                    )

                if put_inv:
                    # self.limit_order(put_sym, close_qty, self._mid(put_sec.bid_price, put_sec.ask_price) or put_close)

                    self._place_limit(
                        put_sym, close_qty, self._mid(put_sec.bid_price, put_sec.ask_price) or put_close,
                        action="CLOSE", reason="PARTIAL_TP", key=key, leg="STRANGLE_PUT",
                        est_pnl=total_pnl
                    )


                pos["quantity"] = qty - close_qty
                pos["partial_profit_taken"] = True

                self._append_trade_row(event="PARTIAL_CLOSE", key=key, detail=close_reason, qty=close_qty, est_pnl=total_pnl)
                self.debug(f"PARTIAL {key}: {close_reason} closed={close_qty}/{qty}")

            else:
                self._close_strangle_explicit(
                    key, pos,
                    action="CLOSE",
                    reason=("FULL_TP" or "TIME_EXIT" or "STOP"),   # pass the rule name
                    est_pnl=total_pnl,
                    note=close_reason
                )

                to_remove.append(key)
                self._append_trade_row(event="FULL_CLOSE", key=key, detail=close_reason, qty=qty, est_pnl=total_pnl)
                self.debug(f"CLOSE {key}: {close_reason} estPnL={self._fmt_money(total_pnl)}")

        for k in to_remove:
            self.open_positions.pop(k, None)

    # -----------------------------
    # Wheel detection: put assignment -> shares appear
    # -----------------------------
    def _check_for_put_assignment(self):
        """
        If shares >= 100 and wheel is not active, assume a short put assignment occurred.
        We then:
          - Close any remaining short calls from tracked strangles (simplify)
          - Enter wheel mode (covered calls)
        """
        shares = int(self.portfolio[self.nvda].quantity)

        # Emergency: short stock indicates call assignment without shares (shouldn't happen in covered flow)
        if shares < 0:
            self.debug(f"EMERGENCY SHORT STOCK: sh={shares}. Covering and resetting.")
            # Close any open option calls we know about
            for _, pos in list(self.open_positions.items()):
                call_sym = pos["call"]
                if self.portfolio[call_sym].invested:
                    self.liquidate(call_sym)
            if self.wheel_cc_symbol and self.portfolio[self.wheel_cc_symbol].invested:
                self.liquidate(self.wheel_cc_symbol)
                self.wheel_cc_symbol = None
            # Cover the short stock immediately
            self.market_order(self.nvda, abs(shares))
            # Reset all state
            self.wheel_active = False
            self.wheel_entry_price = None
            self.wheel_entry_date = None
            self.open_positions.clear()
            return

        if shares >= 100 and not self.wheel_active:
            # Close remaining short calls from tracked strangles and stop tracking those strangles
            for key, pos in list(self.open_positions.items()):
                call_sym = pos["call"]
                if self.portfolio[call_sym].invested:
                    self.liquidate(call_sym)
                self.open_positions.pop(key, None)

            self.wheel_active = True
            self.wheel_entry_price = float(self.securities[self.nvda].price)
            self.wheel_entry_date = self.time
            self.wheel_cc_symbol = None

            self._append_trade_row(event="WHEEL_ON", detail=f"entry~{self.wheel_entry_price:.2f}", qty=shares, price=self.wheel_entry_price)
            self.debug(f"WHEEL ON: shares={shares} entry~{self.wheel_entry_price:.2f}")

    # -----------------------------
    # Wheel exit: shares go away (called away or sold)
    # -----------------------------
    def _check_wheel_exit(self):
        """
        Exit wheel mode when:
          - shares == 0 (called away / sold)
          - shares < 100 (partial/odd-lot state)
          - wheel timeout
          - wheel stop loss (down 20% from entry)
        """
        if not self.wheel_active:
            return

        shares = int(self.portfolio[self.nvda].quantity)
        spot = float(self.securities[self.nvda].price)

        if shares == 0:
            self.wheel_active = False
            self.wheel_entry_price = None
            self.wheel_entry_date = None
            self.wheel_cc_symbol = None
            self._append_trade_row(event="WHEEL_OFF", detail="shares=0")
            self.debug("WHEEL OFF: shares called away/sold.")
            return

        if shares < 100:
            # Keep it simple: liquidate odd lot and reset.
            self.liquidate(self.nvda)
            self.wheel_active = False
            self.wheel_entry_price = None
            self.wheel_entry_date = None
            self.wheel_cc_symbol = None
            self._append_trade_row(event="WHEEL_OFF", detail=f"oddlot shares={shares}")
            self.debug(f"WHEEL OFF: oddlot shares={shares}, liquidated.")
            return

        days_in_wheel = (self.time - self.wheel_entry_date).days if self.wheel_entry_date else 0
        if days_in_wheel > self.max_wheel_days:
            if self.wheel_cc_symbol and self.portfolio[self.wheel_cc_symbol].invested:
                self.liquidate(self.wheel_cc_symbol)
            self.liquidate(self.nvda)
            self.wheel_active = False
            self.wheel_entry_price = None
            self.wheel_entry_date = None
            self.wheel_cc_symbol = None
            self._append_trade_row(event="WHEEL_OFF", detail=f"timeout {days_in_wheel}d")
            self.debug(f"WHEEL OFF: timeout {days_in_wheel}d.")
            return

        if self.wheel_entry_price and spot < self.wheel_entry_price * 0.80:
            if self.wheel_cc_symbol and self.portfolio[self.wheel_cc_symbol].invested:
                self.liquidate(self.wheel_cc_symbol)
            self.liquidate(self.nvda)
            self.wheel_active = False
            self.wheel_entry_price = None
            self.wheel_entry_date = None
            self.wheel_cc_symbol = None
            self._append_trade_row(event="WHEEL_OFF", detail="stoploss -20%")
            self.debug("WHEEL OFF: stoploss hit (-20% from entry).")
            return

    # -----------------------------
    # Covered call selling (wheel)
    # -----------------------------
    def _execute_sell_covered_call(self, data: Slice):
        """
        Sell 1 covered call per 100 shares, only when:
          - wheel_active
          - shares >= 100
          - no existing invested CC
        Contract selection:
          - 7–30 DTE calls
          - target ~0.25 delta
          - prefer OTM strikes (>= spot)
          - bid/ask quality gate
          - limit @ mid
        """
        if not self.wheel_active:
            return

        shares = int(self.portfolio[self.nvda].quantity)
        if shares < 100:
            return

        # Clear stale CC symbol (expired/closed)
        if self.wheel_cc_symbol and not self.portfolio[self.wheel_cc_symbol].invested:
            self.wheel_cc_symbol = None

        # If CC is still open, don't sell another
        if self.wheel_cc_symbol and self.portfolio[self.wheel_cc_symbol].invested:
            return

        chain = data.option_chains.get(self.nvda_opt)
        if not chain:
            return

        calls = [
            x for x in chain
            if x.right == OptionRight.CALL
            and 7 <= (x.expiry - self.time).days <= 30
            and x.greeks and x.greeks.delta is not None
            and x.bid_price > 0 and x.ask_price > 0
        ]
        if not calls:
            return

        spot = float(self.securities[self.nvda].price)
        otm_calls = [c for c in calls if c.strike >= spot]
        pool = otm_calls if otm_calls else calls

        call = min(pool, key=lambda x: abs(x.greeks.delta - 0.25))

        if not self._spread_ok(call.bid_price, call.ask_price):
            return

        contracts = shares // 100
        limit_px = self._mid(call.bid_price, call.ask_price)
        if limit_px <= 0:
            return

        # Tag contract (only used if log_fills=True)
        self.option_tags[call.symbol] = "COVERED_CALL"

        cc_key = f"WHEEL_{self.time.date()}"
        self._place_limit(
            call.symbol, -contracts, limit_px,
            action="OPEN", reason="COVERED_CALL", key=cc_key, leg="COVERED_CALL",
            note=f"strike={call.strike} Δ={call.greeks.delta:.2f} DTE={(call.expiry - self.time).days}"
        )
        
        self.wheel_cc_symbol = call.symbol

        credit = limit_px * 100 * contracts
        self._append_trade_row(
            event="SELL_CC",
            detail=f"strike={call.strike} Δ={call.greeks.delta:.2f} DTE={(call.expiry - self.time).days}",
            qty=contracts,
            price=limit_px,
            credit_debit=credit
        )
        self.debug(f"SELL CC: qty={contracts} strike={call.strike} prem={limit_px:.2f}")

    # -----------------------------
    # ObjectStore CSV helpers
    # -----------------------------
    def _ensure_objectstore_csv(self, key: str, header: str):
        """Create CSV in ObjectStore with header if missing. No debug unless error."""
        try:
            if not self.object_store.contains_key(key):
                self.object_store.save(key, header)
                return
            existing = self.object_store.read(key)
            if not existing or len(existing) < len(header):
                self.object_store.save(key, header)
        except Exception as e:
            self.debug(f"[OS] ensure failed {key}: {e}")

    def _append_daily_row(self):
        """Buffer one daily snapshot row for CSV."""
        today = self.time.date()
        now = self.time.strftime("%H:%M:%S")

        equity = float(self.portfolio.total_portfolio_value)
        cash = float(self.portfolio.cash)
        margin_used = float(self.portfolio.total_margin_used)

        spot = float(self.securities[self.nvda].price)
        shares = int(self.portfolio[self.nvda].quantity)
        avg_price = float(self.portfolio[self.nvda].average_price) if shares != 0 else 0.0

        opt_unrl = float(self._compute_options_unrealized())
        stk_unrl = float(self._compute_stock_unrealized())
        opt_real = float(self.pnl.get("options_realized", 0.0))
        hold = self.portfolio[self.nvda]
        stk_real = float(hold.profit - hold.unrealized_profit) if hold.invested or hold.profit != 0 else 0.0

        # stk_real = float(self.pnl.get("stock_realized", 0.0))

        open_strangles = int(len(self.open_positions))
        wheel_active = int(1 if self.wheel_active else 0)
        wheel_cc = str(self.wheel_cc_symbol) if self.wheel_cc_symbol else ""

        self._daily_rows.append(
            f"{today},{now},{equity:.2f},{cash:.2f},{margin_used:.2f},"
            f"{spot:.2f},{shares},{avg_price:.4f},"
            f"{opt_unrl:.2f},{stk_unrl:.2f},{opt_real:.2f},{stk_real:.2f},"
            f"{open_strangles},{wheel_active},{wheel_cc}\n"
        )

    def _append_trade_row(self, event: str, key: str = "", detail: str = "",
                          qty: float = 0.0, price: float = 0.0,
                          credit_debit: float = 0.0, est_pnl: float = 0.0):
        """Buffer one event row for CSV (OPEN/CLOSE/WHEEL/CC)."""
        today = self.time.date()
        now = self.time.strftime("%H:%M:%S")
        equity = float(self.portfolio.total_portfolio_value)
        spot = float(self.securities[self.nvda].price)
        shares = int(self.portfolio[self.nvda].quantity)

        self._trade_rows.append(
            f"{today},{now},{event},{key},\"{detail}\",{qty:.0f},{price:.4f},"
            f"{credit_debit:.2f},{est_pnl:.2f},{equity:.2f},{spot:.2f},{shares}\n"
        )

    def _flush_ledgers_to_objectstore(self):
        """Append buffered rows to ObjectStore. No debug unless error."""
        try:
            if self._daily_rows:
                existing = self.object_store.read(self.os_daily_key)
                self.object_store.save(self.os_daily_key, existing + "".join(self._daily_rows))
                self._daily_rows = []

            if self._trade_rows:
                existing = self.object_store.read(self.os_trade_key)
                self.object_store.save(self.os_trade_key, existing + "".join(self._trade_rows))
                self._trade_rows = []
        except Exception as e:
            self.debug(f"[OS] flush failed: {e}")

    def on_end_of_algorithm(self):
        """Final snapshot + final flush."""
        try:
            self._append_daily_row()
            self._flush_ledgers_to_objectstore()
            self.debug(f"[OS] Saved: {self.os_daily_key}, {self.os_trade_key}")
        except Exception as e:
            self.debug(f"[OS] end flush failed: {e}")


    def _btc_qty(self, symbol) -> int:
        """Return the absolute quantity needed to close the current holding for this symbol."""
        q = int(self.portfolio[symbol].quantity)
        return abs(q)

    def _safe_mid_or_fallback(self, sec, fallback: float) -> float:
        """Prefer mid, fallback to ask/last if mid invalid."""
        mid = self._mid(sec.bid_price, sec.ask_price)
        if mid > 0:
            return mid
        # for shorts we expect to pay ask; if ask missing use last
        if sec.ask_price and sec.ask_price > 0:
            return float(sec.ask_price)
        return float(fallback)

    def _buy_to_close_option(self, symbol, *, qty: int, limit_px: float,
                            action: str, reason: str, key: str, leg: str,
                            est_pnl: float = 0.0, note: str = ""):
        """
        Place an explicit BUY-to-close (limit) for a short option position with tagging + context.
        - qty must be positive
        """
        if qty <= 0:
            return None
        return self._place_limit(
            symbol, +qty, limit_px,
            action=action, reason=reason, key=key, leg=leg,
            est_pnl=est_pnl, note=note
        )

    def _sell_to_close_option(self, symbol, *, qty: int, limit_px: float,
                            action: str, reason: str, key: str, leg: str,
                            est_pnl: float = 0.0, note: str = ""):
        """
        Explicit SELL-to-close for long options (rare in your strat, but included for completeness).
        - qty must be positive
        """
        if qty <= 0:
            return None
        return self._place_limit(
            symbol, -qty, limit_px,
            action=action, reason=reason, key=key, leg=leg,
            est_pnl=est_pnl, note=note
        )

    def _close_option_leg_explicit(self, symbol, *, action: str, reason: str, key: str, leg: str,
                                est_pnl: float = 0.0, note: str = ""):
        """
        Close any option leg explicitly with a tagged limit order:
        - If short: BUY to close at mid (fallback ask)
        - If long:  SELL to close at mid (fallback bid/last)
        """
        hold = self.portfolio[symbol]
        if not hold.invested:
            return None

        sec = self.securities[symbol]
        qty_abs = abs(int(hold.quantity))
        if qty_abs <= 0:
            return None

        # Choose limit price intelligently:
        # - short -> pay mid (fallback ask)
        # - long  -> receive mid (fallback bid)
        if hold.quantity < 0:
            fallback = float(sec.ask_price) if sec.ask_price and sec.ask_price > 0 else float(sec.price)
            limit_px = self._safe_mid_or_fallback(sec, fallback)
            return self._buy_to_close_option(
                symbol, qty=qty_abs, limit_px=limit_px,
                action=action, reason=reason, key=key, leg=leg,
                est_pnl=est_pnl, note=note
            )
        else:
            # long option close (sell)
            mid = self._mid(sec.bid_price, sec.ask_price)
            limit_px = mid if mid > 0 else (float(sec.bid_price) if sec.bid_price and sec.bid_price > 0 else float(sec.price))
            return self._sell_to_close_option(
                symbol, qty=qty_abs, limit_px=limit_px,
                action=action, reason=reason, key=key, leg=leg,
                est_pnl=est_pnl, note=note
            )

    def _close_strangle_explicit(self, key: str, pos: dict, *, action: str, reason: str,
                                est_pnl: float = 0.0, note: str = ""):
        """
        Close BOTH legs explicitly using tagged BTC/STC orders (no Liquidate()).
        """
        call_sym = pos["call"]
        put_sym  = pos["put"]

        if self.portfolio[call_sym].invested:
            self._close_option_leg_explicit(
                call_sym, action=action, reason=reason, key=key, leg="STRANGLE_CALL",
                est_pnl=est_pnl, note=note
            )

        if self.portfolio[put_sym].invested:
            self._close_option_leg_explicit(
                put_sym, action=action, reason=reason, key=key, leg="STRANGLE_PUT",
                est_pnl=est_pnl, note=note
            )
