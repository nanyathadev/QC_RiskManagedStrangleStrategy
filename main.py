# region imports
from AlgorithmImports import *
from datetime import timedelta
import math
# endregion


class RiskManagedStrangleStrategy(QCAlgorithm):
    """
    Multi-symbol HYBRID short strangle strategy + ATR% regime filter + wheel-on-put-assignment (covered calls)
    + lightweight logging + CSV/ObjectStore P&L ledgers.

    HYBRID SYSTEM:
      - MONTHLY CORE: open strangles in 30–45 DTE by default
      - WEEKLY OPPORTUNISTIC: open strangles in 7–10 DTE ONLY when volatility is very calm (ATR% <= weekly_atr_pct_max)
        and optionally VIX <= weekly_vix_max
      - Weeklies use smaller sizing + faster time-exit.

    ENTRY SCHEDULING:
      - Per-symbol STAGGER by weekday (map below). If ticker not in map, defaults to Monday.
      - Only one entry attempt per symbol per day.

    Risk controls already present:
      - VIX gate
      - ATR% gate
      - bid/ask spread gate
      - delta target + tolerance
      - max positions (global)
      - prevent-only call assignment protection (force close)
      - wheel on put assignment + covered calls
    """

    # -----------------------------
    # Init
    # -----------------------------
    def initialize(self):
        self.set_start_date(2025, 1, 1)
        self.set_end_date(2025, 3, 31)
        self.set_cash(100000)
        self.settings.seed_initial_prices = True

        # === Log switches (flip to True when actively debugging) ===
        self.log_fills = False
        self.log_daily_summary = True
        self.log_per_position_debug = False

        # -----------------------------
        # Underlyings + options
        # -----------------------------
        self.underlyings = ["NVDA", "AAPL"]  # extend as needed
        self.symbol_state = {}
        self.equities = []

        for ticker in self.underlyings:
            equity = self.add_equity(ticker, Resolution.HOUR).symbol
            opt = self.add_option(ticker, Resolution.HOUR).symbol

            # HYBRID needs weeklies too: allow short expirations
            self.securities[opt].set_filter(lambda u: u.strikes(-80, 80).expiration(5, 60))

            self.symbol_state[equity] = {
                "ticker": ticker,
                "equity": equity,
                "option": opt,
                "atr": self.atr(equity, 14, MovingAverageType.WILDERS, Resolution.DAILY),

                "open_positions": {},          # key -> pos dict
                "position_counter": 0,

                "wheel_active": False,
                "wheel_entry_price": None,
                "wheel_entry_date": None,
                "wheel_cc_symbol": None,

                "last_strangle_check": None,
                "last_cc_check": None,
                "cooldown_until": None,
                "_log_last_daily_date": None,
            }
            self.equities.append(equity)

        self.primary_symbol = self.equities[0]

        # -----------------------------
        # Entry staggering by symbol (0=Mon,1=Tue,2=Wed,3=Thu,4=Fri)
        # -----------------------------
        self.weekday_map = {
            "NVDA": 0,  # Monday
            "AAPL": 1,  # Tuesday
            # Add more when you extend the universe
        }

        # -----------------------------
        # Macro filter (VIX)
        # -----------------------------
        self.vix = self.add_data(CBOE, "VIX", Resolution.DAILY).symbol
        self.max_vix = 20

        # -----------------------------
        # Volatility proxy: ATR%
        # -----------------------------
        self.set_warm_up(30, Resolution.DAILY)

        self.max_atr_pct_entry = 0.055
        self.atr_low_pct = 0.030
        self.atr_high_pct = 0.045

        # -----------------------------
        # HYBRID parameters
        # -----------------------------
        self.monthly_dte_min = 30
        self.monthly_dte_max = 45

        self.weekly_dte_min = 7
        self.weekly_dte_max = 10
        self.weekly_atr_pct_max = 0.022   # only sell weeklies in very calm regimes
        self.weekly_vix_max = 18          # optional tighter VIX for weeklies
        self.weekly_size_mult = 0.50      # smaller size for weeklies
        self.weekly_time_exit_days = 3    # close when <= 3 DTE for weeklies

        # -----------------------------
        # Strangle parameters
        # -----------------------------
        self.total_stop_loss_multiplier = 2.5
        self.partial_profit_target = 0.25
        self.full_profit_target = 0.50
        self.max_positions = 4

        self.target_delta = 0.10
        self.delta_tolerance = 0.03

        self.base_position_size_pct = 0.10
        self.max_contracts_cap = 2

        # monthly time exit (weeklies override per-position)
        self.time_exit_days = 21
        self.max_spread_pct_of_mid = 0.08

        # -----------------------------
        # Wheel state
        # -----------------------------
        self.max_wheel_days = 45

        # --- Order tagging + fill attribution ---
        self._order_context = {}
        self._last_holdings_profit = {}

        # -----------------------------
        # Lightweight P&L buckets (for quick debugging)
        # -----------------------------
        self.pnl = {
            sym: {"options_realized": 0.0, "stock_realized": 0.0}
            for sym in self.equities
        }

        # Tags for options (used only if log_fills=True)
        self.option_tags = {}

        # -----------------------------
        # ObjectStore CSV ledgers
        # -----------------------------
        self.os_daily_key = "pnl_daily_ledger.csv"
        self.os_trade_key = "pnl_trade_ledger.csv"

        self._daily_rows = []
        self._trade_rows = []

        self._daily_header = (
            "date,time,symbol,equity,cash,margin_used,spot,shares,avg_price,"
            "opt_unrl,stk_unrl,opt_real,stk_real,open_strangles,wheel_active,wheel_cc\n"
        )
        self._trade_header = (
            "date,time,symbol,event,key,detail,qty,price,credit_debit,est_pnl,equity,spot,shares\n"
        )

        # --- Prevent-only call assignment protection ---
        self.prevent_assignment = True
        self.call_delta_threat = 0.65
        self.itm_buffer_pct = 0.01
        self.assignment_risk_dte = 7
        self.cooldown_days_after_threat = 3

        self._ensure_objectstore_csv(self.os_daily_key, self._daily_header)
        self._ensure_objectstore_csv(self.os_trade_key, self._trade_header)

        # -----------------------------
        # Scheduling
        # -----------------------------
        self.schedule.on(
            self.date_rules.every_day(self.primary_symbol),
            self.time_rules.after_market_open(self.primary_symbol, 35),
            self._manage_positions
        )

        self.schedule.on(
            self.date_rules.every_day(self.primary_symbol),
            self.time_rules.before_market_close(self.primary_symbol, 1),
            self._flush_ledgers_to_objectstore
        )

    # -----------------------------
    # Helpers (pricing + filters)
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

    def _get_atr_pct(self, state: dict):
        equity = state["equity"]
        atr = state["atr"]
        price = self.securities[equity].price
        if self.is_warming_up or (not atr.is_ready) or price <= 0:
            return None
        return float(atr.current.value) / float(price)

    def _fmt_money(self, x: float) -> str:
        return f"${x:,.2f}"

    def _total_open_positions(self) -> int:
        total = 0
        for st in self.symbol_state.values():
            total += len(st["open_positions"])
        return total

    def _in_cooldown(self, state: dict) -> bool:
        return state["cooldown_until"] is not None and self.time.date() < state["cooldown_until"]

    def _eligible_weekday(self, ticker: str) -> bool:
        target = self.weekday_map.get(ticker, 0)
        return self.time.weekday() == int(target)

    def _make_tag(self, *, action: str, reason: str, key: str = "", leg: str = "", extra: str = "") -> str:
        parts = [action]
        if reason:
            parts.append(reason)
        if key:
            parts.append(key)
        if leg:
            parts.append(leg)
        if extra:
            parts.append(extra)
        return "|".join(parts)[:180]

    def _track_ticket(self, ticket, *, action: str, reason: str, key: str, leg: str,
                      est_pnl: float = 0.0, note: str = ""):
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
        tag = self._make_tag(action=action, reason=reason, key=key, leg=leg, extra=note)
        ticket = self.limit_order(symbol, qty, limit_price, tag=tag)
        self._track_ticket(ticket, action=action, reason=reason, key=key, leg=leg, est_pnl=est_pnl, note=note)
        return ticket

    # -----------------------------
    # P&L computations
    # -----------------------------
    def _compute_options_unrealized(self, equity_symbol=None) -> float:
        total = 0.0
        for kvp in self.portfolio:
            sec = kvp.key
            hold = kvp.value
            if hold.invested and sec.security_type == SecurityType.OPTION:
                if equity_symbol is None or sec.underlying == equity_symbol:
                    total += float(hold.unrealized_profit)
        return total

    def _compute_stock_unrealized(self, equity_symbol) -> float:
        hold = self.portfolio[equity_symbol]
        return float(hold.unrealized_profit) if hold.invested else 0.0

    # -----------------------------
    # Daily snapshot (minimal debug + always to CSV)
    # -----------------------------
    def _log_daily_snapshot(self, state: dict):
        today = self.time.date()
        if state["_log_last_daily_date"] == today:
            return
        state["_log_last_daily_date"] = today

        equity = float(self.portfolio.total_portfolio_value)
        cash = float(self.portfolio.cash)
        margin_used = float(self.portfolio.total_margin_used)

        equity_symbol = state["equity"]
        opt_unrl = float(self._compute_options_unrealized(equity_symbol))
        stk_unrl = float(self._compute_stock_unrealized(equity_symbol))

        shares = int(self.portfolio[equity_symbol].quantity)
        avg_price = float(self.portfolio[equity_symbol].average_price) if shares != 0 else 0.0
        spot = float(self.securities[equity_symbol].price)
        ticker = state["ticker"]

        if self.log_daily_summary:
            self.debug(
                f"[DAILY] {today} {ticker} eq={self._fmt_money(equity)} "
                f"optU={self._fmt_money(opt_unrl)} stkU={self._fmt_money(stk_unrl)} "
                f"optR={self._fmt_money(self.pnl[equity_symbol]['options_realized'])} "
                f"stkR={self._fmt_money(self.pnl[equity_symbol]['stock_realized'])} "
                f"spot={spot:.2f} sh={shares} openStr={len(state['open_positions'])} wheel={state['wheel_active']}"
            )

        self._append_daily_row(state)

    def _log_position_snapshot(self, state: dict, key: str, pos: dict, reason: str = "SNAP"):
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

        call_close = call_sec.ask_price if (call_inv and call_sec.ask_price > 0) else call_sec.price
        put_close = put_sec.ask_price if (put_inv and put_sec.ask_price > 0) else put_sec.price

        call_pnl = (pos["call_open_price"] - call_close) * 100 * qty if call_inv else 0.0
        put_pnl = (pos["put_open_price"] - put_close) * 100 * qty if put_inv else 0.0
        total = call_pnl + put_pnl

        dte = (pos["expiry"] - self.time).days
        spot = float(self.securities[state["equity"]].price)

        self.debug(
            f"[{reason}] {key} type={pos.get('cycle','?')} exp={expiry} DTE={dte} spot={spot:.2f} "
            f"PnL={self._fmt_money(total)} call={self._fmt_money(call_pnl)} put={self._fmt_money(put_pnl)}"
        )

    # -----------------------------
    # Main data loop
    # -----------------------------
    def on_data(self, data: Slice):
        for state in self.symbol_state.values():
            if state["option"] not in data.option_chains:
                continue

            # HYBRID entry: stagger by symbol weekday, once/day
            if self._eligible_weekday(state["ticker"]) and state["last_strangle_check"] != self.time.date():
                state["last_strangle_check"] = self.time.date()
                self._execute_strangle_open_hybrid(data, state)

            # Covered call check daily while wheel is active (once per day)
            if state["wheel_active"] and state["last_cc_check"] != self.time.date():
                state["last_cc_check"] = self.time.date()
                self._execute_sell_covered_call(data, state)

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

        hold = self.portfolio[symbol]
        prev_profit = float(self._last_holdings_profit.get(symbol, 0.0))
        cur_profit = float(hold.profit)
        self._last_holdings_profit[symbol] = cur_profit
        realized_delta = cur_profit - prev_profit

        if self.log_fills:
            if ctx:
                self.debug(
                    f"[FILL] {ctx['action']} {ctx['reason']} {ctx['key']} {ctx['leg']} "
                    f"{symbol} qty={qty:+.0f} px={px:.2f} profΔ={realized_delta:+.2f}"
                )
            else:
                self.debug(f"[FILL] {symbol} qty={qty:+.0f} px={px:.2f} profΔ={realized_delta:+.2f}")

        # CSV fill row
        ledger_symbol = symbol.underlying if symbol.security_type == SecurityType.OPTION else symbol
        if ctx:
            detail = f"{ctx['action']}|{ctx['reason']}|{ctx['leg']}|{ctx.get('note','')}"
            self._append_trade_row(
                symbol=ledger_symbol,
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
                symbol=ledger_symbol,
                event="FILL",
                key="",
                detail=f"NOCTX|oid={oid}|sym={symbol}",
                qty=qty,
                price=px,
                credit_debit=0.0,
                est_pnl=0.0
            )

        # update realized bucket for options when the holding closes
        if symbol.security_type == SecurityType.OPTION and not hold.invested:
            underlying = symbol.underlying
            if underlying in self.pnl:
                self.pnl[underlying]["options_realized"] += float(hold.profit)

    # -----------------------------
    # HYBRID Entry: monthly core OR weekly opportunistic
    # -----------------------------
    def _execute_strangle_open_hybrid(self, data: Slice, state: dict):
        if self.is_warming_up:
            return

        ticker = state["ticker"]

        if self._in_cooldown(state):
            self.debug(f"Skipping {ticker} entry — cooldown until {state['cooldown_until']}")
            return

        # No new strangles while wheeling (risk stacking)
        if state["wheel_active"]:
            return

        # Max 1 strangle per symbol at a time
        if len(state["open_positions"]) >= 1:
            return

        # Global cap
        if self._total_open_positions() >= self.max_positions:
            return

        # Buying power buffer
        buying_power = self.portfolio.total_portfolio_value - abs(self.portfolio.total_margin_used)
        if buying_power < self.portfolio.total_portfolio_value * 0.30:
            return

        # VIX gate (macro safety)
        vix_val = None
        if self.vix in data and data[self.vix]:
            vix_val = float(data[self.vix].value)
            if vix_val > self.max_vix:
                return

        # ATR% gate (hard)
        atr_pct = self._get_atr_pct(state)
        if atr_pct is None or atr_pct > self.max_atr_pct_entry:
            return

        # Decide WEEKLY vs MONTHLY
        weekly_ok = atr_pct <= self.weekly_atr_pct_max and (vix_val is None or vix_val <= self.weekly_vix_max)
        if weekly_ok:
            cycle = "WEEKLY"
            dte_min, dte_max = self.weekly_dte_min, self.weekly_dte_max
            size_mult = self.weekly_size_mult
            time_exit_days = self.weekly_time_exit_days
        else:
            cycle = "MONTHLY"
            dte_min, dte_max = self.monthly_dte_min, self.monthly_dte_max
            # keep your existing ATR sizing logic for monthly
            if atr_pct < self.atr_low_pct:
                size_mult = 1.25
            elif atr_pct > self.atr_high_pct:
                size_mult = 0.65
            else:
                size_mult = 1.0
            time_exit_days = self.time_exit_days

        adjusted_size_pct = self.base_position_size_pct * size_mult

        chain = data.option_chains.get(state["option"])
        if not chain:
            return

        # Pick earliest expiry in the chosen DTE window with greeks
        expiries = sorted({
            x.expiry for x in chain
            if dte_min <= (x.expiry - self.time).days <= dte_max
            and x.greeks and x.greeks.delta is not None
        })
        if not expiries:
            return
        expiry = expiries[0]

        calls = [x for x in chain if x.expiry == expiry and x.right == OptionRight.CALL
                 and x.greeks and x.greeks.delta is not None and x.bid_price > 0 and x.ask_price > 0]
        puts = [x for x in chain if x.expiry == expiry and x.right == OptionRight.PUT
                and x.greeks and x.greeks.delta is not None and x.bid_price > 0 and x.ask_price > 0]
        if not calls or not puts:
            return

        tgt, tol = self.target_delta, self.delta_tolerance
        call = min(calls, key=lambda x: abs(abs(x.greeks.delta) - tgt))
        put = min(puts, key=lambda x: abs(abs(x.greeks.delta) - tgt))

        if abs(abs(call.greeks.delta) - tgt) > tol or abs(abs(put.greeks.delta) - tgt) > tol:
            return

        if not self._spread_ok(call.bid_price, call.ask_price) or not self._spread_ok(put.bid_price, put.ask_price):
            return

        call_mid = self._mid(call.bid_price, call.ask_price)
        put_mid = self._mid(put.bid_price, put.ask_price)
        premium_mid = (call_mid + put_mid) * 100
        if premium_mid <= 0:
            return

        estimated_margin_per_contract = max(call.strike, put.strike) * 100 * 0.20
        available_margin = self.portfolio.total_portfolio_value - abs(self.portfolio.total_margin_used)

        budget = self.portfolio.total_portfolio_value * adjusted_size_pct
        qty_from_budget = max(1, int(budget / premium_mid))
        qty_from_margin = max(1, int(available_margin / estimated_margin_per_contract)) if estimated_margin_per_contract > 0 else 1
        qty = min(qty_from_budget, qty_from_margin, self.max_contracts_cap)

        state["position_counter"] += 1
        key = f"{ticker}_{cycle}_pos_{state['position_counter']}_{expiry.date()}"

        call_ticket = self._place_limit(
            call.symbol, -qty, call_mid,
            action="OPEN", reason=f"STRANGLE_{cycle}", key=key, leg="STRANGLE_CALL",
            est_pnl=0.0,
            note=f"exp={expiry.date()} strike={call.strike} Δ={call.greeks.delta:.3f}"
        )
        put_ticket = self._place_limit(
            put.symbol, -qty, put_mid,
            action="OPEN", reason=f"STRANGLE_{cycle}", key=key, leg="STRANGLE_PUT",
            est_pnl=0.0,
            note=f"exp={expiry.date()} strike={put.strike} Δ={put.greeks.delta:.3f}"
        )

        self.option_tags[call.symbol] = "STRANGLE_CALL"
        self.option_tags[put.symbol] = "STRANGLE_PUT"

        call_entry = call_ticket.average_fill_price if call_ticket.average_fill_price else call_mid
        put_entry = put_ticket.average_fill_price if put_ticket.average_fill_price else put_mid
        credit = (call_entry + put_entry) * 100 * qty

        state["open_positions"][key] = {
            "cycle": cycle,
            "time_exit_days": int(time_exit_days),

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

        self._append_trade_row(
            symbol=state["equity"],
            event="OPEN_STRANGLE",
            key=key,
            detail=f"type={cycle} exp={expiry.date()} call={call.strike} put={put.strike} "
                   f"Δc={call.greeks.delta:.3f} Δp={put.greeks.delta:.3f} "
                   f"ATR%={atr_pct*100:.2f} VIX={(vix_val if vix_val is not None else -1):.2f}",
            qty=qty,
            credit_debit=credit
        )

        self.debug(f"OPEN {key}: qty={qty} credit={self._fmt_money(credit)} exp={expiry.date()} type={cycle}")
        self._log_position_snapshot(state, key, state["open_positions"][key], reason="OPEN")

    # -----------------------------
    # Daily management: exits + assignment + wheel state
    # -----------------------------
    def _manage_positions(self):
        for state in self.symbol_state.values():
            self._log_daily_snapshot(state)

            self._check_wheel_exit(state)
            self._check_for_put_assignment(state)

            if state["wheel_active"] and state["wheel_cc_symbol"] and not self.portfolio[state["wheel_cc_symbol"]].invested:
                state["wheel_cc_symbol"] = None

            to_remove = []

            for key, pos in list(state["open_positions"].items()):
                call_sym, put_sym = pos["call"], pos["put"]
                call_inv = self.portfolio[call_sym].invested
                put_inv = self.portfolio[put_sym].invested

                if not call_inv and not put_inv:
                    to_remove.append(key)
                    continue

                qty = int(pos["quantity"])
                dte = int((pos["expiry"] - self.time).days)

                call_sec = self.securities[call_sym]
                put_sec = self.securities[put_sym]
                spot = float(self.securities[state["equity"]].price)

                call_close = call_sec.ask_price if (call_inv and call_sec.ask_price > 0) else call_sec.price
                put_close = put_sec.ask_price if (put_inv and put_sec.ask_price > 0) else put_sec.price

                call_pnl = (pos["call_open_price"] - call_close) * 100 * qty if call_inv else 0.0
                put_pnl = (pos["put_open_price"] - put_close) * 100 * qty if put_inv else 0.0
                total_pnl = call_pnl + put_pnl

                if self.log_per_position_debug:
                    self._log_position_snapshot(state, key, pos, reason="MANAGE")

                # --- Prevent-only: exit before call assignment risk ---
                if self.prevent_assignment and call_inv:
                    call_strike = float(pos.get("call_strike", 0.0)) or float(call_sec.symbol.id.strike_price)

                    call_delta = None
                    try:
                        if call_sec.greeks and call_sec.greeks.delta is not None:
                            call_delta = abs(float(call_sec.greeks.delta))
                    except Exception:
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
                        state["cooldown_until"] = self.time.date() + timedelta(days=self.cooldown_days_after_threat)

                        self._append_trade_row(
                            symbol=state["equity"],
                            event="FORCE_CLOSE_ASSIGNMENT_RISK",
                            key=key,
                            detail=close_reason,
                            qty=qty,
                            credit_debit=0.0,
                            est_pnl=total_pnl
                        )
                        self.debug(
                            f"FORCE CLOSE {key}: {close_reason} estPnL={self._fmt_money(total_pnl)} "
                            f"cooldown={state['cooldown_until']}"
                        )
                        continue

                # Exit rules
                close_reason = None
                partial_close = False

                # per-position time exit (weekly vs monthly)
                time_exit_days = int(pos.get("time_exit_days", self.time_exit_days))

                if dte <= time_exit_days:
                    close_reason = f"TIME_EXIT {dte}DTE (thr={time_exit_days})"
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
                        self._place_limit(
                            call_sym, close_qty, self._mid(call_sec.bid_price, call_sec.ask_price) or call_close,
                            action="CLOSE", reason="PARTIAL_TP", key=key, leg="STRANGLE_CALL",
                            est_pnl=total_pnl
                        )
                    if put_inv:
                        self._place_limit(
                            put_sym, close_qty, self._mid(put_sec.bid_price, put_sec.ask_price) or put_close,
                            action="CLOSE", reason="PARTIAL_TP", key=key, leg="STRANGLE_PUT",
                            est_pnl=total_pnl
                        )

                    pos["quantity"] = qty - close_qty
                    pos["partial_profit_taken"] = True

                    self._append_trade_row(
                        symbol=state["equity"],
                        event="PARTIAL_CLOSE",
                        key=key,
                        detail=close_reason,
                        qty=close_qty,
                        est_pnl=total_pnl
                    )
                    self.debug(f"PARTIAL {key}: {close_reason} closed={close_qty}/{qty}")

                else:
                    self._close_strangle_explicit(
                        key, pos,
                        action="CLOSE",
                        reason="EXIT",
                        est_pnl=total_pnl,
                        note=close_reason
                    )
                    to_remove.append(key)
                    self._append_trade_row(
                        symbol=state["equity"],
                        event="FULL_CLOSE",
                        key=key,
                        detail=close_reason,
                        qty=qty,
                        est_pnl=total_pnl
                    )
                    self.debug(f"CLOSE {key}: {close_reason} estPnL={self._fmt_money(total_pnl)}")

            for k in to_remove:
                state["open_positions"].pop(k, None)

    # -----------------------------
    # Wheel detection: put assignment -> shares appear
    # -----------------------------
    def _check_for_put_assignment(self, state: dict):
        equity = state["equity"]
        shares = int(self.portfolio[equity].quantity)

        # Emergency short stock
        if shares < 0:
            self.debug(f"EMERGENCY SHORT STOCK {state['ticker']}: sh={shares}. Covering and resetting.")
            for _, pos in list(state["open_positions"].items()):
                call_sym = pos["call"]
                if self.portfolio[call_sym].invested:
                    self.liquidate(call_sym)
            if state["wheel_cc_symbol"] and self.portfolio[state["wheel_cc_symbol"]].invested:
                self.liquidate(state["wheel_cc_symbol"])
                state["wheel_cc_symbol"] = None
            self.market_order(equity, abs(shares))
            state["wheel_active"] = False
            state["wheel_entry_price"] = None
            state["wheel_entry_date"] = None
            state["open_positions"].clear()
            return

        if shares >= 100 and not state["wheel_active"]:
            # Close remaining short calls from tracked strangles and stop tracking those strangles
            for key, pos in list(state["open_positions"].items()):
                call_sym = pos["call"]
                if self.portfolio[call_sym].invested:
                    self.liquidate(call_sym)
                state["open_positions"].pop(key, None)

            state["wheel_active"] = True
            state["wheel_entry_price"] = float(self.securities[equity].price)
            state["wheel_entry_date"] = self.time
            state["wheel_cc_symbol"] = None

            self._append_trade_row(
                symbol=equity,
                event="WHEEL_ON",
                detail=f"entry~{state['wheel_entry_price']:.2f}",
                qty=shares,
                price=state["wheel_entry_price"]
            )
            self.debug(f"WHEEL ON {state['ticker']}: shares={shares} entry~{state['wheel_entry_price']:.2f}")

    # -----------------------------
    # Wheel exit
    # -----------------------------
    def _check_wheel_exit(self, state: dict):
        if not state["wheel_active"]:
            return

        equity = state["equity"]
        shares = int(self.portfolio[equity].quantity)
        spot = float(self.securities[equity].price)

        if shares == 0:
            state["wheel_active"] = False
            state["wheel_entry_price"] = None
            state["wheel_entry_date"] = None
            state["wheel_cc_symbol"] = None
            self._append_trade_row(symbol=equity, event="WHEEL_OFF", detail="shares=0")
            self.debug(f"WHEEL OFF {state['ticker']}: shares called away/sold.")
            return

        if shares < 100:
            self.liquidate(equity)
            state["wheel_active"] = False
            state["wheel_entry_price"] = None
            state["wheel_entry_date"] = None
            state["wheel_cc_symbol"] = None
            self._append_trade_row(symbol=equity, event="WHEEL_OFF", detail=f"oddlot shares={shares}")
            self.debug(f"WHEEL OFF {state['ticker']}: oddlot shares={shares}, liquidated.")
            return

        days_in_wheel = (self.time - state["wheel_entry_date"]).days if state["wheel_entry_date"] else 0
        if days_in_wheel > self.max_wheel_days:
            if state["wheel_cc_symbol"] and self.portfolio[state["wheel_cc_symbol"]].invested:
                self.liquidate(state["wheel_cc_symbol"])
            self.liquidate(equity)
            state["wheel_active"] = False
            state["wheel_entry_price"] = None
            state["wheel_entry_date"] = None
            state["wheel_cc_symbol"] = None
            self._append_trade_row(symbol=equity, event="WHEEL_OFF", detail=f"timeout {days_in_wheel}d")
            self.debug(f"WHEEL OFF {state['ticker']}: timeout {days_in_wheel}d.")
            return

        if state["wheel_entry_price"] and spot < state["wheel_entry_price"] * 0.80:
            if state["wheel_cc_symbol"] and self.portfolio[state["wheel_cc_symbol"]].invested:
                self.liquidate(state["wheel_cc_symbol"])
            self.liquidate(equity)
            state["wheel_active"] = False
            state["wheel_entry_price"] = None
            state["wheel_entry_date"] = None
            state["wheel_cc_symbol"] = None
            self._append_trade_row(symbol=equity, event="WHEEL_OFF", detail="stoploss -20%")
            self.debug(f"WHEEL OFF {state['ticker']}: stoploss hit (-20% from entry).")
            return

    # -----------------------------
    # Covered call selling (wheel)
    # -----------------------------
    def _execute_sell_covered_call(self, data: Slice, state: dict):
        if not state["wheel_active"]:
            return

        equity = state["equity"]
        shares = int(self.portfolio[equity].quantity)
        if shares < 100:
            return

        if state["wheel_cc_symbol"] and not self.portfolio[state["wheel_cc_symbol"]].invested:
            state["wheel_cc_symbol"] = None

        if state["wheel_cc_symbol"] and self.portfolio[state["wheel_cc_symbol"]].invested:
            return

        chain = data.option_chains.get(state["option"])
        if not chain:
            return

        calls = [
            x for x in chain
            if x.right == OptionRight.CALL
            and self.weekly_dte_min <= (x.expiry - self.time).days <= self.weekly_dte_max
            and x.greeks and x.greeks.delta is not None
            and x.bid_price > 0 and x.ask_price > 0
        ]
        if not calls:
            return

        spot = float(self.securities[equity].price)
        otm_calls = [c for c in calls if c.strike >= spot]
        pool = otm_calls if otm_calls else calls

        call = min(pool, key=lambda x: abs(x.greeks.delta - 0.25))

        if not self._spread_ok(call.bid_price, call.ask_price):
            return

        contracts = shares // 100
        limit_px = self._mid(call.bid_price, call.ask_price)
        if limit_px <= 0:
            return

        self.option_tags[call.symbol] = "COVERED_CALL"

        cc_key = f"{state['ticker']}_WHEEL_{self.time.date()}"
        self._place_limit(
            call.symbol, -contracts, limit_px,
            action="OPEN", reason="COVERED_CALL", key=cc_key, leg="COVERED_CALL",
            note=f"strike={call.strike} Δ={call.greeks.delta:.2f} DTE={(call.expiry - self.time).days}"
        )

        state["wheel_cc_symbol"] = call.symbol

        credit = limit_px * 100 * contracts
        self._append_trade_row(
            symbol=equity,
            event="SELL_CC",
            key=cc_key,
            detail=f"strike={call.strike} Δ={call.greeks.delta:.2f} DTE={(call.expiry - self.time).days}",
            qty=contracts,
            price=limit_px,
            credit_debit=credit
        )
        self.debug(f"SELL CC {state['ticker']}: qty={contracts} strike={call.strike} prem={limit_px:.2f}")

    # -----------------------------
    # ObjectStore CSV helpers
    # -----------------------------
    def _ensure_objectstore_csv(self, key: str, header: str):
        try:
            if not self.object_store.contains_key(key):
                self.object_store.save(key, header)
                return
            existing = self.object_store.read(key)
            if not existing or len(existing) < len(header):
                self.object_store.save(key, header)
        except Exception as e:
            self.debug(f"[OS] ensure failed {key}: {e}")

    def _append_daily_row(self, state: dict):
        today = self.time.date()
        now = self.time.strftime("%H:%M:%S")

        equity = float(self.portfolio.total_portfolio_value)
        cash = float(self.portfolio.cash)
        margin_used = float(self.portfolio.total_margin_used)

        equity_symbol = state["equity"]
        spot = float(self.securities[equity_symbol].price)
        shares = int(self.portfolio[equity_symbol].quantity)
        avg_price = float(self.portfolio[equity_symbol].average_price) if shares != 0 else 0.0

        opt_unrl = float(self._compute_options_unrealized(equity_symbol))
        stk_unrl = float(self._compute_stock_unrealized(equity_symbol))
        opt_real = float(self.pnl[equity_symbol].get("options_realized", 0.0))

        hold = self.portfolio[equity_symbol]
        stk_real = float(hold.profit - hold.unrealized_profit) if hold.invested or hold.profit != 0 else 0.0

        open_strangles = int(len(state["open_positions"]))
        wheel_active = int(1 if state["wheel_active"] else 0)
        wheel_cc = str(state["wheel_cc_symbol"]) if state["wheel_cc_symbol"] else ""

        self._daily_rows.append(
            f"{today},{now},{state['ticker']},{equity:.2f},{cash:.2f},{margin_used:.2f},"
            f"{spot:.2f},{shares},{avg_price:.4f},"
            f"{opt_unrl:.2f},{stk_unrl:.2f},{opt_real:.2f},{stk_real:.2f},"
            f"{open_strangles},{wheel_active},{wheel_cc}\n"
        )

    def _append_trade_row(self, event: str, key: str = "", detail: str = "",
                          symbol=None,
                          qty: float = 0.0, price: float = 0.0,
                          credit_debit: float = 0.0, est_pnl: float = 0.0):
        today = self.time.date()
        now = self.time.strftime("%H:%M:%S")
        equity = float(self.portfolio.total_portfolio_value)

        if symbol is None:
            symbol = self.primary_symbol

        spot = float(self.securities[symbol].price)
        shares = int(self.portfolio[symbol].quantity)
        ticker = self.symbol_state[symbol]["ticker"] if symbol in self.symbol_state else ""

        self._trade_rows.append(
            f"{today},{now},{ticker},{event},{key},\"{detail}\",{qty:.0f},{price:.4f},"
            f"{credit_debit:.2f},{est_pnl:.2f},{equity:.2f},{spot:.2f},{shares}\n"
        )

    def _flush_ledgers_to_objectstore(self):
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
        try:
            for state in self.symbol_state.values():
                self._append_daily_row(state)
            self._flush_ledgers_to_objectstore()
            self.debug(f"[OS] Saved: {self.os_daily_key}, {self.os_trade_key}")
        except Exception as e:
            self.debug(f"[OS] end flush failed: {e}")

    # -----------------------------
    # Explicit close path (no Liquidate) — used in force-close & full-close
    # -----------------------------
    def _safe_mid_or_fallback(self, sec, fallback: float) -> float:
        mid = self._mid(sec.bid_price, sec.ask_price)
        if mid > 0:
            return mid
        if sec.ask_price and sec.ask_price > 0:
            return float(sec.ask_price)
        return float(fallback)

    def _close_option_leg_explicit(self, symbol, *, action: str, reason: str, key: str, leg: str,
                                   est_pnl: float = 0.0, note: str = ""):
        hold = self.portfolio[symbol]
        if not hold.invested:
            return None

        sec = self.securities[symbol]
        qty_abs = abs(int(hold.quantity))
        if qty_abs <= 0:
            return None

        # short -> BUY to close; long -> SELL to close
        if hold.quantity < 0:
            fallback = float(sec.ask_price) if sec.ask_price and sec.ask_price > 0 else float(sec.price)
            limit_px = self._safe_mid_or_fallback(sec, fallback)
            return self._place_limit(
                symbol, +qty_abs, limit_px,
                action=action, reason=reason, key=key, leg=leg,
                est_pnl=est_pnl, note=note
            )
        else:
            mid = self._mid(sec.bid_price, sec.ask_price)
            limit_px = mid if mid > 0 else (float(sec.bid_price) if sec.bid_price and sec.bid_price > 0 else float(sec.price))
            return self._place_limit(
                symbol, -qty_abs, limit_px,
                action=action, reason=reason, key=key, leg=leg,
                est_pnl=est_pnl, note=note
            )

    def _close_strangle_explicit(self, key: str, pos: dict, *, action: str, reason: str,
                                 est_pnl: float = 0.0, note: str = ""):
        call_sym = pos["call"]
        put_sym = pos["put"]

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
