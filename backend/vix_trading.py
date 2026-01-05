# backend/vix_trading.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .supabase_client import supabase
from .vix import fetch_vix_daily  # usamos tu tabla vix_daily ya calculada


# ============================================================
# Config de ejecución / reglas
# ============================================================

@dataclass
class TradeRules:
    stop_pct: float
    tp1_pct: float
    trailing_pct: float
    tp1_take: float = 0.50          # % de posición que se vende en TP1
    min_hold_days: int = 2          # 48h ~ 2 sesiones
    max_hold_days_no_tp1: int = 5   # time-stop si no hay TP1
    min_gain_to_keep_after_maxhold: float = 0.02  # +2% mínimo si no hay TP1


# Payoff-focused UVIX
UVIX_RULES = TradeRules(
    stop_pct=0.10,        # -10%
    tp1_pct=0.10,         # +10%
    trailing_pct=0.06,    # 6%
    tp1_take=0.50,
    min_hold_days=2,
    max_hold_days_no_tp1=5,
    min_gain_to_keep_after_maxhold=0.02,
)

# Dejamos SVIX definido por compatibilidad, pero el motor NO lo usa en esta versión.
SVIX_RULES = TradeRules(
    stop_pct=0.04,
    tp1_pct=0.06,
    trailing_pct=0.04,
    tp1_take=0.50,
    min_hold_days=2,
    max_hold_days_no_tp1=10,
    min_gain_to_keep_after_maxhold=0.01,
)

RULES_BY_TICKER: Dict[str, TradeRules] = {
    "UVIX": UVIX_RULES,
    "SVIX": SVIX_RULES,
}

# ✅ En esta iteración SOLO operamos UVIX (así no “rompemos” SVIX ni generamos duplicados SVIX)
TRADE_TICKERS = ["UVIX"]


# ============================================================
# Helpers
# ============================================================

def _json_sanitize_value(x: Any) -> Any:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, pd.Timestamp):
        # quitamos tz para no dar problemas (Excel / PostgREST)
        try:
            if x.tzinfo is not None:
                x = x.tz_convert(None)
        except Exception:
            try:
                x = x.tz_localize(None)
            except Exception:
                pass
        return x.isoformat()

    if hasattr(x, "isoformat"):
        try:
            return x.isoformat()
        except Exception:
            return str(x)

    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
    except Exception:
        pass

    return x


def _sanitize_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _json_sanitize_value(v) for k, v in d.items()}


def _to_date(x: Any) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    t = pd.to_datetime(x, errors="coerce")
    if pd.isna(t):
        return None
    try:
        if t.tzinfo is not None:
            t = t.tz_convert(None)
    except Exception:
        try:
            t = t.tz_localize(None)
        except Exception:
            pass
    return t.normalize()


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


def _get_ohlc_for_ticker(row: pd.Series, ticker: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    t = (ticker or "").upper().strip()
    prefix = "uvix" if t == "UVIX" else "svix"
    o = _num(row.get(f"{prefix}_open"))
    h = _num(row.get(f"{prefix}_high"))
    l = _num(row.get(f"{prefix}_low"))
    c = _num(row.get(f"{prefix}_close"))
    return o, h, l, c


def _key_entry(ticker: str, entry_date: Any) -> Optional[Tuple[str, str]]:
    """
    Clave idempotente (ticker, entry_date ISO YYYY-MM-DD)
    """
    t = (ticker or "").upper().strip()
    d = _to_date(entry_date)
    if not t or d is None:
        return None
    return (t, d.date().isoformat())


# ============================================================
# Supabase I/O: vix_positions
# ============================================================

def fetch_vix_positions(limit: int = 500) -> pd.DataFrame:
    resp = supabase.table("vix_positions").select("*").order("entry_date", desc=True).limit(limit).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)

    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)

    # normalizamos fechas a pandas (naive)
    for c in ["entry_signal_date", "entry_date", "exit_date", "created_at", "updated_at"]:
        if c in df.columns and not df.empty:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            try:
                df[c] = df[c].dt.tz_localize(None)
            except Exception:
                pass
    return df


def _insert_position(payload: Dict[str, Any]) -> int:
    payload = _sanitize_payload(payload)
    resp = supabase.table("vix_positions").insert(payload).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)
    data = getattr(resp, "data", None) or []
    if data and isinstance(data, list) and "id" in data[0]:
        return int(data[0]["id"])
    return 1


def _update_position(pos_id: int, patch: Dict[str, Any]) -> None:
    patch = _sanitize_payload(patch)
    resp = supabase.table("vix_positions").update(patch).eq("id", int(pos_id)).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)


# ============================================================
# Motor principal (SOLO UVIX + idempotencia)
# ============================================================

def run_vix_execution() -> None:
    """
    Motor UVIX:
    - Lee vix_daily + vix_positions.
    - Crea UVIX cuando estado del día anterior == UVIX y NO existe ya una posición UVIX con esa entry_date.
    - Gestiona la posición OPEN con: hard stop (siempre), TP1 + trailing (tras 48h), time-stop, regime-off (2 días).
    """

    daily = fetch_vix_daily()
    if daily.empty:
        return

    if "fecha" not in daily.columns:
        raise RuntimeError("vix_daily no tiene columna 'fecha'")

    daily = daily.copy()
    daily["fecha"] = pd.to_datetime(daily["fecha"], errors="coerce").dt.normalize()
    daily = daily.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

    # Cargamos posiciones existentes
    pos_df = fetch_vix_positions(limit=5000)
    if pos_df.empty:
        pos_df = pd.DataFrame(columns=[
            "id","ticker","status","entry_signal_date","entry_date","entry_price","qty","capital_usd",
            "stop_pct","tp1_pct","trailing_pct","tp1_taken","trailing_active",
            "hard_stop_price","tp1_price","trail_price","high_watermark",
            "exit_date","exit_price","pl_usd","pl_pct","notes",
        ])

    # ✅ Set de claves existentes para NO duplicar entradas (idempotencia)
    existing_keys = set()
    if not pos_df.empty and "ticker" in pos_df.columns and "entry_date" in pos_df.columns:
        for _, r in pos_df.iterrows():
            k = _key_entry(str(r.get("ticker")), r.get("entry_date"))
            if k is not None:
                existing_keys.add(k)

    # OPEN por ticker (solo UVIX en esta versión)
    open_pos: Dict[str, Dict[str, Any]] = {}
    if not pos_df.empty and "status" in pos_df.columns and "ticker" in pos_df.columns:
        for tkr in TRADE_TICKERS:
            w = pos_df[(pos_df["ticker"] == tkr) & (pos_df["status"] == "OPEN")].copy()
            if not w.empty:
                w = w.sort_values("entry_date", ascending=False)
                open_pos[tkr] = w.iloc[0].to_dict()

    # Recorremos días
    for i in range(1, len(daily)):
        prev = daily.iloc[i - 1]
        cur = daily.iloc[i]

        prev_state = str(prev.get("estado", "NEUTRAL") or "NEUTRAL").upper().strip()
        cur_date = _to_date(cur.get("fecha"))
        if cur_date is None:
            continue

        # =========================
        # ENTRADA UVIX (si no hay OPEN)
        # =========================
        for tkr in TRADE_TICKERS:
            if tkr in open_pos:
                continue

            if prev_state != tkr:
                continue

            # ✅ idempotencia: si ya hay posición para ese día, NO insertamos
            k = _key_entry(tkr, cur_date)
            if k is not None and k in existing_keys:
                continue

            o, h, l, c = _get_ohlc_for_ticker(cur, tkr)
            if o is None:
                # sin open no podemos simular entrada
                continue

            rules = RULES_BY_TICKER.get(tkr)
            if rules is None:
                continue

            entry_price = float(o)
            capital_usd = 10000.0  # (mantengo tu diseño)
            qty = capital_usd / entry_price if entry_price > 0 else 0.0

            hard_stop_price = entry_price * (1.0 - rules.stop_pct)
            tp1_price = entry_price * (1.0 + rules.tp1_pct)

            prev_fecha = _to_date(prev.get("fecha"))
            payload = {
                "ticker": tkr,
                "status": "OPEN",
                "entry_signal_date": (prev_fecha.date().isoformat() if prev_fecha is not None else None),
                "entry_date": cur_date.date().isoformat(),
                "entry_price": entry_price,
                "qty": qty,
                "capital_usd": capital_usd,
                "stop_pct": rules.stop_pct,
                "tp1_pct": rules.tp1_pct,
                "trailing_pct": rules.trailing_pct,
                "tp1_taken": False,
                "trailing_active": False,
                "hard_stop_price": hard_stop_price,
                "tp1_price": tp1_price,
                "trail_price": None,
                "high_watermark": entry_price,
                "notes": f"Auto-entry. prev_estado={prev_state}",
            }

            new_id = _insert_position(payload)
            payload["id"] = new_id
            open_pos[tkr] = payload
            if k is not None:
                existing_keys.add(k)

        # =========================
        # GESTIÓN UVIX OPEN
        # =========================
        for tkr, pos in list(open_pos.items()):
            if str(pos.get("status", "")).upper().strip() != "OPEN":
                continue

            entry_date = _to_date(pos.get("entry_date"))
            if entry_date is None:
                continue

            rules = RULES_BY_TICKER.get(tkr)
            if rules is None:
                continue

            days_in_trade = (cur_date - entry_date).days
            allow_exits = days_in_trade >= rules.min_hold_days  # 48h

            entry_price = float(pos.get("entry_price") or 0.0)
            qty = float(pos.get("qty") or 0.0)

            hard_stop_price = _num(pos.get("hard_stop_price"))
            tp1_price = _num(pos.get("tp1_price"))
            trailing_pct = float(pos.get("trailing_pct") or rules.trailing_pct)

            tp1_taken = bool(pos.get("tp1_taken") is True)
            trailing_active = bool(pos.get("trailing_active") is True)
            high_watermark = _num(pos.get("high_watermark")) or entry_price

            o, h, l, c = _get_ohlc_for_ticker(cur, tkr)
            if o is None and h is None and l is None and c is None:
                continue

            if h is not None:
                high_watermark = max(high_watermark, float(h))

            trail_price = _num(pos.get("trail_price"))
            if trailing_active:
                trail_price = high_watermark * (1.0 - trailing_pct)

            exit_reason = None
            exit_price = None

            # 1) HARD STOP (se permite incluso sin 48h)
            if hard_stop_price is not None:
                if o is not None and float(o) <= float(hard_stop_price):
                    exit_reason = "HARD_STOP_GAP"
                    exit_price = float(o)
                elif l is not None and float(l) <= float(hard_stop_price):
                    exit_reason = "HARD_STOP"
                    exit_price = float(hard_stop_price)

            # 2) TP1 (solo si allow_exits)
            tp1_exec = False
            if exit_reason is None and allow_exits and (not tp1_taken) and tp1_price is not None:
                if h is not None and float(h) >= float(tp1_price):
                    tp1_exec = True

            # 3) Trailing stop (solo si allow_exits y trailing activo)
            if exit_reason is None and allow_exits and trailing_active and trail_price is not None:
                if o is not None and float(o) <= float(trail_price):
                    exit_reason = "TRAIL_GAP"
                    exit_price = float(o)
                elif l is not None and float(l) <= float(trail_price):
                    exit_reason = "TRAIL"
                    exit_price = float(trail_price)

            # 4) Time-stop si no hay TP1 tras X días
            if exit_reason is None and allow_exits and (not tp1_taken):
                if days_in_trade >= rules.max_hold_days_no_tp1:
                    if c is not None and entry_price > 0:
                        if float(c) < entry_price * (1.0 + rules.min_gain_to_keep_after_maxhold):
                            exit_reason = "TIME_STOP"
                            exit_price = float(c)

            # 5) Regime exit: UVIX se apaga 2 días seguidos (solo allow_exits)
            if exit_reason is None and allow_exits and tkr == "UVIX":
                cur_state = str(cur.get("estado", "NEUTRAL") or "NEUTRAL").upper().strip()
                prev2_state = str(prev.get("estado", "NEUTRAL") or "NEUTRAL").upper().strip()
                if (prev2_state != "UVIX") and (cur_state != "UVIX"):
                    if o is not None:
                        exit_reason = "REGIME_OFF"
                        exit_price = float(o)

            # Aplicar TP1 (si toca)
            if tp1_exec:
                qty_left = qty * (1.0 - rules.tp1_take)
                _update_position(int(pos["id"]), {
                    "tp1_taken": True,
                    "trailing_active": True,
                    "high_watermark": high_watermark,
                    "trail_price": high_watermark * (1.0 - trailing_pct),
                    "qty": qty_left,
                    "notes": (str(pos.get("notes") or "") + f" | TP1 hit @ {tp1_price:.4f}").strip(),
                })
                pos["tp1_taken"] = True
                pos["trailing_active"] = True
                pos["high_watermark"] = high_watermark
                pos["trail_price"] = high_watermark * (1.0 - trailing_pct)
                pos["qty"] = qty_left

            # Cerrar posición (si hay exit)
            if exit_reason is not None and exit_price is not None:
                qty_now = float(pos.get("qty") or 0.0)
                pl_usd = (float(exit_price) - entry_price) * qty_now
                pl_pct = (float(exit_price) / entry_price - 1.0) if entry_price > 0 else None

                _update_position(int(pos["id"]), {
                    "status": "CLOSED",
                    "exit_date": cur_date.date().isoformat(),
                    "exit_price": float(exit_price),
                    "pl_usd": float(pl_usd),
                    "pl_pct": float(pl_pct) if pl_pct is not None else None,
                    "high_watermark": high_watermark,
                    "trail_price": trail_price,
                    "notes": (str(pos.get("notes") or "") + f" | EXIT {exit_reason} @ {exit_price:.4f}").strip(),
                })

                open_pos.pop(tkr, None)
                continue

            # Si sigue OPEN: actualizar watermark/trail
            patch = {"high_watermark": high_watermark}
            if trailing_active:
                patch["trail_price"] = high_watermark * (1.0 - trailing_pct)

            _update_position(int(pos["id"]), patch)

            pos["high_watermark"] = high_watermark
            if trailing_active and "trail_price" in patch:
                pos["trail_price"] = patch["trail_price"]


# alias (por compatibilidad con imports anteriores)
run_vix_execution = run_vix_execution
