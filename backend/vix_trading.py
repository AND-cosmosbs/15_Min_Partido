# backend/vix_trading.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable

import time
import pandas as pd

from .supabase_client import supabase
from .vix import fetch_vix_daily


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
    trailing_pct=0.08,    # ✅ 8% (captura explosiones más largas)
    tp1_take=0.50,
    min_hold_days=2,
    max_hold_days_no_tp1=5,
    min_gain_to_keep_after_maxhold=0.02,
)

# SVIX: NO lo tocamos (solo definido para no romper el motor si lo usas)
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


# ============================================================
# Helpers: sanitizado / tipos
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
        # guardamos naive ISO (Excel no soporta tz)
        if x.tzinfo is not None:
            try:
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


# ============================================================
# Supabase: retry robusto (Errno 11 / transitorios)
# ============================================================

def _is_transient_error(e: Exception) -> bool:
    # OSError 11 = EAGAIN
    if isinstance(e, OSError) and getattr(e, "errno", None) == 11:
        return True

    msg = str(e).lower()
    transient_markers = [
        "resource temporarily unavailable",
        "timeout",
        "timed out",
        "temporarily",
        "connection reset",
        "connection aborted",
        "server disconnected",
        "bad gateway",
        "gateway timeout",
        "service unavailable",
        "too many requests",
        "rate limit",
    ]
    return any(m in msg for m in transient_markers)


def _sb_exec(call: Callable[[], Any], attempts: int = 7) -> Any:
    last = None
    for i in range(attempts):
        try:
            resp = call()
            if getattr(resp, "error", None):
                raise RuntimeError(resp.error)
            return resp
        except Exception as e:
            last = e
            if (not _is_transient_error(e)) or (i == attempts - 1):
                raise
            time.sleep(0.25 + i * 0.40)
    if last:
        raise last
    raise RuntimeError("Supabase call failed (unknown)")


# ============================================================
# Supabase I/O: vix_positions
# ============================================================

def fetch_vix_positions(limit: int = 500) -> pd.DataFrame:
    resp = _sb_exec(lambda: supabase.table("vix_positions").select("*").order("entry_date", desc=True).limit(limit).execute())
    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    for c in ["entry_signal_date", "entry_date", "exit_date", "created_at", "updated_at"]:
        if c in df.columns and not df.empty:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            try:
                df[c] = df[c].dt.tz_localize(None)
            except Exception:
                pass
    return df


def _find_position_by_ticker_entry_date(ticker: str, entry_date_iso: str) -> Optional[Dict[str, Any]]:
    """
    Blindaje anti-duplicado SIN índice único:
    - Busca si ya existe (ticker, entry_date).
    - Si existe, devuelve la fila.
    """
    t = (ticker or "").upper().strip()
    d = pd.to_datetime(entry_date_iso, errors="coerce")
    if pd.isna(d):
        return None
    d_iso = d.date().isoformat()

    resp = _sb_exec(
        lambda: supabase.table("vix_positions")
        .select("*")
        .eq("ticker", t)
        .eq("entry_date", d_iso)
        .limit(1)
        .execute()
    )
    rows = getattr(resp, "data", None) or []
    return rows[0] if rows else None


def _insert_position(payload: Dict[str, Any]) -> int:
    payload = _sanitize_payload(payload)
    resp = _sb_exec(lambda: supabase.table("vix_positions").insert(payload).execute())
    data = getattr(resp, "data", None) or []
    if data and isinstance(data, list) and "id" in data[0]:
        return int(data[0]["id"])
    return 1


def _update_position(pos_id: int, patch: Dict[str, Any]) -> None:
    patch = _sanitize_payload(patch)
    _sb_exec(lambda: supabase.table("vix_positions").update(patch).eq("id", int(pos_id)).execute())


# ============================================================
# Motor principal
# ============================================================

def run_vix_execution() -> None:
    """
    Motor: lee vix_daily + posiciones y aplica reglas.

    Garantías:
    - ✅ No abre si ya hay OPEN del ticker
    - ✅ No abre si falta OHLC open del ticker ese día
    - ✅ No duplica aunque NO exista índice único (ticker, entry_date)
    - ✅ Retry para Errno 11 y fallos transitorios
    - ✅ Reduce escrituras (solo update si cambia)
    """

    daily = fetch_vix_daily()
    if daily is None or daily.empty:
        return

    if "fecha" not in daily.columns:
        raise RuntimeError("vix_daily no tiene columna 'fecha'")

    daily = daily.copy()
    daily["fecha"] = pd.to_datetime(daily["fecha"], errors="coerce").dt.normalize()
    daily = daily.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

    pos_df = fetch_vix_positions(limit=2000)
    if pos_df.empty:
        pos_df = pd.DataFrame(columns=[
            "id", "ticker", "status",
            "entry_signal_date", "entry_date", "entry_price", "qty", "capital_usd",
            "stop_pct", "tp1_pct", "trailing_pct",
            "tp1_taken", "trailing_active",
            "hard_stop_price", "tp1_price", "trail_price", "high_watermark",
            "exit_date", "exit_price", "pl_usd", "pl_pct", "notes",
        ])

    # OPEN por ticker (solo 1 permitido)
    open_pos: Dict[str, Dict[str, Any]] = {}
    if not pos_df.empty and ("ticker" in pos_df.columns) and ("status" in pos_df.columns):
        for tkr in ["UVIX", "SVIX"]:
            w = pos_df[(pos_df["ticker"] == tkr) & (pos_df["status"] == "OPEN")].copy()
            if not w.empty:
                w = w.sort_values("entry_date", ascending=False)
                open_pos[tkr] = w.iloc[0].to_dict()

    for i in range(1, len(daily)):
        prev = daily.iloc[i - 1]
        cur = daily.iloc[i]

        prev_state = str(prev.get("estado", "NEUTRAL") or "NEUTRAL").upper().strip()
        cur_date = _to_date(cur.get("fecha"))
        if cur_date is None:
            continue

        # =====================================================
        # ENTRADAS
        # =====================================================
        for tkr in ["UVIX", "SVIX"]:
            tkr_u = tkr.upper()

            # 1) si ya hay OPEN => no abrimos
            if tkr_u in open_pos:
                continue

            # 2) señal del día anterior
            if prev_state != tkr_u:
                continue

            # 3) necesitamos open del ticker hoy
            o, h, l, c = _get_ohlc_for_ticker(cur, tkr_u)
            if o is None:
                continue

            rules = RULES_BY_TICKER.get(tkr_u)
            if rules is None:
                continue

            # 4) blindaje anti-duplicado sin índice: (ticker, entry_date)
            entry_date_iso = cur_date.date().isoformat()
            existing = _find_position_by_ticker_entry_date(tkr_u, entry_date_iso)
            if existing is not None:
                # Si existe y está OPEN, lo adoptamos como open_pos
                if str(existing.get("status", "")).upper().strip() == "OPEN":
                    open_pos[tkr_u] = existing
                # si existe CLOSED, NO volvemos a crear nada ese mismo día
                continue

            entry_price = float(o)
            capital_usd = 10000.0
            qty = capital_usd / entry_price if entry_price > 0 else 0.0

            hard_stop_price = entry_price * (1.0 - rules.stop_pct)
            tp1_price = entry_price * (1.0 + rules.tp1_pct)

            prev_date = _to_date(prev.get("fecha"))

            payload = {
                "ticker": tkr_u,
                "status": "OPEN",
                "entry_signal_date": prev_date.date().isoformat() if prev_date is not None else None,
                "entry_date": entry_date_iso,
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
            open_pos[tkr_u] = payload

        # =====================================================
        # GESTIÓN OPEN
        # =====================================================
        for tkr_u, pos in list(open_pos.items()):
            if str(pos.get("status", "")).upper().strip() != "OPEN":
                continue

            rules = RULES_BY_TICKER.get(tkr_u)
            if rules is None:
                continue

            entry_date = _to_date(pos.get("entry_date"))
            if entry_date is None:
                continue

            days_in_trade = (cur_date - entry_date).days
            allow_exits = days_in_trade >= rules.min_hold_days

            entry_price = float(pos.get("entry_price") or 0.0)
            qty = float(pos.get("qty") or 0.0)

            hard_stop_price = _num(pos.get("hard_stop_price"))
            tp1_price = _num(pos.get("tp1_price"))
            trailing_pct = float(pos.get("trailing_pct") or rules.trailing_pct)

            tp1_taken = bool(pos.get("tp1_taken") is True)
            trailing_active = bool(pos.get("trailing_active") is True)
            high_watermark = _num(pos.get("high_watermark")) or entry_price

            o, h, l, c = _get_ohlc_for_ticker(cur, tkr_u)
            if o is None and h is None and l is None and c is None:
                continue

            # watermark
            if h is not None:
                high_watermark = max(high_watermark, float(h))

            trail_price = _num(pos.get("trail_price"))
            if trailing_active:
                trail_price = high_watermark * (1.0 - trailing_pct)

            exit_reason = None
            exit_price = None

            # 1) HARD STOP (aunque no haya 48h)
            if hard_stop_price is not None:
                if o is not None and float(o) <= float(hard_stop_price):
                    exit_reason = "HARD_STOP_GAP"
                    exit_price = float(o)
                elif l is not None and float(l) <= float(hard_stop_price):
                    exit_reason = "HARD_STOP"
                    exit_price = float(hard_stop_price)

            # 2) TP1
            tp1_exec = False
            if exit_reason is None and allow_exits and (not tp1_taken) and tp1_price is not None:
                if h is not None and float(h) >= float(tp1_price):
                    tp1_exec = True

            # 3) Trailing
            if exit_reason is None and allow_exits and trailing_active and trail_price is not None:
                if o is not None and float(o) <= float(trail_price):
                    exit_reason = "TRAIL_GAP"
                    exit_price = float(o)
                elif l is not None and float(l) <= float(trail_price):
                    exit_reason = "TRAIL"
                    exit_price = float(trail_price)

            # 4) Time-stop si no hay TP1
            if exit_reason is None and allow_exits and (not tp1_taken):
                if days_in_trade >= rules.max_hold_days_no_tp1:
                    if c is not None and entry_price > 0:
                        if float(c) < entry_price * (1.0 + rules.min_gain_to_keep_after_maxhold):
                            exit_reason = "TIME_STOP"
                            exit_price = float(c)

            # 5) Regime-off UVIX (2 días seguidos fuera)
            if exit_reason is None and allow_exits and tkr_u == "UVIX":
                cur_state = str(cur.get("estado", "NEUTRAL") or "NEUTRAL").upper().strip()
                prev2_state = str(prev.get("estado", "NEUTRAL") or "NEUTRAL").upper().strip()
                if (prev2_state != "UVIX") and (cur_state != "UVIX"):
                    if o is not None:
                        exit_reason = "REGIME_OFF"
                        exit_price = float(o)

            # TP1 aplica
            if tp1_exec:
                qty_left = qty * (1.0 - rules.tp1_take)
                new_trail = high_watermark * (1.0 - trailing_pct)

                _update_position(int(pos["id"]), {
                    "tp1_taken": True,
                    "trailing_active": True,
                    "high_watermark": high_watermark,
                    "trail_price": new_trail,
                    "qty": qty_left,
                    "notes": (str(pos.get("notes") or "") + f" | TP1 hit @ {tp1_price:.4f}").strip(),
                })

                pos["tp1_taken"] = True
                pos["trailing_active"] = True
                pos["high_watermark"] = high_watermark
                pos["trail_price"] = new_trail
                pos["qty"] = qty_left

            # Cerrar si toca
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

                open_pos.pop(tkr_u, None)
                continue

            # Si sigue OPEN: update SOLO si cambia
            patch: Dict[str, Any] = {}

            old_hw = _num(pos.get("high_watermark")) or entry_price
            if abs(high_watermark - old_hw) > 1e-9:
                patch["high_watermark"] = high_watermark
                pos["high_watermark"] = high_watermark

            if trailing_active:
                new_trail = high_watermark * (1.0 - trailing_pct)
                old_trail = _num(pos.get("trail_price"))
                if old_trail is None or abs(new_trail - old_trail) > 1e-9:
                    patch["trail_price"] = new_trail
                    pos["trail_price"] = new_trail

            if patch:
                _update_position(int(pos["id"]), patch)


# alias por compatibilidad
run_vix_execution = run_vix_execution
