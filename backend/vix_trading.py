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

# SVIX: NO lo tocamos aquí (si quieres, se parametriza igual, pero no lo cambio)
# Lo dejo definido para no romper el motor si lo usas:
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
        # guardamos ISO date para columnas date; para timestamp usamos isoformat
        if x.tzinfo is not None:
            x = x.tz_convert(None)
        return x.isoformat()

    # dates python / datetime python
    if hasattr(x, "isoformat"):
        try:
            return x.isoformat()
        except Exception:
            return str(x)

    # numpy / pandas scalars
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


def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index)
    return df[col]


def _get_ohlc_for_ticker(row: pd.Series, ticker: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    t = (ticker or "").upper().strip()
    prefix = "uvix" if t == "UVIX" else "svix"
    o = _num(row.get(f"{prefix}_open"))
    h = _num(row.get(f"{prefix}_high"))
    l = _num(row.get(f"{prefix}_low"))
    c = _num(row.get(f"{prefix}_close"))
    return o, h, l, c


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
            # quitamos tz si viene con tz (para exportar a Excel sin error)
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
# Motor principal
# ============================================================

def run_vix_execution() -> None:
    """
    Motor: lee vix_daily + posiciones y aplica reglas.
    - No rompe SVIX: solo aplica reglas si hay posición OPEN.
    - Crea nuevas posiciones cuando estado = UVIX o SVIX, si no hay OPEN.
    """

    daily = fetch_vix_daily()
    if daily.empty:
        return

    # Aseguramos fecha y orden
    if "fecha" not in daily.columns:
        raise RuntimeError("vix_daily no tiene columna 'fecha'")
    daily = daily.copy()
    daily["fecha"] = pd.to_datetime(daily["fecha"], errors="coerce").dt.normalize()
    daily = daily.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

    # Posiciones actuales
    pos_df = fetch_vix_positions(limit=2000)
    if pos_df.empty:
        pos_df = pd.DataFrame(columns=[
            "id","ticker","status","entry_signal_date","entry_date","entry_price","qty","capital_usd",
            "stop_pct","tp1_pct","trailing_pct","tp1_taken","trailing_active",
            "hard_stop_price","tp1_price","trail_price","high_watermark",
            "exit_date","exit_price","pl_usd","pl_pct","notes",
        ])

    # Buscar si hay OPEN por ticker
    open_pos = {}
    if "status" in pos_df.columns and "ticker" in pos_df.columns:
        for tkr in ["UVIX", "SVIX"]:
            w = pos_df[(pos_df["ticker"] == tkr) & (pos_df["status"] == "OPEN")].copy()
            if not w.empty:
                # si hubiese más de una por error, cogemos la más reciente
                w = w.sort_values("entry_date", ascending=False)
                open_pos[tkr] = w.iloc[0].to_dict()

    # Recorremos días y aplicamos:
    # 1) entradas al OPEN del día siguiente (basado en señal del día anterior)
    # 2) gestión de stops/tp/trailing en días posteriores
    for i in range(1, len(daily)):
        prev = daily.iloc[i - 1]
        cur = daily.iloc[i]

        prev_state = str(prev.get("estado", "NEUTRAL") or "NEUTRAL").upper().strip()
        cur_date = _to_date(cur.get("fecha"))
        if cur_date is None:
            continue

        # =========================
        # ENTRADAS (si no hay OPEN)
        # =========================
        for tkr in ["UVIX", "SVIX"]:
            if tkr in open_pos:
                continue  # ya hay una abierta

            if prev_state != tkr:
                continue

            # Entramos al open de hoy (cur)
            o, h, l, c = _get_ohlc_for_ticker(cur, tkr)
            if o is None:
                # si no hay open, no se puede simular entrada
                continue

            rules = RULES_BY_TICKER.get(tkr)
            if rules is None:
                continue

            entry_price = float(o)
            capital_usd = 10000.0  # si tú lo calculas fuera, aquí podrías leer NAV; lo dejamos fijo por sistema
            qty = capital_usd / entry_price if entry_price > 0 else 0.0

            hard_stop_price = entry_price * (1.0 - rules.stop_pct)
            tp1_price = entry_price * (1.0 + rules.tp1_pct)

            payload = {
                "ticker": tkr,
                "status": "OPEN",
                "entry_signal_date": _to_date(prev.get("fecha")).date().isoformat() if _to_date(prev.get("fecha")) is not None else None,
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

        # =========================
        # GESTIÓN DE POSICIONES OPEN
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

            days_in_trade = (cur_date - entry_date).days  # sesiones aprox.
            allow_exits = days_in_trade >= rules.min_hold_days

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

            # actualizar high watermark si hay high
            if h is not None:
                high_watermark = max(high_watermark, float(h))

            # calcular trail_price si trailing activo
            trail_price = _num(pos.get("trail_price"))
            if trailing_active:
                trail_price = high_watermark * (1.0 - trailing_pct)

            # ------------------------------------------------------------
            # 1) HARD STOP (excepción: se permite aunque no haya 48h)
            # ------------------------------------------------------------
            exit_reason = None
            exit_price = None

            if hard_stop_price is not None:
                # gap down: si open ya está por debajo del stop => ejecuta a open
                if o is not None and float(o) <= float(hard_stop_price):
                    exit_reason = "HARD_STOP_GAP"
                    exit_price = float(o)
                # intradía toca stop
                elif l is not None and float(l) <= float(hard_stop_price):
                    exit_reason = "HARD_STOP"
                    exit_price = float(hard_stop_price)

            # ------------------------------------------------------------
            # 2) TP1 (solo si allow_exits)
            # ------------------------------------------------------------
            tp1_exec = False
            if exit_reason is None and allow_exits and (not tp1_taken) and tp1_price is not None:
                if h is not None and float(h) >= float(tp1_price):
                    tp1_exec = True

            # ------------------------------------------------------------
            # 3) Trailing stop (solo si allow_exits y trailing activo)
            # ------------------------------------------------------------
            if exit_reason is None and allow_exits and trailing_active and trail_price is not None:
                # gap: si open <= trail => salida a open
                if o is not None and float(o) <= float(trail_price):
                    exit_reason = "TRAIL_GAP"
                    exit_price = float(o)
                elif l is not None and float(l) <= float(trail_price):
                    exit_reason = "TRAIL"
                    exit_price = float(trail_price)

            # ------------------------------------------------------------
            # 4) Time-stop si no hay TP1 tras X días
            # ------------------------------------------------------------
            if exit_reason is None and allow_exits and (not tp1_taken):
                if days_in_trade >= rules.max_hold_days_no_tp1:
                    if c is not None and entry_price > 0:
                        if float(c) < entry_price * (1.0 + rules.min_gain_to_keep_after_maxhold):
                            exit_reason = "TIME_STOP"
                            exit_price = float(c)

            # ------------------------------------------------------------
            # 5) Regime exit: si UVIX se apaga 2 días seguidos (solo allow_exits)
            # ------------------------------------------------------------
            if exit_reason is None and allow_exits:
                # comprobamos el estado de los 2 últimos cierres (prev y cur)
                cur_state = str(cur.get("estado", "NEUTRAL") or "NEUTRAL").upper().strip()
                prev2_state = str(prev.get("estado", "NEUTRAL") or "NEUTRAL").upper().strip()
                if tkr == "UVIX":
                    if (prev2_state != "UVIX") and (cur_state != "UVIX"):
                        # salida a open del día actual (cur)
                        if o is not None:
                            exit_reason = "REGIME_OFF"
                            exit_price = float(o)

            # ------------------------------------------------------------
            # Aplicar TP1 (si toca)
            # ------------------------------------------------------------
            if tp1_exec:
                # vendemos rules.tp1_take del qty
                qty_left = qty * (1.0 - rules.tp1_take)
                _update_position(int(pos["id"]), {
                    "tp1_taken": True,
                    "trailing_active": True,          # activamos trailing tras TP1
                    "high_watermark": high_watermark,
                    "trail_price": high_watermark * (1.0 - trailing_pct),
                    "qty": qty_left,                  # dejamos solo la mitad
                    "notes": (str(pos.get("notes") or "") + f" | TP1 hit @ {tp1_price:.4f}").strip(),
                })
                # refrescamos el dict local
                pos["tp1_taken"] = True
                pos["trailing_active"] = True
                pos["high_watermark"] = high_watermark
                pos["trail_price"] = high_watermark * (1.0 - trailing_pct)
                pos["qty"] = qty_left

                # IMPORTANTE: en el mismo día, NO forzamos salida del resto por trailing.
                # Seguimos al siguiente día.

            # ------------------------------------------------------------
            # Cerrar posición (si hay exit_reason)
            # ------------------------------------------------------------
            if exit_reason is not None and exit_price is not None:
                # P/L aproximado sobre lo que quede (qty actual)
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

                # sacamos de open_pos
                open_pos.pop(tkr, None)
                continue

            # ------------------------------------------------------------
            # Si sigue OPEN: actualizar watermark/trail si aplica
            # ------------------------------------------------------------
            patch = {"high_watermark": high_watermark}
            if trailing_active:
                patch["trail_price"] = high_watermark * (1.0 - trailing_pct)
            _update_position(int(pos["id"]), patch)

            # refresco local
            pos["high_watermark"] = high_watermark
            if trailing_active:
                pos["trail_price"] = patch["trail_price"]


# alias (por si tu main importaba otro nombre en alguna iteración)
run_vix_execution = run_vix_execution
