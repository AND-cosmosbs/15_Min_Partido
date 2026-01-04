# backend/vix_trading.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from .supabase_client import supabase
from .vix import fetch_vix_daily  # usamos tu vix.py existente (señales + paginación)


# ============================================================
# CONFIG CERRADA (la que ya venías usando)
# ============================================================

@dataclass
class TradeRules:
    # Mínimo 48h salvo excepción (STOP siempre manda)
    min_hold_days: int = 2

    # Parcial TP1
    tp1_sell_fraction: float = 0.50  # 50%

    # Reglas % por activo (diferentes)
    svix_stop_pct: float = 0.04
    svix_tp1_pct: float = 0.04   # TP1 más bajo (ajustable)
    svix_trailing_pct: float = 0.03  # trailing más cerca (ajustable)

    uvix_stop_pct: float = 0.08
    uvix_tp1_pct: float = 0.10   # objetivo ~10%
    uvix_trailing_pct: float = 0.06


DEFAULT_RULES = TradeRules()


def _now_iso_date() -> str:
    return pd.Timestamp.utcnow().date().isoformat()


def _json_sanitize_value(x: Any) -> Any:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, pd.Timestamp):
        return x.date().isoformat()
    if hasattr(x, "isoformat"):
        try:
            return x.isoformat()
        except Exception:
            return str(x)
    if isinstance(x, (int, float, str, bool)):
        return x
    return str(x)


def _sanitize_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _json_sanitize_value(v) for k, v in d.items()}


# ============================================================
# SUPABASE CRUD: vix_positions
# ============================================================

def fetch_vix_positions(limit: int = 200) -> pd.DataFrame:
    resp = (
        supabase.table("vix_positions")
        .select("*")
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )
    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)
    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    for c in ["entry_signal_date", "entry_date", "exit_date", "created_at", "updated_at"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def fetch_open_position() -> Optional[Dict[str, Any]]:
    resp = (
        supabase.table("vix_positions")
        .select("*")
        .eq("status", "OPEN")
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)
    data = getattr(resp, "data", None) or []
    return data[0] if data else None


def fetch_open_or_partial_position() -> Optional[Dict[str, Any]]:
    # Si estás PARTIAL también es posición viva
    resp = (
        supabase.table("vix_positions")
        .select("*")
        .in_("status", ["OPEN", "PARTIAL"])
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)
    data = getattr(resp, "data", None) or []
    return data[0] if data else None


def insert_position(payload: Dict[str, Any]) -> int:
    payload = _sanitize_payload(payload)
    resp = supabase.table("vix_positions").insert(payload).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)
    data = getattr(resp, "data", None) or []
    if data and isinstance(data, list) and "id" in data[0]:
        return int(data[0]["id"])
    return 1


def update_position(pos_id: int, patch: Dict[str, Any]) -> None:
    patch = dict(patch)
    patch["updated_at"] = _now_iso_date()
    patch = _sanitize_payload(patch)
    resp = supabase.table("vix_positions").update(patch).eq("id", int(pos_id)).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)


# ============================================================
# UTILIDADES TRADE
# ============================================================

def _rules_for_ticker(ticker: str, rules: TradeRules) -> Tuple[float, float, float]:
    t = (ticker or "").upper().strip()
    if t == "UVIX":
        return rules.uvix_stop_pct, rules.uvix_tp1_pct, rules.uvix_trailing_pct
    return rules.svix_stop_pct, rules.svix_tp1_pct, rules.svix_trailing_pct


def _ohlc_cols(ticker: str) -> Tuple[str, str, str, str]:
    p = (ticker or "").lower().strip()
    return f"{p}_open", f"{p}_high", f"{p}_low", f"{p}_close"


def _get_ohlc_row(daily_row: pd.Series, ticker: str) -> Dict[str, Optional[float]]:
    c_open, c_high, c_low, c_close = _ohlc_cols(ticker)
    out = {
        "open": daily_row.get(c_open),
        "high": daily_row.get(c_high),
        "low": daily_row.get(c_low),
        "close": daily_row.get(c_close),
    }
    # normaliza a float/None
    for k, v in list(out.items()):
        try:
            out[k] = float(v) if pd.notna(v) else None
        except Exception:
            out[k] = None
    return out


def _days_held(entry_date: Optional[pd.Timestamp], current_date: pd.Timestamp) -> int:
    if entry_date is None or pd.isna(entry_date):
        return 0
    try:
        return int((current_date.date() - entry_date.date()).days)
    except Exception:
        return 0


# ============================================================
# MOTOR “CERRADO” (sin subjetividad)
# - Entrada: señal de vix_daily (accion/estado)
# - Ejecución: al OPEN del día siguiente si hay datos OHLC
# - Gestión: stop / tp1 / trailing con HIGH/LOW diario
# - Restricción: mínimo 48h para TP1/trailing (STOP siempre manda)
# ============================================================

def _desired_ticker_from_signal(row: pd.Series) -> Optional[str]:
    estado = str(row.get("estado", "") or "").upper().strip()
    accion = str(row.get("accion", "") or "").upper().strip()

    if "NO DATA" in accion or "NO NEW" in accion:
        return None

    # Señales que implican estar largos
    if "SVIX" in estado:
        return "SVIX"
    if "UVIX" in estado:
        return "UVIX"

    return None


def run_vix_execution(
    rules: TradeRules = DEFAULT_RULES,
    capital_usd: float = 10000.0,
    max_rows: int = 5000,
) -> Dict[str, Any]:
    """
    Ejecuta gestión completa recorriendo vix_daily en orden temporal:
    - Si no hay posición viva, abre cuando haya señal (al open del día siguiente).
    - Si hay posición, aplica stop/tp1/trailing y evita flip inmediato.
    - Respeta 48h mínimo para TP1/trailing (stop siempre).
    """

    daily = fetch_vix_daily()
    if daily is None or daily.empty:
        return {"ok": False, "msg": "vix_daily vacío.", "opened": 0, "closed": 0, "partial": 0}

    # orden por fecha
    if "fecha" not in daily.columns:
        return {"ok": False, "msg": "vix_daily no tiene columna 'fecha'.", "opened": 0, "closed": 0, "partial": 0}

    daily = daily.copy()
    daily["fecha"] = pd.to_datetime(daily["fecha"], errors="coerce")
    daily = daily.dropna(subset=["fecha"]).sort_values("fecha").tail(max_rows)

    opened = 0
    closed = 0
    partial = 0

    pos = fetch_open_or_partial_position()

    # Recorremos día a día
    for i in range(len(daily) - 1):  # hasta penúltimo (porque entrada usa día siguiente open)
        row = daily.iloc[i]
        next_row = daily.iloc[i + 1]

        d = pd.to_datetime(row["fecha"])
        d_next = pd.to_datetime(next_row["fecha"])

        desired = _desired_ticker_from_signal(row)

        # --- Si NO hay posición viva: abrir al open del día siguiente si hay señal ---
        if pos is None:
            if desired is None:
                continue

            ohlc_next = _get_ohlc_row(next_row, desired)
            entry_px = ohlc_next["open"]
            if entry_px is None:
                continue  # no podemos ejecutar sin open

            stop_pct, tp1_pct, tr_pct = _rules_for_ticker(desired, rules)

            hard_stop = entry_px * (1.0 - stop_pct)
            tp1_price = entry_px * (1.0 + tp1_pct)

            payload = {
                "ticker": desired,
                "status": "OPEN",
                "entry_signal_date": d.date(),
                "entry_date": d_next.date(),
                "entry_price": float(entry_px),
                "qty": float(capital_usd / entry_px) if entry_px > 0 else None,
                "capital_usd": float(capital_usd),

                "stop_pct": float(stop_pct),
                "tp1_pct": float(tp1_pct),
                "trailing_pct": float(tr_pct),

                "tp1_taken": False,
                "trailing_active": False,

                "hard_stop_price": float(hard_stop),
                "tp1_price": float(tp1_price),
                "trail_price": None,
                "high_watermark": float(entry_px),

                "notes": f"AUTO OPEN from signal {row.get('estado','')}/{row.get('accion','')}",
            }

            pos_id = insert_position(payload)
            pos = {"id": pos_id, **payload}
            opened += 1
            continue

        # --- Si hay posición viva: gestionarla con OHLC del día SIGUIENTE a entry_date en adelante ---
        pos_id = int(pos["id"])
        ticker = str(pos.get("ticker", "")).upper().strip()
        entry_date = pd.to_datetime(pos.get("entry_date"), errors="coerce")

        # Sólo gestionar si ya estamos "dentro" (d >= entry_date)
        if pd.isna(entry_date) or d.date() < entry_date.date():
            continue

        ohlc_today = _get_ohlc_row(row, ticker)
        if ohlc_today["low"] is None or ohlc_today["high"] is None:
            continue  # sin high/low no hay stops intradía

        low = float(ohlc_today["low"])
        high = float(ohlc_today["high"])
        close_px = float(ohlc_today["close"]) if ohlc_today["close"] is not None else None

        # update high watermark
        hwm = pos.get("high_watermark")
        try:
            hwm = float(hwm) if hwm is not None else float(pos["entry_price"])
        except Exception:
            hwm = float(pos["entry_price"])

        if high > hwm:
            hwm = high

        hard_stop = float(pos.get("hard_stop_price") or 0.0)
        tp1_price = float(pos.get("tp1_price") or 0.0)
        trail_price = pos.get("trail_price")
        trail_price = float(trail_price) if trail_price is not None else None

        tp1_taken = bool(pos.get("tp1_taken") or False)
        trailing_active = bool(pos.get("trailing_active") or False)

        # regla 48h (para TP1/trailing). STOP siempre manda.
        held_days = _days_held(entry_date, d)

        # 1) STOP (excepción: se ejecuta siempre)
        if hard_stop > 0 and low <= hard_stop:
            exit_px = hard_stop  # asumimos fill al nivel
            entry_px = float(pos["entry_price"])
            pl_pct = (exit_px / entry_px) - 1.0
            pl_usd = float(pos.get("capital_usd") or 0.0) * pl_pct

            update_position(pos_id, {
                "status": "CLOSED",
                "exit_date": d.date(),
                "exit_price": float(exit_px),
                "pl_pct": float(pl_pct),
                "pl_usd": float(pl_usd),
                "high_watermark": float(hwm),
                "trail_price": float(trail_price) if trail_price is not None else None,
                "notes": (pos.get("notes") or "") + " | AUTO STOP",
            })
            pos = None
            closed += 1
            continue

        # 2) TP1 + activar trailing (solo si held_days >= min_hold_days)
        if (held_days >= rules.min_hold_days) and (not tp1_taken) and (tp1_price > 0) and (high >= tp1_price):
            # tomamos parcial al tp1_price
            update_position(pos_id, {
                "status": "PARTIAL",
                "tp1_taken": True,
                "trailing_active": True,
                "high_watermark": float(hwm),
                "notes": (pos.get("notes") or "") + f" | AUTO TP1 {rules.tp1_sell_fraction*100:.0f}%",
            })
            # set trail basado en HWM tras TP1
            tr_pct = float(pos.get("trailing_pct") or 0.0)
            trail_price = hwm * (1.0 - tr_pct) if tr_pct > 0 else None
            if trail_price is not None:
                update_position(pos_id, {"trail_price": float(trail_price)})

            pos["status"] = "PARTIAL"
            pos["tp1_taken"] = True
            pos["trailing_active"] = True
            pos["high_watermark"] = hwm
            pos["trail_price"] = trail_price
            partial += 1
            # seguimos al siguiente día
            continue

        # 3) Trailing update + exit (solo si held_days >= min_hold_days)
        if (held_days >= rules.min_hold_days) and trailing_active:
            tr_pct = float(pos.get("trailing_pct") or 0.0)
            if tr_pct > 0:
                new_trail = hwm * (1.0 - tr_pct)
                if trail_price is None or new_trail > trail_price:
                    trail_price = new_trail
                    update_position(pos_id, {"trail_price": float(trail_price), "high_watermark": float(hwm)})

                # salida por trailing si low toca trail
                if trail_price is not None and low <= trail_price:
                    exit_px = float(trail_price)
                    entry_px = float(pos["entry_price"])
                    pl_pct = (exit_px / entry_px) - 1.0
                    pl_usd = float(pos.get("capital_usd") or 0.0) * pl_pct

                    update_position(pos_id, {
                        "status": "CLOSED",
                        "exit_date": d.date(),
                        "exit_price": float(exit_px),
                        "pl_pct": float(pl_pct),
                        "pl_usd": float(pl_usd),
                        "high_watermark": float(hwm),
                        "trail_price": float(trail_price),
                        "notes": (pos.get("notes") or "") + " | AUTO TRAIL EXIT",
                    })
                    pos = None
                    closed += 1
                    continue

        # 4) Anti-flip: NO cerramos solo por “cambio de estado”,
        # salvo que la señal sea explícitamente el otro ticker y ya cumplimos 48h.
        if desired is not None and desired != ticker and (held_days >= rules.min_hold_days):
            # cerramos al CLOSE de hoy (o al close disponible) y ya se abrirá el otro mañana vía señal
            if close_px is not None:
                exit_px = close_px
                entry_px = float(pos["entry_price"])
                pl_pct = (exit_px / entry_px) - 1.0
                pl_usd = float(pos.get("capital_usd") or 0.0) * pl_pct

                update_position(pos_id, {
                    "status": "CLOSED",
                    "exit_date": d.date(),
                    "exit_price": float(exit_px),
                    "pl_pct": float(pl_pct),
                    "pl_usd": float(pl_usd),
                    "high_watermark": float(hwm),
                    "trail_price": float(trail_price) if trail_price is not None else None,
                    "notes": (pos.get("notes") or "") + f" | AUTO FLIP EXIT -> {desired}",
                })
                pos = None
                closed += 1
                continue

        # persist watermark (si cambió)
        update_position(pos_id, {
            "high_watermark": float(hwm),
        })
        pos["high_watermark"] = hwm

    return {"ok": True, "msg": "Execution done", "opened": opened, "closed": closed, "partial": partial}
