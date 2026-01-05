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
        # guardamos ISO; Excel no soporta tz, así que lo hacemos naive
        try:
            if x.tzinfo is not None:
                x = x.tz_convert(None)
        except Exception:
            pass
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
    # importante: lo hacemos naive y normalizado
    try:
        if getattr(t, "tzinfo", None) is not None:
            t = t.tz_convert(None)  # type: ignore
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
# UVIX: pánico real (entrada)
#   - No dependemos de uvix_score en DB (no existe).
#   - Recalculamos score con columnas existentes en vix_daily.
# ============================================================

def _uvix_score_from_row(r: pd.Series) -> int:
    """
    Score compatible con el de vix.py (aprox):
      1) vix > p65
      2) ratio > 1.30 y ratio_up True
      3) vixy_ma_3 > vixy_ma_10
      4) spy_ret < -0.008
    """
    vix = _num(r.get("vix"))
    p65 = _num(r.get("vix_p65"))
    ratio = _num(r.get("vxn_vix_ratio"))
    ratio_up = r.get("ratio_up")
    spy_ret = _num(r.get("spy_ret"))
    ma3 = _num(r.get("vixy_ma_3"))
    ma10 = _num(r.get("vixy_ma_10"))

    cond1 = (vix is not None and p65 is not None and vix > p65)
    cond2 = (ratio is not None and ratio > 1.30 and bool(ratio_up) is True)
    cond3 = (ma3 is not None and ma10 is not None and ma3 > ma10)
    cond4 = (spy_ret is not None and spy_ret < -0.008)

    return int(cond1) + int(cond2) + int(cond3) + int(cond4)


def _is_uvix_panic_real(prev_row: pd.Series) -> bool:
    """
    UVIX pánico real:
      - estado == "UVIX" (señal ya calculada en vix_daily)
      - score >= 3
      - vix > p85
      - spy_ret <= -1.2% (más estricto)
    """
    estado = str(prev_row.get("estado", "NEUTRAL") or "NEUTRAL").upper().strip()
    if estado != "UVIX":
        return False

    score = _uvix_score_from_row(prev_row)

    vix = _num(prev_row.get("vix"))
    p85 = _num(prev_row.get("vix_p85"))
    spy_ret = _num(prev_row.get("spy_ret"))

    if vix is None or p85 is None or spy_ret is None:
        return False

    return (score >= 3) and (vix > p85) and (spy_ret <= -0.012)


def _last_uvix_exit_date(pos_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if pos_df is None or pos_df.empty:
        return None
    if "ticker" not in pos_df.columns or "status" not in pos_df.columns or "exit_date" not in pos_df.columns:
        return None

    w = pos_df[(pos_df["ticker"] == "UVIX") & (pos_df["status"] == "CLOSED")].copy()
    if w.empty:
        return None

    w["exit_date"] = pd.to_datetime(w["exit_date"], errors="coerce")
    try:
        w["exit_date"] = w["exit_date"].dt.tz_localize(None)
    except Exception:
        pass

    w = w.dropna(subset=["exit_date"]).sort_values("exit_date", ascending=False)
    if w.empty:
        return None
    return _to_date(w.iloc[0]["exit_date"])


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
# Motor principal
# ============================================================

def run_vix_execution() -> None:
    """
    Motor: lee vix_daily + posiciones y aplica reglas.
    - Mantiene SVIX tal cual.
    - Mejora UVIX (pánico real + prioridad + cooldown).
    - Una posición por ticker (como antes).
    """

    daily = fetch_vix_daily()
    if daily.empty:
        return

    # Aseguramos fecha y orden
    if "fecha" not in daily.columns:
        raise RuntimeError("vix_daily no tiene columna 'fecha'")
    daily = daily.copy()
    daily["fecha"] = pd.to_datetime(daily["fecha"], errors="coerce")
    try:
        daily["fecha"] = daily["fecha"].dt.tz_localize(None)
    except Exception:
        pass
    daily["fecha"] = daily["fecha"].dt.normalize()
    daily = daily.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

    # Posiciones actuales
    pos_df = fetch_vix_positions(limit=2000)
    if pos_df.empty:
        pos_df = pd.DataFrame(columns=[
            "id", "ticker", "status", "entry_signal_date", "entry_date", "entry_price", "qty", "capital_usd",
            "stop_pct", "tp1_pct", "trailing_pct", "tp1_taken", "trailing_active",
            "hard_stop_price", "tp1_price", "trail_price", "high_watermark",
            "exit_date", "exit_price", "pl_usd", "pl_pct", "notes",
        ])

    # Buscar si hay OPEN por ticker
    open_pos: Dict[str, Dict[str, Any]] = {}
    if "status" in pos_df.columns and "ticker" in pos_df.columns:
        for tkr in ["UVIX", "SVIX"]:
            w = pos_df[(pos_df["ticker"] == tkr) & (pos_df["status"] == "OPEN")].copy()
            if not w.empty:
                w = w.sort_values("entry_date", ascending=False)
                open_pos[tkr] = w.iloc[0].to_dict()

    # UVIX cooldown base (post-trade)
    UVIX_COOLDOWN_DAYS = 5
    last_uvix_exit = _last_uvix_exit_date(pos_df)

    # Recorremos días:
    # 1) entradas al OPEN del día siguiente (basado en señal del día anterior)
    # 2) gestión de stops/tp/trailing en días posteriores
    for i in range(1, len(daily)):
        prev = daily.iloc[i - 1]
        cur = daily.iloc[i]

        cur_date = _to_date(cur.get("fecha"))
        if cur_date is None:
            continue

        # ==========================================
        # UVIX PRIORITY: si hay pánico real en prev,
        # cerramos SVIX (si existe) para poder abrir UVIX
        # ==========================================
        uvix_panic = _is_uvix_panic_real(prev)

        if uvix_panic:
            # Cooldown UVIX: solo bloquea si NO sigue pánico real
            if last_uvix_exit is not None:
                days_from_last_exit = (cur_date - last_uvix_exit).days
            else:
                days_from_last_exit = 10_000

            uvix_allowed_by_cooldown = (days_from_last_exit >= UVIX_COOLDOWN_DAYS)

            # Si hay SVIX abierta, forzamos cierre para permitir UVIX (solo si vamos a intentar UVIX)
            if ("SVIX" in open_pos) and (("UVIX" not in open_pos)) and (uvix_allowed_by_cooldown or uvix_panic):
                sv = open_pos.get("SVIX")
                if sv is not None and str(sv.get("status", "")).upper().strip() == "OPEN":
                    o_svix, _, _, _ = _get_ohlc_for_ticker(cur, "SVIX")
                    if o_svix is not None:
                        entry_price_sv = float(sv.get("entry_price") or 0.0)
                        qty_sv = float(sv.get("qty") or 0.0)
                        pl_usd_sv = (float(o_svix) - entry_price_sv) * qty_sv
                        pl_pct_sv = (float(o_svix) / entry_price_sv - 1.0) if entry_price_sv > 0 else None

                        _update_position(int(sv["id"]), {
                            "status": "CLOSED",
                            "exit_date": cur_date.date().isoformat(),
                            "exit_price": float(o_svix),
                            "pl_usd": float(pl_usd_sv),
                            "pl_pct": float(pl_pct_sv) if pl_pct_sv is not None else None,
                            "notes": (str(sv.get("notes") or "") + " | EXIT FLIP_TO_UVIX @ open").strip(),
                        })
                        open_pos.pop("SVIX", None)

        # =========================
        # ENTRADAS (si no hay OPEN)
        # =========================

        # --- UVIX: entrada SOLO pánico real ---
        if "UVIX" not in open_pos:
            if uvix_panic:
                # cooldown check (solo bloquea si no hay pánico real; aquí ya hay pánico real)
                if last_uvix_exit is not None:
                    days_from_last_exit = (cur_date - last_uvix_exit).days
                else:
                    days_from_last_exit = 10_000

                if days_from_last_exit >= UVIX_COOLDOWN_DAYS or uvix_panic:
                    o, _, _, _ = _get_ohlc_for_ticker(cur, "UVIX")
                    if o is not None:
                        rules = RULES_BY_TICKER["UVIX"]
                        entry_price = float(o)
                        capital_usd = 10000.0
                        qty = capital_usd / entry_price if entry_price > 0 else 0.0

                        hard_stop_price = entry_price * (1.0 - rules.stop_pct)
                        tp1_price = entry_price * (1.0 + rules.tp1_pct)

                        payload = {
                            "ticker": "UVIX",
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
                            "notes": "Auto-entry UVIX (panic real).",
                        }

                        new_id = _insert_position(payload)
                        payload["id"] = new_id
                        open_pos["UVIX"] = payload

        # --- SVIX: NO tocamos la lógica de entrada existente ---
        prev_state = str(prev.get("estado", "NEUTRAL") or "NEUTRAL").upper().strip()
        if "SVIX" not in open_pos:
            if prev_state == "SVIX":
                o, _, _, _ = _get_ohlc_for_ticker(cur, "SVIX")
                if o is not None:
                    rules = RULES_BY_TICKER.get("SVIX")
                    if rules is not None:
                        entry_price = float(o)
                        capital_usd = 10000.0
                        qty = capital_usd / entry_price if entry_price > 0 else 0.0

                        hard_stop_price = entry_price * (1.0 - rules.stop_pct)
                        tp1_price = entry_price * (1.0 + rules.tp1_pct)

                        payload = {
                            "ticker": "SVIX",
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
                        open_pos["SVIX"] = payload

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

            exit_reason = None
            exit_price = None

            # ------------------------------------------------------------
            # 1) HARD STOP (se permite aunque no haya 48h)
            # ------------------------------------------------------------
            if hard_stop_price is not None:
                if o is not None and float(o) <= float(hard_stop_price):
                    exit_reason = "HARD_STOP_GAP"
                    exit_price = float(o)
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
            # 5) Regime exit (NO CAMBIO SVIX). UVIX: se mantiene, pero no lo usa para entrada.
            # ------------------------------------------------------------
            if exit_reason is None and allow_exits:
                cur_state = str(cur.get("estado", "NEUTRAL") or "NEUTRAL").upper().strip()
                prev2_state = str(prev.get("estado", "NEUTRAL") or "NEUTRAL").upper().strip()
                if tkr == "UVIX":
                    if (prev2_state != "UVIX") and (cur_state != "UVIX"):
                        if o is not None:
                            exit_reason = "REGIME_OFF"
                            exit_price = float(o)

            # ------------------------------------------------------------
            # Aplicar TP1 (si toca)
            # ------------------------------------------------------------
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

            # ------------------------------------------------------------
            # Cerrar posición (si hay exit_reason)
            # ------------------------------------------------------------
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

                # si cerramos UVIX, guardamos exit para cooldown
                if tkr == "UVIX":
                    last_uvix_exit = cur_date

                open_pos.pop(tkr, None)
                continue

            # ------------------------------------------------------------
            # Si sigue OPEN: actualizar watermark/trail si aplica
            # ------------------------------------------------------------
            patch = {"high_watermark": high_watermark}
            if trailing_active:
                patch["trail_price"] = high_watermark * (1.0 - trailing_pct)
            _update_position(int(pos["id"]), patch)

            pos["high_watermark"] = high_watermark
            if trailing_active:
                pos["trail_price"] = patch["trail_price"]


# alias (por si tu main importaba otro nombre en alguna iteración)
run_vix_execution = run_vix_execution
