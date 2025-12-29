# backend/vix.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Iterable

import numpy as np
import pandas as pd

from .supabase_client import supabase


# -----------------------------
# Config
# -----------------------------

@dataclass
class VixConfig:
    lookback_pct: int = 252
    ratio_alert: float = 1.30
    ratio_ok: float = 1.25

    # Guardarraíl “VIX demasiado bajo”
    use_guardrail: bool = True
    guardrail_vix_floor: float = 12.5  # si VIX < 12.5 => no abrir SVIX

    # A) HOLD mínimo (días) para posiciones abiertas
    min_hold_days_svix: int = 2
    min_hold_days_uvix: int = 2

    # B) Cooldown tras cerrar posición (días)
    cooldown_days: int = 3

    # Ticker proxy contango (guardamos en columna "vixy" para no tocar DB)
    term_etp_ticker: str = "VXX"   # antes era VIXY, ahora VXX


DEFAULT_CFG = VixConfig()


# -----------------------------
# Utilidades robustas
# -----------------------------

def _safe_num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        return out

    if isinstance(out.index, (pd.DatetimeIndex,)):
        out = out.reset_index()
        if "Date" in out.columns:
            out.rename(columns={"Date": "date"}, inplace=True)
        elif "Datetime" in out.columns:
            out.rename(columns={"Datetime": "date"}, inplace=True)
        else:
            out.rename(columns={out.columns[0]: "date"}, inplace=True)

        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        return out

    raise RuntimeError("No se pudo detectar columna/índice de fecha en la descarga de Yahoo.")


def _pick_close_column(df: pd.DataFrame) -> pd.Series:
    cols = list(df.columns)

    if isinstance(df.columns, pd.MultiIndex):
        try:
            s = df["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return s
        except Exception:
            pass

    for c in ["Close", "close", "Adj Close", "adjclose", "AdjClose"]:
        if c in cols:
            return df[c]

    raise RuntimeError(f"No se encontró columna de cierre en Yahoo. Columnas: {cols}")


def _ensure_expected_columns(out: pd.DataFrame, expected: Iterable[str]) -> None:
    missing = [c for c in expected if c not in out.columns]
    if missing:
        raise RuntimeError(
            "Descarga/merge incompleto. Faltan columnas: "
            f"{missing}. Columnas presentes: {list(out.columns)}"
        )


def _json_sanitize_value(x: Any) -> Any:
    if x is None:
        return None

    # pandas NA/NaN
    if pd.isna(x):
        return None

    # pandas Timestamp
    if isinstance(x, pd.Timestamp):
        return x.isoformat()

    # python date/datetime
    if hasattr(x, "isoformat"):
        try:
            return x.isoformat()
        except Exception:
            return str(x)

    # numpy scalars
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)

    return x


def _json_sanitize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean: List[Dict[str, Any]] = []
    for r in records:
        clean.append({k: _json_sanitize_value(v) for k, v in r.items()})
    return clean


# -----------------------------
# Supabase: macro events
# -----------------------------

def fetch_macro_events() -> pd.DataFrame:
    """
    Tabla esperada: macro_events(fecha, label, impacto, activo)
    """
    resp = supabase.table("macro_events").select("*").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error leyendo macro_events: {resp.error}")

    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    if "activo" in df.columns:
        df["activo"] = df["activo"].fillna(True)
    return df


def macro_tomorrow_flag(fecha: pd.Timestamp, macro_df: pd.DataFrame) -> bool:
    if macro_df is None or macro_df.empty:
        return False
    tomorrow = (fecha + pd.Timedelta(days=1)).date()
    w = macro_df.copy()
    w = w[(w.get("activo", True) == True) & (w["fecha"] == tomorrow)]
    return len(w) > 0


# -----------------------------
# Yahoo download (robusto)
# -----------------------------

def download_yahoo_daily(start: str, end: str, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    """
    Descarga diaria de: ^VIX, ^VXN, (term_etp_ticker), SPY
    Devuelve columnas: date, vix, vxn, vixy, spy
    Nota: guardamos el term ETP en columna 'vixy' para no tocar el esquema de Supabase.
    """
    import yfinance as yf

    tickers = {
        "^VIX": "vix",
        "^VXN": "vxn",
        cfg.term_etp_ticker: "vixy",  # <- VXX (o el que pongas), guardado como 'vixy'
        "SPY": "spy",
    }

    out: Optional[pd.DataFrame] = None

    for tkr, col in tickers.items():
        data = yf.download(
            tkr,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            actions=False,
        )

        if data is None or data.empty:
            raise RuntimeError(f"No hay datos para {tkr} en Yahoo en el rango {start}..{end}")

        data = _normalize_date_index(data)
        close = _pick_close_column(data)

        df_one = pd.DataFrame({"date": data["date"], col: close.values})
        df_one["date"] = pd.to_datetime(df_one["date"], errors="coerce").dt.normalize()

        if out is None:
            out = df_one
        else:
            out = out.merge(df_one, on="date", how="outer")

    assert out is not None
    out = out.sort_values("date").reset_index(drop=True)

    _ensure_expected_columns(out, ["date", "vix", "vxn", "vixy", "spy"])
    return out


# -----------------------------
# Features
# -----------------------------

def compute_features(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df.copy()
    _ensure_expected_columns(w, ["date", "vix", "vxn", "vixy", "spy"])

    w["vix"] = _safe_num_series(w["vix"])
    w["vxn"] = _safe_num_series(w["vxn"])
    w["vixy"] = _safe_num_series(w["vixy"])  # en realidad puede ser VXX
    w["spy"] = _safe_num_series(w["spy"])

    # retorno SPY
    w["spy_ret"] = w["spy"].pct_change()

    # ratio VXN/VIX + dirección
    w["vxn_vix_ratio"] = w["vxn"] / w["vix"]
    w["ratio_up"] = w["vxn_vix_ratio"].diff() > 0

    lb = int(cfg.lookback_pct)
    w["vix_p10"] = w["vix"].rolling(lb).quantile(0.10)
    w["vix_p25"] = w["vix"].rolling(lb).quantile(0.25)
    w["vix_p50"] = w["vix"].rolling(lb).quantile(0.50)
    w["vix_p65"] = w["vix"].rolling(lb).quantile(0.65)
    w["vix_p85"] = w["vix"].rolling(lb).quantile(0.85)

    # MA3 vs MA10 del term ETP (VXX)
    w["vixy_ma_3"] = w["vixy"].rolling(3).mean()
    w["vixy_ma_10"] = w["vixy"].rolling(10).mean()

    # proxy contango estable: MA3 < MA10
    w["contango_ok"] = w["vixy_ma_3"] < w["vixy_ma_10"]

    return w


# -----------------------------
# Señal "bruta" (sin A/B)
# -----------------------------

def decide_state_row(row: pd.Series, cfg: VixConfig = DEFAULT_CFG) -> Dict[str, Any]:
    vix = row.get("vix")
    p25 = row.get("vix_p25")
    p65 = row.get("vix_p65")
    p85 = row.get("vix_p85")

    ratio = row.get("vxn_vix_ratio")
    ratio_up = bool(row.get("ratio_up")) if pd.notna(row.get("ratio_up")) else False
    contango_ok = bool(row.get("contango_ok")) if pd.notna(row.get("contango_ok")) else False
    spy_ret = row.get("spy_ret")
    macro_tomorrow = bool(row.get("macro_tomorrow")) if pd.notna(row.get("macro_tomorrow")) else False

    # sin percentiles aún => no operar
    if pd.isna(p25) or pd.isna(p65) or pd.isna(p85) or pd.isna(vix):
        return {"raw_estado": "NEUTRAL", "raw_accion": "NO DATA", "raw_comentario": "Insuficiente histórico para rolling 252."}

    # guardarraíl VIX demasiado bajo
    if cfg.use_guardrail and pd.notna(vix) and float(vix) < float(cfg.guardrail_vix_floor):
        return {
            "raw_estado": "NEUTRAL",
            "raw_accion": "NO OPEN SVIX",
            "raw_comentario": "Guardarraíl: VIX extremadamente bajo (riesgo snapback).",
        }

    # SVIX candidate
    cond_svix = (
        (vix < p25)
        and (pd.notna(ratio) and ratio < cfg.ratio_ok)
        and contango_ok
        and (macro_tomorrow is False)
    )
    if cond_svix:
        return {"raw_estado": "SVIX", "raw_accion": "OPEN/HOLD SVIX", "raw_comentario": "Calma + contango + sin macro mañana."}

    # UVIX candidate (LARGO)
    uvix_cond1 = vix > p65
    uvix_cond2 = (pd.notna(ratio) and ratio > cfg.ratio_alert and ratio_up)
    uvix_cond3 = (pd.notna(row.get("vixy_ma_3")) and pd.notna(row.get("vixy_ma_10")) and (row.get("vixy_ma_3") > row.get("vixy_ma_10")))
    uvix_cond4 = (pd.notna(spy_ret) and spy_ret < -0.008)

    uvix_score = sum([bool(uvix_cond1), bool(uvix_cond2), bool(uvix_cond3), bool(uvix_cond4)])
    if uvix_score >= 2:
        return {"raw_estado": "UVIX", "raw_accion": "BUY/HOLD UVIX", "raw_comentario": f"Stress score={uvix_score} (UVIX LONG)."}

    # Purple: pánico agotándose
    cond_purple = (vix > p85) and (ratio_up is False) and contango_ok
    if cond_purple:
        return {"raw_estado": "PREP_SVIX", "raw_accion": "CLOSE UVIX / PREPARE SVIX", "raw_comentario": "Pánico se agota + contango vuelve."}

    return {"raw_estado": "NEUTRAL", "raw_accion": "NO NEW POSITION", "raw_comentario": "Régimen mixto / transición."}


# -----------------------------
# Aplicación de reglas A/B (hold + cooldown)
# -----------------------------

def _days_between(d1: pd.Timestamp, d2: pd.Timestamp) -> int:
    # d2 - d1 en días enteros (suponiendo normalizados)
    return int((d2.normalize() - d1.normalize()).days)


def apply_position_rules(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    """
    Convierte raw_estado/raw_accion/raw_comentario en estado/accion/comentario finales
    aplicando:
      A) hold mínimo
      B) cooldown tras cierre
    """
    w = df.copy().sort_values("date").reset_index(drop=True)

    position: str = "NONE"  # NONE | SVIX | UVIX
    position_open_date: Optional[pd.Timestamp] = None
    last_close_date: Optional[pd.Timestamp] = None

    estados: List[str] = []
    acciones: List[str] = []
    comentarios: List[str] = []

    for _, r in w.iterrows():
        date = pd.to_datetime(r["date"], errors="coerce").normalize()
        raw_estado = str(r.get("raw_estado", "NEUTRAL"))
        raw_accion = str(r.get("raw_accion", ""))
        raw_com = str(r.get("raw_comentario", ""))

        # cooldown activo?
        cooldown_active = False
        if last_close_date is not None and pd.notna(date):
            cooldown_active = _days_between(last_close_date, date) < int(cfg.cooldown_days)

        # helper: hold cumplido?
        def hold_ok() -> bool:
            if position == "SVIX":
                if position_open_date is None:
                    return True
                return _days_between(position_open_date, date) >= int(cfg.min_hold_days_svix)
            if position == "UVIX":
                if position_open_date is None:
                    return True
                return _days_between(position_open_date, date) >= int(cfg.min_hold_days_uvix)
            return True

        # --- si NO hay posición ---
        if position == "NONE":
            if cooldown_active:
                estados.append("NEUTRAL")
                acciones.append("COOLDOWN (NO OPEN)")
                comentarios.append(f"Cooldown activo ({cfg.cooldown_days}d) tras cierre. Raw={raw_estado}. {raw_com}")
                continue

            if raw_estado == "SVIX":
                position = "SVIX"
                position_open_date = date
                estados.append("SVIX")
                acciones.append("OPEN SVIX")
                comentarios.append(raw_com)
                continue

            if raw_estado == "UVIX":
                position = "UVIX"
                position_open_date = date
                estados.append("UVIX")
                acciones.append("BUY UVIX")
                comentarios.append(raw_com)
                continue

            estados.append("NEUTRAL")
            acciones.append("NO POSITION")
            comentarios.append(raw_com)
            continue

        # --- si HAY posición SVIX ---
        if position == "SVIX":
            if not hold_ok():
                estados.append("SVIX")
                acciones.append("HOLD SVIX (MIN HOLD)")
                comentarios.append(f"Hold mínimo SVIX {cfg.min_hold_days_svix}d. Raw={raw_estado}. {raw_com}")
                continue

            # tras hold mínimo: permitir cerrar / rotar
            if raw_estado == "UVIX":
                # rotación: cerrar SVIX y comprar UVIX (pero respeta cooldown? aquí rotamos directo)
                position = "UVIX"
                position_open_date = date
                estados.append("UVIX")
                acciones.append("ROTATE: CLOSE SVIX -> BUY UVIX")
                comentarios.append(f"Rotación por señal UVIX. {raw_com}")
                continue

            if raw_estado in ("NEUTRAL", "PREP_SVIX"):
                # cerrar SVIX
                position = "NONE"
                position_open_date = None
                last_close_date = date
                estados.append("NEUTRAL")
                acciones.append("CLOSE SVIX")
                comentarios.append(f"Cierre SVIX por régimen {raw_estado}. {raw_com}")
                continue

            # raw sigue SVIX
            estados.append("SVIX")
            acciones.append("HOLD SVIX")
            comentarios.append(raw_com)
            continue

        # --- si HAY posición UVIX ---
        if position == "UVIX":
            if not hold_ok():
                estados.append("UVIX")
                acciones.append("HOLD UVIX (MIN HOLD)")
                comentarios.append(f"Hold mínimo UVIX {cfg.min_hold_days_uvix}d. Raw={raw_estado}. {raw_com}")
                continue

            # tras hold mínimo: permitir cerrar / rotar
            if raw_estado == "SVIX":
                position = "SVIX"
                position_open_date = date
                estados.append("SVIX")
                acciones.append("ROTATE: CLOSE UVIX -> OPEN SVIX")
                comentarios.append(f"Rotación por señal SVIX. {raw_com}")
                continue

            if raw_estado in ("NEUTRAL", "PREP_SVIX"):
                position = "NONE"
                position_open_date = None
                last_close_date = date
                estados.append("NEUTRAL")
                acciones.append("CLOSE UVIX")
                comentarios.append(f"Cierre UVIX por régimen {raw_estado}. {raw_com}")
                continue

            estados.append("UVIX")
            acciones.append("HOLD UVIX")
            comentarios.append(raw_com)
            continue

        # fallback
        estados.append("NEUTRAL")
        acciones.append("NO POSITION")
        comentarios.append(raw_com)

    w["estado"] = estados
    w["accion"] = acciones
    w["comentario"] = comentarios
    return w


# -----------------------------
# Compute states (macro + raw + A/B)
# -----------------------------

def compute_states(df_feat: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df_feat.copy()

    macro = fetch_macro_events()
    w["macro_tomorrow"] = w["date"].apply(
        lambda d: macro_tomorrow_flag(pd.to_datetime(d), macro) if pd.notna(d) else False
    )

    raws = w.apply(lambda row: decide_state_row(row, cfg=cfg), axis=1, result_type="expand")
    w = pd.concat([w, raws], axis=1)

    w = apply_position_rules(w, cfg=cfg)
    return w


# -----------------------------
# Supabase: vix_daily (única tabla)
# -----------------------------

def upsert_vix_daily(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    w = df.copy()
    w["fecha"] = pd.to_datetime(w["date"], errors="coerce").dt.date

    keep_cols = [
        "fecha",
        "vix", "vxn", "vixy", "spy",
        "spy_ret",
        "vxn_vix_ratio",
        "vix_p10", "vix_p25", "vix_p50", "vix_p65", "vix_p85",
        "vixy_ma_3", "vixy_ma_10",
        "contango_ok",
        "macro_tomorrow",
        "estado", "accion", "comentario",
        # opcional: guardar también el raw para auditar
        "raw_estado", "raw_accion", "raw_comentario",
    ]
    w = w[[c for c in keep_cols if c in w.columns]].copy()

    records: List[Dict[str, Any]] = w.to_dict(orient="records")
    records = _json_sanitize_records(records)

    resp = supabase.table("vix_daily").upsert(records, on_conflict="fecha").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)

    return len(records)


def fetch_vix_daily() -> pd.DataFrame:
    resp = supabase.table("vix_daily").select("*").order("fecha", desc=False).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)
    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    if not df.empty and "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


# -----------------------------
# Pipeline 1-click
# -----------------------------

def run_vix_pipeline(start: str, end: str, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    raw = download_yahoo_daily(start=start, end=end, cfg=cfg)
    feat = compute_features(raw, cfg=cfg)
    out = compute_states(feat, cfg=cfg)
    upsert_vix_daily(out)
    return out
