# backend/vix.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import math
import pandas as pd

from .supabase_client import supabase


# ============================================================
# CONFIG
# ============================================================

@dataclass
class VixConfig:
    lookback_pct: int = 252
    ratio_alert: float = 1.30
    ratio_ok: float = 1.25

    # percentiles para estados
    p25: float = 0.25
    p50: float = 0.50
    p65: float = 0.65
    p85: float = 0.85

    # guardarraíl anti “VIX demasiado bajo”
    use_guardrail: bool = True
    guardrail_vix_floor: float = 12.5   # si VIX < 12.5 => no abrir SVIX


DEFAULT_CFG = VixConfig()


# ============================================================
# JSON SAFE (SOLUCIÓN DEFINITIVA NAType / date / numpy)
# ============================================================

def _is_nan_like(x: Any) -> bool:
    try:
        if x is None:
            return True
        if x is pd.NA:
            return True
        if isinstance(x, float) and math.isnan(x):
            return True
        # pandas NaT
        if isinstance(x, (pd.Timestamp,)) and pd.isna(x):
            return True
        return pd.isna(x)  # cubre NaN/NaT/NA
    except Exception:
        return False


def _json_safe_value(x: Any) -> Any:
    """Convierte tipos pandas/numpy/date a valores serializables por JSON."""
    if _is_nan_like(x):
        return None

    # fechas -> ISO string
    if isinstance(x, pd.Timestamp):
        # normalizamos a fecha si parece date
        return x.to_pydatetime().date().isoformat()
    if hasattr(x, "isoformat") and "date" in str(type(x)).lower():
        try:
            return x.isoformat()
        except Exception:
            pass

    # numpy scalars -> python
    try:
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            v = float(x)
            return None if math.isnan(v) else v
        if isinstance(x, (np.bool_,)):
            return bool(x)
    except Exception:
        pass

    # bool/int/float/str ok
    return x


def _records_json_safe(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        rr: Dict[str, Any] = {}
        for k, v in r.items():
            rr[k] = _json_safe_value(v)
        out.append(rr)
    return out


# ============================================================
# YAHOO DOWNLOAD
# ============================================================

def download_yahoo_daily(start: str, end: str) -> pd.DataFrame:
    """
    Descarga diaria de:
      ^VIX, ^VXN, VIXY, SPY

    Devuelve df con columnas:
      date, vix, vxn, vixy, spy
    """
    import yfinance as yf

    tickers = {
        "^VIX": "vix",
        "^VXN": "vxn",
        "VIXY": "vixy",
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
        )

        if data is None or data.empty:
            raise RuntimeError(f"No hay datos Yahoo para {tkr} en rango {start}..{end}")

        # yfinance a veces devuelve columnas MultiIndex. Nos quedamos con Close.
        close = None
        if "Close" in data.columns:
            close = data["Close"]
        else:
            # fallback: intenta primer nivel
            close = data.iloc[:, 0]

        s = close.copy()
        s.name = col

        df = s.reset_index()
        # yfinance suele usar Date, a veces Datetime
        if "Date" in df.columns:
            df.rename(columns={"Date": "date"}, inplace=True)
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "date"}, inplace=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        if out is None:
            out = df
        else:
            out = out.merge(df, on="date", how="outer")

    if out is None:
        raise RuntimeError("No se pudo construir el dataframe de Yahoo (out=None).")

    out = out.sort_values("date").reset_index(drop=True)
    return out


# ============================================================
# FEATURES + ESTADOS (ALINEADO CON TUS TABLAS)
# ============================================================

def compute_features(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df.copy()

    for c in ["vix", "vxn", "vixy", "spy"]:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")

    # retornos SPY
    w["spy_return"] = w["spy"].pct_change()

    # ratio VXN/VIX y si sube
    w["vxn_vix_ratio"] = w["vxn"] / w["vix"]
    w["ratio_up"] = w["vxn_vix_ratio"].diff() > 0

    # percentiles rolling VIX (252)
    lb = int(cfg.lookback_pct)
    w["vix_p25"] = w["vix"].rolling(lb).quantile(cfg.p25)
    w["vix_p50"] = w["vix"].rolling(lb).quantile(cfg.p50)
    w["vix_p65"] = w["vix"].rolling(lb).quantile(cfg.p65)
    w["vix_p85"] = w["vix"].rolling(lb).quantile(cfg.p85)

    # VIXY MA3 / MA10 (tus nombres: vixy_ma_3, vixy_ma_10)
    w["vixy_ma_3"] = w["vixy"].rolling(3).mean()
    w["vixy_ma_10"] = w["vixy"].rolling(10).mean()

    return w


def _vix_regime(row: pd.Series) -> Optional[str]:
    vix = row.get("vix")
    p25 = row.get("vix_p25")
    p50 = row.get("vix_p50")
    p65 = row.get("vix_p65")
    p85 = row.get("vix_p85")

    if pd.isna(vix) or pd.isna(p25) or pd.isna(p50) or pd.isna(p65) or pd.isna(p85):
        return None

    if vix < p25:
        return "CALMA"
    if vix < p50:
        return "ALERTA"
    if vix < p65:
        return "ALERTA"
    if vix < p85:
        return "TENSION"
    return "PANICO"


def _contango_estado(row: pd.Series) -> Optional[str]:
    ma3 = row.get("vixy_ma_3")
    ma10 = row.get("vixy_ma_10")
    if pd.isna(ma3) or pd.isna(ma10):
        return None

    if ma3 < ma10:
        return "CONTANGO"
    if ma3 > ma10:
        return "BACKWARDATION"
    return "TRANSICION"


def decide_estado(row: pd.Series, cfg: VixConfig = DEFAULT_CFG) -> Tuple[str, str]:
    """
    Devuelve (estado, motivo)
    estado en tu tabla vix_signal:
      'SVIX' | 'NEUTRAL' | 'UVIX' | 'CERRAR_UVIX'
    """
    vix = row.get("vix")
    ratio = row.get("vxn_vix_ratio")
    ratio_up = bool(row.get("ratio_up")) if pd.notna(row.get("ratio_up")) else False

    p25 = row.get("vix_p25")
    p65 = row.get("vix_p65")
    p85 = row.get("vix_p85")

    contango = _contango_estado(row)
    spy_ret = row.get("spy_return")

    # sin rolling: neutral
    if pd.isna(vix) or pd.isna(p25) or pd.isna(p65) or pd.isna(p85):
        return ("NEUTRAL", "NO DATA: faltan 252 sesiones para percentiles.")

    # guardarraíl: VIX extremadamente bajo -> no abrir SVIX
    if cfg.use_guardrail and pd.notna(vix) and vix < cfg.guardrail_vix_floor:
        return ("NEUTRAL", "Guardarraíl: VIX demasiado bajo (snapback risk).")

    # SVIX: calma + ratio ok + contango
    if (vix < p25) and (pd.notna(ratio) and ratio < cfg.ratio_ok) and (contango == "CONTANGO"):
        return ("SVIX", "VIX < P25 + ratio VXN/VIX ok + contango.")

    # UVIX: score >=2
    uv1 = (vix > p65)
    uv2 = (pd.notna(ratio) and ratio > cfg.ratio_alert and ratio_up)
    uv3 = (contango == "BACKWARDATION")
    uv4 = (pd.notna(spy_ret) and spy_ret < -0.008)
    score = sum([bool(uv1), bool(uv2), bool(uv3), bool(uv4)])
    if score >= 2:
        return ("UVIX", f"Stress score={score} (VIX/radio/contango/SPY).")

    # cierre UVIX / preparar SVIX: pánico (VIX > P85) y deja de subir ratio y vuelve contango
    if (vix > p85) and (ratio_up is False) and (contango == "CONTANGO"):
        return ("CERRAR_UVIX", "Pánico se agota: ratio deja de subir + contango vuelve.")

    return ("NEUTRAL", "Régimen mixto / transición.")


def compute_all(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = compute_features(df, cfg=cfg)
    w["vix_regime"] = w.apply(_vix_regime, axis=1)
    w["contango_estado"] = w.apply(_contango_estado, axis=1)

    estados: List[str] = []
    motivos: List[str] = []
    for _, r in w.iterrows():
        e, m = decide_estado(r, cfg=cfg)
        estados.append(e)
        motivos.append(m)

    w["estado"] = estados
    w["motivo"] = motivos
    return w


# ============================================================
# SUPABASE (ALINEADO CON TUS SQL)
# ============================================================

def upsert_vix_daily(df: pd.DataFrame) -> int:
    """
    Tabla: vix_daily
    Columnas (según tu SQL):
      fecha (PK)
      vix,vxn,vixy,spy
      vxn_vix_ratio
      vix_p25,vix_p50,vix_p65,vix_p85
      vix_regime
      vixy_ma_3,vixy_ma_10
      contango_estado
      created_at (default)
    """
    if df.empty:
        return 0

    w = df.copy()
    w["fecha"] = pd.to_datetime(w["date"], errors="coerce").dt.date

    keep = [
        "fecha",
        "vix", "vxn", "vixy", "spy",
        "vxn_vix_ratio",
        "vix_p25", "vix_p50", "vix_p65", "vix_p85",
        "vix_regime",
        "vixy_ma_3", "vixy_ma_10",
        "contango_estado",
    ]
    w = w[[c for c in keep if c in w.columns]].copy()

    records = w.to_dict(orient="records")
    records = _records_json_safe(records)

    resp = supabase.table("vix_daily").upsert(records, on_conflict="fecha").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error upsert vix_daily: {resp.error}")

    return len(records)


def upsert_vix_signal(df: pd.DataFrame) -> int:
    """
    Tabla: vix_signal
    Columnas (según tu SQL):
      fecha (PK)
      estado (NOT NULL)
      motivo
      vix
      vxn_vix_ratio
      contango_estado
      spy_return
      macro_evento (lo ponemos False)
      created_at (default)
    """
    if df.empty:
        return 0

    w = df.copy()
    w["fecha"] = pd.to_datetime(w["date"], errors="coerce").dt.date
    w["macro_evento"] = False  # no estás usando macro_events ahora

    keep = [
        "fecha",
        "estado",
        "motivo",
        "vix",
        "vxn_vix_ratio",
        "contango_estado",
        "spy_return",
        "macro_evento",
    ]
    w = w[[c for c in keep if c in w.columns]].copy()

    records = w.to_dict(orient="records")
    records = _records_json_safe(records)

    resp = supabase.table("vix_signal").upsert(records, on_conflict="fecha").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error upsert vix_signal: {resp.error}")

    return len(records)


def fetch_vix_daily() -> pd.DataFrame:
    resp = supabase.table("vix_daily").select("*").order("fecha", desc=False).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error leyendo vix_daily: {resp.error}")
    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    if not df.empty and "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


def fetch_vix_signal() -> pd.DataFrame:
    resp = supabase.table("vix_signal").select("*").order("fecha", desc=False).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error leyendo vix_signal: {resp.error}")
    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    if not df.empty and "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


# ============================================================
# PIPELINE ÚNICO
# ============================================================

def run_vix_pipeline(start: str, end: str, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    raw = download_yahoo_daily(start=start, end=end)
    out = compute_all(raw, cfg=cfg)

    # upserts alineados a tu esquema
    upsert_vix_daily(out)
    upsert_vix_signal(out)

    return out
