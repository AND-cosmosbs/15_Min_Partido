# backend/vix.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

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

    # percentiles usados como umbrales
    vix_panic: float = 0.85   # P85
    vix_tension: float = 0.65 # P65
    vix_calm: float = 0.25    # P25

    # Guardarraíl “VIX demasiado bajo”
    use_guardrail: bool = True
    guardrail_p10: float = 0.10
    guardrail_vix_floor: float = 12.5  # si VIX < 12.5 => no abrir SVIX


DEFAULT_CFG = VixConfig()


# -----------------------------
# Helpers
# -----------------------------

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _iso_date(x) -> Optional[str]:
    """Devuelve 'YYYY-MM-DD' o None."""
    try:
        d = pd.to_datetime(x, errors="coerce")
        if pd.isna(d):
            return None
        return d.date().isoformat()
    except Exception:
        return None


def _try_fetch_macro_events() -> pd.DataFrame:
    """
    Si existe tabla macro_events(fecha, label, impacto, activo) la usa.
    Si NO existe, devuelve DF vacío sin romper.
    """
    try:
        resp = supabase.table("macro_events").select("*").execute()
        if getattr(resp, "error", None):
            return pd.DataFrame()
        data = getattr(resp, "data", None) or []
        df = pd.DataFrame(data)
        if df.empty:
            return df
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
        if "activo" in df.columns:
            df["activo"] = df["activo"].fillna(True)
        else:
            df["activo"] = True
        return df
    except Exception:
        return pd.DataFrame()


def _macro_tomorrow_flag(fecha: pd.Timestamp, macro_df: pd.DataFrame) -> bool:
    if macro_df is None or macro_df.empty:
        return False
    tomorrow = (fecha + pd.Timedelta(days=1)).date()
    w = macro_df.copy()
    if "activo" not in w.columns:
        w["activo"] = True
    w = w[(w["activo"] == True) & (w["fecha"] == tomorrow)]
    return len(w) > 0


# -----------------------------
# Yahoo download
# -----------------------------

def download_yahoo_daily(start: str, end: str) -> pd.DataFrame:
    """
    Descarga diaria de:
      ^VIX, ^VXN, VIXY, SPY
    Devuelve df con columnas: date, vix, vxn, vixy, spy
    """
    import yfinance as yf

    tickers = {
        "^VIX": "vix",
        "^VXN": "vxn",
        "VIXY": "vixy",
        "SPY": "spy",
    }

    out = None

    for tkr, col in tickers.items():
        data = yf.download(
            tkr, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False
        )
        if data is None or data.empty:
            raise RuntimeError(f"No hay datos para {tkr} en Yahoo Finance para {start}..{end}")

        s = data["Close"].copy()
        s.name = col
        df = s.reset_index()
        df.rename(columns={"Date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        out = df if out is None else out.merge(df, on="date", how="outer")

    out = out.sort_values("date").reset_index(drop=True)
    return out


# -----------------------------
# Features + estado (alineado con tus tablas)
# -----------------------------

def compute_features(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    """
    df: columnas date, vix, vxn, vixy, spy
    """
    w = df.copy()
    w["vix"] = _safe_num(w["vix"])
    w["vxn"] = _safe_num(w["vxn"])
    w["vixy"] = _safe_num(w["vixy"])
    w["spy"] = _safe_num(w["spy"])

    # SPY retorno diario
    w["spy_ret"] = w["spy"].pct_change()

    # Ratio VXN/VIX + dirección
    w["vxn_vix_ratio"] = w["vxn"] / w["vix"]
    w["ratio_up"] = w["vxn_vix_ratio"].diff() > 0

    # VIX percentiles rolling (252)
    lb = int(cfg.lookback_pct)
    w["vix_p10"] = w["vix"].rolling(lb).quantile(0.10)
    w["vix_p25"] = w["vix"].rolling(lb).quantile(0.25)
    w["vix_p50"] = w["vix"].rolling(lb).quantile(0.50)
    w["vix_p65"] = w["vix"].rolling(lb).quantile(0.65)
    w["vix_p85"] = w["vix"].rolling(lb).quantile(0.85)

    # VIXY MA3 vs MA10 (ojo: en tu SQL son vixy_ma_3 y vixy_ma_10)
    w["vixy_ma_3"] = w["vixy"].rolling(3).mean()
    w["vixy_ma_10"] = w["vixy"].rolling(10).mean()

    return w


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


def _vix_regime(row: pd.Series) -> Optional[str]:
    vix = row.get("vix")
    p25 = row.get("vix_p25")
    p65 = row.get("vix_p65")
    p85 = row.get("vix_p85")
    if pd.isna(vix) or pd.isna(p25) or pd.isna(p65) or pd.isna(p85):
        return None
    if vix < p25:
        return "CALMA"
    if vix < p65:
        return "ALERTA"
    if vix < p85:
        return "TENSION"
    return "PANICO"


def decide_estado(row: pd.Series, cfg: VixConfig = DEFAULT_CFG) -> Tuple[str, str, bool]:
    """
    Devuelve (estado_signal, motivo, macro_evento)
    estado_signal: 'SVIX' | 'NEUTRAL' | 'UVIX' | 'CERRAR_UVIX'
    """
    vix = row.get("vix")
    p10 = row.get("vix_p10")
    p25 = row.get("vix_p25")
    p65 = row.get("vix_p65")
    p85 = row.get("vix_p85")

    ratio = row.get("vxn_vix_ratio")
    ratio_up = bool(row.get("ratio_up")) if pd.notna(row.get("ratio_up")) else False
    spy_ret = row.get("spy_ret")
    contango = _contango_estado(row)
    macro_tomorrow = bool(row.get("macro_tomorrow")) if pd.notna(row.get("macro_tomorrow")) else False

    # Si aún no hay rolling
    if pd.isna(vix) or pd.isna(p25) or pd.isna(p65) or pd.isna(p85):
        return ("NEUTRAL", "Insuficiente histórico para percentiles (rolling 252).", macro_tomorrow)

    # Guardarraíl anti “VIX demasiado bajo”
    if cfg.use_guardrail:
        too_low_by_p10 = (pd.notna(p10) and vix < p10)
        too_low_by_floor = (pd.notna(vix) and vix < cfg.guardrail_vix_floor)
        if too_low_by_p10 or too_low_by_floor:
            return ("NEUTRAL", "Guardarraíl: VIX extremadamente bajo (snapback).", macro_tomorrow)

    # SVIX: todas
    cond_svix = (
        (vix < p25)
        and (pd.notna(ratio) and ratio < cfg.ratio_ok)
        and (contango == "CONTANGO")
        and (macro_tomorrow is False)
    )
    if cond_svix:
        return ("SVIX", "VIX < P25 + ratio VXN/VIX bajo + contango + sin macro mañana.", macro_tomorrow)

    # UVIX: mínimo 2
    uvix_cond1 = vix > p65
    uvix_cond2 = (pd.notna(ratio) and ratio > cfg.ratio_alert and ratio_up)
    uvix_cond3 = (contango == "BACKWARDATION")
    uvix_cond4 = (pd.notna(spy_ret) and spy_ret < -0.008)

    score = sum([bool(uvix_cond1), bool(uvix_cond2), bool(uvix_cond3), bool(uvix_cond4)])
    if score >= 2:
        return ("UVIX", f"Stress (score={score}) -> UVIX.", macro_tomorrow)

    # CERRAR_UVIX / preparar SVIX (tu “purple”)
    cond_close_uvix = (vix > p85) and (ratio_up is False) and (contango == "CONTANGO")
    if cond_close_uvix:
        return ("CERRAR_UVIX", "Pánico agotándose + contango vuelve. Cerrar UVIX / preparar SVIX.", macro_tomorrow)

    return ("NEUTRAL", "Régimen mixto / transición.", macro_tomorrow)


def compute_and_store_states(df_raw: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    """
    Calcula features, estado y guarda en:
      - vix_daily (tu tabla)
      - vix_signal (tu tabla)
    """
    feat = compute_features(df_raw, cfg=cfg)

    macro_df = _try_fetch_macro_events()
    feat["macro_tomorrow"] = feat["date"].apply(
        lambda d: _macro_tomorrow_flag(pd.to_datetime(d), macro_df) if pd.notna(d) else False
    )

    # construir filas de daily + signal
    daily_records: List[Dict[str, Any]] = []
    signal_records: List[Dict[str, Any]] = []

    for _, r in feat.iterrows():
        fecha_str = _iso_date(r.get("date"))
        if not fecha_str:
            continue

        # daily
        vix_reg = _vix_regime(r)
        cont = _contango_estado(r)

        daily_records.append({
            "fecha": fecha_str,                      # STRING (JSON OK)
            "vix": float(r["vix"]) if pd.notna(r.get("vix")) else None,
            "vxn": float(r["vxn"]) if pd.notna(r.get("vxn")) else None,
            "vixy": float(r["vixy"]) if pd.notna(r.get("vixy")) else None,
            "spy": float(r["spy"]) if pd.notna(r.get("spy")) else None,

            "vxn_vix_ratio": float(r["vxn_vix_ratio"]) if pd.notna(r.get("vxn_vix_ratio")) else None,

            "vix_p25": float(r["vix_p25"]) if pd.notna(r.get("vix_p25")) else None,
            "vix_p50": float(r["vix_p50"]) if pd.notna(r.get("vix_p50")) else None,
            "vix_p65": float(r["vix_p65"]) if pd.notna(r.get("vix_p65")) else None,
            "vix_p85": float(r["vix_p85"]) if pd.notna(r.get("vix_p85")) else None,

            "vix_regime": vix_reg,
            "vixy_ma_3": float(r["vixy_ma_3"]) if pd.notna(r.get("vixy_ma_3")) else None,
            "vixy_ma_10": float(r["vixy_ma_10"]) if pd.notna(r.get("vixy_ma_10")) else None,
            "contango_estado": cont,
        })

        # signal
        estado, motivo, macro_flag = decide_estado(r, cfg=cfg)
        signal_records.append({
            "fecha": fecha_str,  # PK DATE en tu SQL, mandamos string
            "estado": estado,
            "motivo": motivo,

            "vix": float(r["vix"]) if pd.notna(r.get("vix")) else None,
            "vxn_vix_ratio": float(r["vxn_vix_ratio"]) if pd.notna(r.get("vxn_vix_ratio")) else None,
            "contango_estado": cont,
            "spy_return": float(r["spy_ret"]) if pd.notna(r.get("spy_ret")) else None,

            "macro_evento": bool(macro_flag),
        })

    # upsert en vix_daily por fecha (PK)
    if daily_records:
        resp = supabase.table("vix_daily").upsert(daily_records, on_conflict="fecha").execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Error upsert vix_daily: {resp.error}")

    # upsert en vix_signal por fecha (PK)
    if signal_records:
        resp = supabase.table("vix_signal").upsert(signal_records, on_conflict="fecha").execute()
        if getattr(resp, "error", None):
            raise RuntimeError(f"Error upsert vix_signal: {resp.error}")

    return feat


# -----------------------------
# Public API (lo que usará main.py)
# -----------------------------

def run_vix_pipeline(start: str, end: str, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    raw = download_yahoo_daily(start=start, end=end)
    out = compute_and_store_states(raw, cfg=cfg)
    return out


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
