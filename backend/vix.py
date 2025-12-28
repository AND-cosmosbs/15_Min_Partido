# backend/vix.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

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

    use_guardrail: bool = True
    guardrail_p10: float = 0.10
    guardrail_vix_floor: float = 12.5  # si VIX < 12.5 => no abrir SVIX


DEFAULT_CFG = VixConfig()


# -----------------------------
# Helpers
# -----------------------------
def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _clean_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte NaN/NA -> None y normaliza tipos para que Supabase no rompa:
    - fechas -> string YYYY-MM-DD cuando toque
    - boolean -> bool real
    """
    out = df.copy()

    # NaN/NA -> None
    out = out.where(pd.notna(out), None)

    # bool pandas -> bool python
    for c in out.columns:
        if out[c].dtype == "boolean":
            out[c] = out[c].astype(object).apply(lambda x: bool(x) if x is not None else None)

    return out


# -----------------------------
# Yahoo download
# -----------------------------
def download_yahoo_daily(start: str, end: str) -> pd.DataFrame:
    """
    Descarga diaria de: ^VIX, ^VXN, VIXY, SPY
    Devuelve df con columnas: date, vix, vxn, vixy, spy
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
        )
        if data is None or data.empty:
            raise RuntimeError(f"No hay datos para {tkr} en Yahoo en rango {start}..{end}")

        s = data["Close"].copy()
        s.name = col

        df = s.reset_index()
        df.rename(columns={"Date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        out = df if out is None else out.merge(df, on="date", how="outer")

    out = out.sort_values("date").reset_index(drop=True)
    return out


# -----------------------------
# Features + estado
# -----------------------------
def compute_features(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df.copy()

    for c in ["vix", "vxn", "vixy", "spy"]:
        w[c] = _safe_num(w[c])

    w["spy_ret"] = w["spy"].pct_change()
    w["vxn_vix_ratio"] = w["vxn"] / w["vix"]
    w["ratio_up"] = w["vxn_vix_ratio"].diff() > 0

    lb = int(cfg.lookback_pct)
    w["vix_p10"] = w["vix"].rolling(lb).quantile(0.10)
    w["vix_p25"] = w["vix"].rolling(lb).quantile(0.25)
    w["vix_p50"] = w["vix"].rolling(lb).quantile(0.50)
    w["vix_p65"] = w["vix"].rolling(lb).quantile(0.65)
    w["vix_p85"] = w["vix"].rolling(lb).quantile(0.85)

    w["vixy_ma_3"] = w["vixy"].rolling(3).mean()
    w["vixy_ma_10"] = w["vixy"].rolling(10).mean()

    return w


def _contango_estado(row: pd.Series) -> Optional[str]:
    ma3 = row.get("vixy_ma_3")
    ma10 = row.get("vixy_ma_10")
    if ma3 is None or ma10 is None:
        return None
    try:
        if ma3 < ma10:
            return "CONTANGO"
        if ma3 > ma10:
            return "BACKWARDATION"
        return "TRANSICION"
    except Exception:
        return None


def decide_signal(row: pd.Series, cfg: VixConfig = DEFAULT_CFG) -> Dict[str, Any]:
    """
    Devuelve estado de trading (SVIX/NEUTRAL/UVIX/PREP_SVIX) + motivo.
    NOTA: aquí no meto macro real (tu tabla macro_events no está en tus SQL finales).
    """
    vix = row.get("vix")
    p10 = row.get("vix_p10")
    p25 = row.get("vix_p25")
    p65 = row.get("vix_p65")
    p85 = row.get("vix_p85")
    ratio = row.get("vxn_vix_ratio")
    ratio_up = bool(row.get("ratio_up")) if row.get("ratio_up") is not None else False
    spy_ret = row.get("spy_ret")

    # sin percentiles todavía
    if vix is None or p25 is None or p65 is None or p85 is None:
        return {"estado": "NEUTRAL", "motivo": "Insuficiente histórico (rolling 252 no disponible aún)."}

    # guardarraíl VIX demasiado bajo
    if cfg.use_guardrail:
        too_low_by_p10 = (p10 is not None and vix < p10)
        too_low_by_floor = (vix is not None and vix < cfg.guardrail_vix_floor)
        if too_low_by_p10 or too_low_by_floor:
            return {"estado": "NEUTRAL", "motivo": "Guardarraíl: VIX extremadamente bajo (snapback risk)."}

    cont_estado = _contango_estado(row)
    contango_ok = (cont_estado == "CONTANGO")

    # SVIX
    if (vix < p25) and (ratio is not None and ratio < cfg.ratio_ok) and contango_ok:
        return {"estado": "SVIX", "motivo": "VIX < P25 + ratio VXN/VIX OK + contango estable."}

    # UVIX (2 de 4)
    uv1 = vix > p65
    uv2 = (ratio is not None and ratio > cfg.ratio_alert and ratio_up)
    uv3 = (cont_estado == "BACKWARDATION")
    uv4 = (spy_ret is not None and spy_ret < -0.008)
    score = sum([bool(uv1), bool(uv2), bool(uv3), bool(uv4)])

    if score >= 2:
        return {"estado": "UVIX", "motivo": f"Stress score={score} (VIX/ratio/contango/SPY)."}

    # PREP_SVIX
    if (vix > p85) and (ratio_up is False) and contango_ok:
        return {"estado": "PREP_SVIX", "motivo": "Pánico agotándose + contango vuelve (preparar SVIX)."}

    return {"estado": "NEUTRAL", "motivo": "Régimen mixto / transición."}


def compute_all(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = compute_features(df, cfg=cfg)

    # columnas de salida “daily”
    w["contango_estado"] = w.apply(_contango_estado, axis=1)

    # señal diaria (para vix_signal)
    sig_estado: List[str] = []
    sig_motivo: List[str] = []
    for _, r in w.iterrows():
        res = decide_signal(r, cfg=cfg)
        sig_estado.append(res["estado"])
        sig_motivo.append(res["motivo"])

    w["signal_estado"] = sig_estado
    w["signal_motivo"] = sig_motivo

    return w


# -----------------------------
# Supabase write/read
# -----------------------------
def upsert_vix_daily(df_all: pd.DataFrame) -> int:
    """
    Escribe en public.vix_daily (solo columnas existentes en tu tabla).
    """
    if df_all is None or df_all.empty:
        return 0

    w = df_all.copy()
    w["fecha"] = pd.to_datetime(w["date"], errors="coerce").dt.date.astype(str)

    keep = [
        "fecha",
        "vix", "vxn", "vixy", "spy",
        "vxn_vix_ratio",
        "vix_p10", "vix_p25", "vix_p50", "vix_p65", "vix_p85",
        "vixy_ma_3", "vixy_ma_10",
        "contango_estado",
        "spy_ret",
        "macro_tomorrow",  # si existe
    ]
    w = w[[c for c in keep if c in w.columns]].copy()
    w = _clean_for_json(w)

    records = w.to_dict(orient="records")
    resp = supabase.table("vix_daily").upsert(records, on_conflict="fecha").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error upsert vix_daily: {resp.error}")
    return len(records)


def upsert_vix_signal(df_all: pd.DataFrame) -> int:
    """
    Escribe en public.vix_signal (TABLA, no VIEW).
    """
    if df_all is None or df_all.empty:
        return 0

    w = df_all.copy()
    w["fecha"] = pd.to_datetime(w["date"], errors="coerce").dt.date.astype(str)

    out = pd.DataFrame({
        "fecha": w["fecha"],
        "estado": w["signal_estado"],
        "motivo": w["signal_motivo"],
        "vix": w.get("vix"),
        "vxn_vix_ratio": w.get("vxn_vix_ratio"),
        "contango_estado": w.get("contango_estado"),
        "spy_return": w.get("spy_ret"),
        "macro_evento": w.get("macro_tomorrow", False),
    })

    out = _clean_for_json(out)

    records = out.to_dict(orient="records")
    resp = supabase.table("vix_signal").upsert(records, on_conflict="fecha").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error upsert vix_signal: {resp.error}")
    return len(records)


def fetch_vix_daily() -> pd.DataFrame:
    resp = supabase.table("vix_daily").select("*").order("fecha", desc=False).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error leyendo vix_daily: {resp.error}")
    df = pd.DataFrame(getattr(resp, "data", None) or [])
    if not df.empty and "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


def fetch_vix_signal() -> pd.DataFrame:
    resp = supabase.table("vix_signal").select("*").order("fecha", desc=False).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error leyendo vix_signal: {resp.error}")
    df = pd.DataFrame(getattr(resp, "data", None) or [])
    if not df.empty and "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


# -----------------------------
# Pipeline
# -----------------------------
def run_vix_pipeline(start: str, end: str, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    raw = download_yahoo_daily(start=start, end=end)
    all_df = compute_all(raw, cfg=cfg)

    # escribe a tablas (vix_daily + vix_signal)
    upsert_vix_daily(all_df)
    upsert_vix_signal(all_df)

    return all_df
