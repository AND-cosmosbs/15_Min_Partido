# backend/vix.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd

from .supabase_client import supabase


# ============================================================
# Config
# ============================================================

@dataclass
class VixConfig:
    lookback_pct: int = 252

    # Umbrales ratio VXN/VIX
    ratio_alert: float = 1.30
    ratio_ok: float = 1.25

    # Estados por percentil de VIX (rolling)
    p25: float = 0.25
    p50: float = 0.50
    p65: float = 0.65
    p85: float = 0.85

    # Guardarraíl anti “VIX demasiado bajo”
    use_guardrail: bool = True
    guardrail_vix_floor: float = 12.5  # si VIX < 12.5 => no abrir SVIX


DEFAULT_CFG = VixConfig()


# ============================================================
# Helpers (tipos / JSON safe)
# ============================================================

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _json_safe_value(x):
    # Convierte a tipos serializables por JSON/PostgREST
    if x is None:
        return None
    # pandas NA / NaN
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    # fechas
    if isinstance(x, (pd.Timestamp,)):
        return x.date().isoformat()
    if hasattr(x, "isoformat") and "date" in str(type(x)).lower():
        # datetime.date
        return x.isoformat()

    # numpy types
    try:
        import numpy as np  # noqa
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
    except Exception:
        pass

    # python scalars
    if isinstance(x, (int, float, bool, str)):
        return x

    # fallback
    try:
        return float(x)
    except Exception:
        return str(x)


def _to_records_json_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    recs = df.to_dict(orient="records")
    out: List[Dict[str, Any]] = []
    for r in recs:
        rr = {k: _json_safe_value(v) for k, v in r.items()}
        out.append(rr)
    return out


# ============================================================
# Yahoo Finance (descarga)
# ============================================================

def download_yahoo_daily(start: str, end: str) -> pd.DataFrame:
    """
    Descarga diaria (Close ajustado) de:
      ^VIX -> vix
      ^VXN -> vxn
      VIXY -> vixy
      SPY  -> spy

    Devuelve: date, vix, vxn, vixy, spy
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
            raise RuntimeError(f"No hay datos para {tkr} en Yahoo Finance ({start}..{end}).")

        if "Close" not in data.columns:
            raise RuntimeError(f"Yahoo Finance no devolvió 'Close' para {tkr}.")

        s = data["Close"].copy()
        s.name = col

        df = s.reset_index()
        # yfinance usa "Date" normalmente
        if "Date" in df.columns:
            df.rename(columns={"Date": "date"}, inplace=True)
        elif "Datetime" in df.columns:
            df.rename(columns={"Datetime": "date"}, inplace=True)
        else:
            # último recurso
            df.rename(columns={df.columns[0]: "date"}, inplace=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        if out is None:
            out = df
        else:
            out = out.merge(df, on="date", how="outer")

    assert out is not None
    out = out.sort_values("date").reset_index(drop=True)

    # Garantiza columnas
    for c in ["vix", "vxn", "vixy", "spy"]:
        if c not in out.columns:
            out[c] = pd.NA

    return out


# ============================================================
# Features + estados (para tus tablas)
# ============================================================

def compute_features(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    """
    df: date, vix, vxn, vixy, spy
    añade:
      - vxn_vix_ratio
      - spy_return
      - vix_p25/p50/p65/p85 (rolling 252)
      - vixy_ma_3 / vixy_ma_10
      - contango_estado (CONTANGO/TRANSICION/BACKWARDATION)
      - vix_regime (CALMA/ALERTA/TENSION/PANICO)
    """
    w = df.copy()

    for c in ["vix", "vxn", "vixy", "spy"]:
        w[c] = _safe_num(w[c])

    w["spy_return"] = w["spy"].pct_change()
    w["vxn_vix_ratio"] = w["vxn"] / w["vix"]

    lb = int(cfg.lookback_pct)
    w["vix_p25"] = w["vix"].rolling(lb).quantile(cfg.p25)
    w["vix_p50"] = w["vix"].rolling(lb).quantile(cfg.p50)
    w["vix_p65"] = w["vix"].rolling(lb).quantile(cfg.p65)
    w["vix_p85"] = w["vix"].rolling(lb).quantile(cfg.p85)

    w["vixy_ma_3"] = w["vixy"].rolling(3).mean()
    w["vixy_ma_10"] = w["vixy"].rolling(10).mean()

    # contango proxy
    def _contango_state(ma3, ma10) -> Optional[str]:
        if pd.isna(ma3) or pd.isna(ma10):
            return None
        # tolerancia pequeña
        if ma3 < ma10 * 0.995:
            return "CONTANGO"
        if ma3 > ma10 * 1.005:
            return "BACKWARDATION"
        return "TRANSICION"

    w["contango_estado"] = w.apply(lambda r: _contango_state(r["vixy_ma_3"], r["vixy_ma_10"]), axis=1)

    # vix regime por percentiles
    def _vix_regime(vix, p25, p50, p65, p85) -> Optional[str]:
        if pd.isna(vix) or pd.isna(p25) or pd.isna(p50) or pd.isna(p65) or pd.isna(p85):
            return None
        if vix < p25:
            return "CALMA"
        if vix < p65:
            return "ALERTA"
        if vix < p85:
            return "TENSION"
        return "PANICO"

    w["vix_regime"] = w.apply(lambda r: _vix_regime(r["vix"], r["vix_p25"], r["vix_p50"], r["vix_p65"], r["vix_p85"]), axis=1)

    return w


def decide_state_row(row: pd.Series, cfg: VixConfig = DEFAULT_CFG) -> Tuple[str, str, bool]:
    """
    Devuelve:
      estado: 'SVIX' | 'NEUTRAL' | 'UVIX' | 'CERRAR_UVIX'
      motivo: texto
      macro_evento: bool (de momento siempre False si no tienes macro integrado aquí)
    """
    vix = row.get("vix")
    ratio = row.get("vxn_vix_ratio")
    contango = row.get("contango_estado")
    spy_ret = row.get("spy_return")

    p25 = row.get("vix_p25")
    p65 = row.get("vix_p65")
    p85 = row.get("vix_p85")

    macro_evento = False  # (lo dejaremos así para que no falle nada)

    # Sin percentiles aún => no operar
    if pd.isna(p25) or pd.isna(p65) or pd.isna(p85) or pd.isna(vix):
        return "NEUTRAL", "Insuficiente histórico (rolling 252) para percentiles.", macro_evento

    # Guardarraíl (tu preocupación “suicidio” con VIX mínimo)
    if cfg.use_guardrail and pd.notna(vix) and vix < cfg.guardrail_vix_floor:
        return "NEUTRAL", f"Guardarraíl: VIX muy bajo (< {cfg.guardrail_vix_floor}).", macro_evento

    # SVIX: régimen calmado + ratio ok + contango
    cond_svix = (
        (vix < p25)
        and (pd.notna(ratio) and ratio < cfg.ratio_ok)
        and (contango == "CONTANGO")
        and (macro_evento is False)
    )
    if cond_svix:
        return "SVIX", "CALMA (VIX < P25) + ratio OK + CONTANGO + sin macro.", macro_evento

    # UVIX: mínimo 2 señales de stress
    uv1 = bool(vix > p65)
    uv2 = bool(pd.notna(ratio) and ratio > cfg.ratio_alert)
    uv3 = bool(contango == "BACKWARDATION")
    uv4 = bool(pd.notna(spy_ret) and spy_ret < -0.008)

    score = sum([uv1, uv2, uv3, uv4])
    if score >= 2:
        return "UVIX", f"Stress score={score} (VIX>P65 / ratio / backwardation / SPY<-0.8%).", macro_evento

    # Cerrar UVIX: pánico se agota + vuelve contango
    if (vix > p85) and (contango == "CONTANGO"):
        return "CERRAR_UVIX", "VIX>P85 pero vuelve CONTANGO (posible agotamiento pánico).", macro_evento

    return "NEUTRAL", "Régimen mixto / transición.", macro_evento


def compute_signal(df_feat: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    """
    Genera DF para tabla vix_signal:
      fecha, estado, motivo, vix, vxn_vix_ratio, contango_estado, spy_return, macro_evento
    """
    w = df_feat.copy()

    estados: List[str] = []
    motivos: List[str] = []
    macros: List[bool] = []

    for _, r in w.iterrows():
        estado, motivo, macro = decide_state_row(r, cfg=cfg)
        estados.append(estado)
        motivos.append(motivo)
        macros.append(bool(macro))

    w["estado"] = estados
    w["motivo"] = motivos
    w["macro_evento"] = macros

    sig = pd.DataFrame({
        "fecha": pd.to_datetime(w["date"], errors="coerce").dt.date,
        "estado": w["estado"],
        "motivo": w["motivo"],
        "vix": w["vix"],
        "vxn_vix_ratio": w["vxn_vix_ratio"],
        "contango_estado": w["contango_estado"],
        "spy_return": w["spy_return"],
        "macro_evento": w["macro_evento"],
    })

    return sig


# ============================================================
# Supabase IO (tus tablas reales)
# ============================================================

def upsert_vix_daily(df_feat: pd.DataFrame) -> int:
    """
    Upsert en vix_daily (según TU SQL):
      fecha (PK),
      vix, vxn, vixy, spy,
      vxn_vix_ratio,
      vix_p25, vix_p50, vix_p65, vix_p85,
      vix_regime,
      vixy_ma_3, vixy_ma_10,
      contango_estado
    """
    if df_feat is None or df_feat.empty:
        return 0

    w = df_feat.copy()
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

    records = _to_records_json_safe(w)

    resp = supabase.table("vix_daily").upsert(records, on_conflict="fecha").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error upsert vix_daily: {resp.error}")

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


def upsert_vix_signal(df_signal: pd.DataFrame) -> int:
    """
    Upsert en vix_signal (según TU SQL):
      fecha (PK), estado (NOT NULL), motivo,
      vix, vxn_vix_ratio, contango_estado, spy_return, macro_evento
    """
    if df_signal is None or df_signal.empty:
        return 0

    w = df_signal.copy()

    # seguridad: estado no puede ser null
    if "estado" in w.columns:
        w["estado"] = w["estado"].fillna("NEUTRAL")
    else:
        w["estado"] = "NEUTRAL"

    records = _to_records_json_safe(w)

    resp = supabase.table("vix_signal").upsert(records, on_conflict="fecha").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(f"Error upsert vix_signal: {resp.error}")

    return len(records)


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
# Pipeline 1-click (lo que llamará tu botón en la app)
# ============================================================

def run_vix_pipeline(start: str, end: str, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    """
    1) descarga Yahoo
    2) calcula features
    3) upsert vix_daily
    4) calcula signal
    5) upsert vix_signal
    Devuelve df_signal
    """
    raw = download_yahoo_daily(start=start, end=end)
    feat = compute_features(raw, cfg=cfg)

    # Guardar daily
    upsert_vix_daily(feat)

    # Guardar signal
    sig = compute_signal(feat, cfg=cfg)
    upsert_vix_signal(sig)

    return sig
