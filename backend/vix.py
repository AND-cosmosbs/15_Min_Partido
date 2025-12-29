# backend/vix.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Iterable
import datetime as dt

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

    # yfinance suele devolver índice Date/Datetime
    if isinstance(out.index, (pd.DatetimeIndex,)):
        out = out.reset_index()
        # el nombre puede ser Date o Datetime
        if "Date" in out.columns:
            out.rename(columns={"Date": "date"}, inplace=True)
        elif "Datetime" in out.columns:
            out.rename(columns={"Datetime": "date"}, inplace=True)
        else:
            # último recurso: primera columna
            out.rename(columns={out.columns[0]: "date"}, inplace=True)

        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        return out

    # si llega aquí, no hay forma razonable
    raise RuntimeError("No se pudo detectar columna/índice de fecha en la descarga de Yahoo.")


def _pick_close_column(df: pd.DataFrame) -> pd.Series:
    """
    yfinance con auto_adjust=True debería traer 'Close'.
    Pero por robustez, aceptamos varias variantes.
    """
    cols = list(df.columns)

    # MultiIndex (a veces)
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close" in df.columns.get_level_values(0)) or ("close" in df.columns.get_level_values(0)):
            try:
                s = df["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                return s
            except Exception:
                pass

    # Single index
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
    """
    Convierte valores no serializables (NAType, Timestamp, numpy types, date/datetime) a JSON-safe.
    """
    if x is None:
        return None

    # pandas NA/NaN
    try:
        if pd.isna(x):
            return None
    except Exception:
        # si pd.isna() revienta con algún tipo raro, seguimos
        pass

    # pandas Timestamp / numpy datetime64 / python date/datetime
    if isinstance(x, pd.Timestamp):
        if pd.isna(x):
            return None
        return x.to_pydatetime().date().isoformat()

    if isinstance(x, np.datetime64):
        try:
            return pd.to_datetime(x).to_pydatetime().date().isoformat()
        except Exception:
            return str(x)

    if isinstance(x, (dt.datetime, dt.date)):
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

    # python floats/ints/bools/strings ya son JSON-safe
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

def download_yahoo_daily(start: str, end: str) -> pd.DataFrame:
    """
    Descarga diaria de: ^VIX, ^VXN, VIXY, SPY
    Devuelve columnas: date, vix, vxn, vixy, spy
    """
    import yfinance as yf

    tickers = {
        "^VIX": "vix",
        "^VXN": "vxn",
        "VXX": "vixy",
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
# Señales y estado
# -----------------------------

def compute_features(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df.copy()

    _ensure_expected_columns(w, ["date", "vix", "vxn", "vixy", "spy"])

    w["vix"] = _safe_num_series(w["vix"])
    w["vxn"] = _safe_num_series(w["vxn"])
    w["vixy"] = _safe_num_series(w["vixy"])
    w["spy"] = _safe_num_series(w["spy"])

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

    w["contango_ok"] = w["vixy_ma_3"] < w["vixy_ma_10"]
    return w


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

    if pd.isna(p25) or pd.isna(p65) or pd.isna(p85) or pd.isna(vix):
        return {"estado": "NEUTRAL", "accion": "NO DATA", "comentario": "Insuficiente histórico para rolling 252."}

    if cfg.use_guardrail and pd.notna(vix) and float(vix) < float(cfg.guardrail_vix_floor):
        return {
            "estado": "NEUTRAL",
            "accion": "NO OPEN SVIX",
            "comentario": "Guardarraíl: VIX extremadamente bajo (riesgo snapback).",
        }

    cond_svix = (
        (vix < p25)
        and (pd.notna(ratio) and ratio < cfg.ratio_ok)
        and contango_ok
        and (macro_tomorrow is False)
    )
    if cond_svix:
        return {"estado": "SVIX", "accion": "OPEN/HOLD SVIX", "comentario": "Calma + contango + sin macro mañana."}

    uvix_cond1 = vix > p65
    uvix_cond2 = (pd.notna(ratio) and ratio > cfg.ratio_alert and ratio_up)
    uvix_cond3 = (pd.notna(row.get("vixy_ma_3")) and pd.notna(row.get("vixy_ma_10")) and (row.get("vixy_ma_3") > row.get("vixy_ma_10")))
    uvix_cond4 = (pd.notna(spy_ret) and spy_ret < -0.008)

    uvix_score = sum([bool(uvix_cond1), bool(uvix_cond2), bool(uvix_cond3), bool(uvix_cond4)])
    if uvix_score >= 2:
        return {"estado": "UVIX", "accion": "TRADE UVIX (SHORT)", "comentario": f"Stress score={uvix_score}."}

    cond_purple = (vix > p85) and (ratio_up is False) and contango_ok
    if cond_purple:
        return {"estado": "PREP_SVIX", "accion": "CLOSE UVIX / PREPARE SVIX", "comentario": "Pánico se agota + contango vuelve."}

    return {"estado": "NEUTRAL", "accion": "NO NEW POSITION", "comentario": "Régimen mixto / transición."}


def compute_states(df_feat: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df_feat.copy()

    macro = fetch_macro_events()
    w["macro_tomorrow"] = w["date"].apply(
        lambda d: macro_tomorrow_flag(pd.to_datetime(d), macro) if pd.notna(d) else False
    )

    estados, acciones, comentarios = [], [], []
    for _, r in w.iterrows():
        res = decide_state_row(r, cfg=cfg)
        estados.append(res["estado"])
        acciones.append(res["accion"])
        comentarios.append(res["comentario"])

    w["estado"] = estados
    w["accion"] = acciones
    w["comentario"] = comentarios
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
    raw = download_yahoo_daily(start=start, end=end)
    feat = compute_features(raw, cfg=cfg)
    out = compute_states(feat, cfg=cfg)
    upsert_vix_daily(out)
    return out
