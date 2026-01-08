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

    # --- UVIX: MODO B (raro pero operativo 2–5 trades/año) ---
    # caída SPY mínima para considerar “pánico”
    uvix_spy_panic_ret: float = -0.012  # -1.2% día SPY

    # VIX debe ser extremo (p85) y además alto en absoluto (pero no “imposible”)
    uvix_vix_abs_floor: float = 24.0    # VIX >= 24

    # ratio VXN/VIX debe estar alto y subiendo (stress tech)
    uvix_ratio_floor: float = 1.30      # ratio >= 1.30

    # estructura tensa (sin margen extra en modo B)
    uvix_struct_margin: float = 1.00    # MA3 > MA10 * 1.00


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
        # yfinance a veces devuelve MultiIndex
        if ("Close" in df.columns.get_level_values(0)) or ("close" in df.columns.get_level_values(0)):
            try:
                s = df["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                return s
            except Exception:
                pass

    for c in ["Close", "close", "Adj Close", "adjclose", "AdjClose"]:
        if c in cols:
            s = df[c]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            return s

    raise RuntimeError(f"No se encontró columna de cierre en Yahoo. Columnas: {cols}")


def _ensure_expected_columns(out: pd.DataFrame, expected: Iterable[str]) -> None:
    missing = [c for c in expected if c not in out.columns]
    if missing:
        raise RuntimeError(
            "Descarga/merge incompleto. Faltan columnas: "
            f"{missing}. Columnas presentes: {list(out.columns)}"
        )


def _series1d(x: Any) -> pd.Series:
    """
    Convierte a Series 1D (por si yfinance devuelve DataFrame).
    """
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return pd.Series(x)


def _json_sanitize_value(x: Any) -> Any:
    if x is None:
        return None

    if pd.isna(x):
        return None

    if isinstance(x, pd.Timestamp):
        return x.date().isoformat()

    if hasattr(x, "isoformat") and ("date" in str(type(x)).lower() or "datetime" in str(type(x)).lower()):
        try:
            return x.isoformat()
        except Exception:
            return str(x)

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


def _supabase_select_all(
    table: str,
    order_col: str,
    asc: bool = True,
    batch_size: int = 1000,
) -> List[Dict[str, Any]]:
    """
    PostgREST/Supabase devuelve 1000 filas por defecto si no paginamos.
    Esto descarga TODAS las filas por bloques.
    """
    out: List[Dict[str, Any]] = []
    start = 0

    while True:
        end = start + batch_size - 1
        resp = (
            supabase.table(table)
            .select("*")
            .order(order_col, desc=(not asc))
            .range(start, end)
            .execute()
        )
        if getattr(resp, "error", None):
            raise RuntimeError(resp.error)

        data = getattr(resp, "data", None) or []
        if not data:
            break

        out.extend(data)

        if len(data) < batch_size:
            break

        start += batch_size

    return out


# -----------------------------
# Supabase: macro events
# -----------------------------

def fetch_macro_events() -> pd.DataFrame:
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
    Descarga diaria de: ^VIX, ^VXN, proxy contango VXX (guardado en columna vixy), SPY
    Devuelve columnas: date, vix, vxn, vixy, spy
    """
    import yfinance as yf

    tickers = {
        "^VIX": "vix",
        "^VXN": "vxn",
        "VXX": "vixy",   # guardamos VXX en vixy (schema estable)
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

        df_one = pd.DataFrame({"date": data["date"], col: _series1d(close).values})
        df_one["date"] = pd.to_datetime(df_one["date"], errors="coerce").dt.normalize()

        if out is None:
            out = df_one
        else:
            out = out.merge(df_one, on="date", how="outer")

    assert out is not None
    out = out.sort_values("date").reset_index(drop=True)

    _ensure_expected_columns(out, ["date", "vix", "vxn", "vixy", "spy"])
    return out


def download_trade_ohlc(start: str, end: str) -> pd.DataFrame:
    """
    Descarga OHLC de SVIX y UVIX.
    Devuelve: date, svix_open/high/low/close, uvix_open/high/low/close
    """
    import yfinance as yf

    tickers = {
        "SVIX": "svix",
        "UVIX": "uvix",
    }

    out: Optional[pd.DataFrame] = None

    for tkr, prefix in tickers.items():
        data = yf.download(
            tkr,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="column",
            actions=False,
        )

        if data is None or data.empty:
            raise RuntimeError(f"No hay datos OHLC para {tkr} en Yahoo en el rango {start}..{end}")

        data = _normalize_date_index(data)

        for needed in ["Open", "High", "Low", "Close"]:
            if needed not in data.columns:
                raise RuntimeError(f"{tkr} no trae columna {needed}. Columnas: {list(data.columns)}")

        o = _series1d(data["Open"])
        h = _series1d(data["High"])
        l = _series1d(data["Low"])
        c = _series1d(data["Close"])

        df_one = pd.DataFrame({
            "date": pd.to_datetime(data["date"], errors="coerce").dt.normalize(),
            f"{prefix}_open": pd.to_numeric(o, errors="coerce").values,
            f"{prefix}_high": pd.to_numeric(h, errors="coerce").values,
            f"{prefix}_low": pd.to_numeric(l, errors="coerce").values,
            f"{prefix}_close": pd.to_numeric(c, errors="coerce").values,
        })

        if out is None:
            out = df_one
        else:
            out = out.merge(df_one, on="date", how="outer")

    assert out is not None
    out = out.sort_values("date").reset_index(drop=True)
    return out


# -----------------------------
# Señales y estado
# -----------------------------

def compute_features(df: pd.DataFrame, cfg: Optional[VixConfig] = None) -> pd.DataFrame:
    cfg = cfg or DEFAULT_CFG

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


def decide_state_row(row: pd.Series, cfg: Optional[VixConfig] = None) -> Dict[str, Any]:
    cfg = cfg or DEFAULT_CFG

    vix = row.get("vix")
    p10 = row.get("vix_p10")
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

    # ---------------------------------------------------------
    # SVIX (A1: más selectivo => p10 si existe, si no p25)
    # + A2: filtro SPY (no abrir SVIX si SPY cae fuerte)
    # ---------------------------------------------------------
    svix_thr = p10 if pd.notna(p10) else p25
    spy_filter_ok = True
    if pd.notna(spy_ret):
        spy_filter_ok = float(spy_ret) > -0.007  # -0.7% => evitamos abrir SVIX en día feo

    cond_svix = (
        (pd.notna(vix) and pd.notna(svix_thr) and (vix < svix_thr))
        and (pd.notna(ratio) and ratio < cfg.ratio_ok)
        and contango_ok
        and (macro_tomorrow is False)
        and spy_filter_ok
    )
    if cond_svix:
        return {"estado": "SVIX", "accion": "OPEN/HOLD SVIX", "comentario": "Calma extrema + contango + SPY ok + sin macro mañana."}

    # ---------------------------------------------------------
    # UVIX (B: raro pero operativo 2–5 trades/año)
    # Base: VIX > p85 + suelo absoluto VIX >= abs_floor
    # Trigger: 2 de 3 (ratio, estructura, crash SPY)
    # + filtro macro mañana
    # ---------------------------------------------------------
    uvix_vix_extreme = (pd.notna(vix) and pd.notna(p85) and (float(vix) > float(p85)))
    uvix_vix_abs = (pd.notna(vix) and float(vix) >= float(cfg.uvix_vix_abs_floor))
    uvix_base = uvix_vix_extreme and uvix_vix_abs

    uvix_cond_ratio = (pd.notna(ratio) and float(ratio) >= float(cfg.uvix_ratio_floor) and ratio_up)

    ma3 = row.get("vixy_ma_3")
    ma10 = row.get("vixy_ma_10")
    uvix_cond_struct = (
        pd.notna(ma3) and pd.notna(ma10)
        and float(ma3) > float(ma10) * float(cfg.uvix_struct_margin)
    )

    uvix_cond_spy = (pd.notna(spy_ret) and float(spy_ret) <= float(cfg.uvix_spy_panic_ret))

    uvix_score = int(uvix_cond_ratio) + int(uvix_cond_struct) + int(uvix_cond_spy)
    uvix_macro_ok = (macro_tomorrow is False)

    if uvix_base and uvix_macro_ok and uvix_score >= 2:
        return {
            "estado": "UVIX",
            "accion": "OPEN/HOLD UVIX",
            "comentario": f"UVIX B: VIX extremo+abs + score={uvix_score}/3 (ratio/struct/spy), sin macro mañana.",
        }

    # PREP_SVIX
    cond_prep = (pd.notna(vix) and pd.notna(p85) and (vix > p85)) and (ratio_up is False) and contango_ok
    if cond_prep:
        return {"estado": "PREP_SVIX", "accion": "WAIT / PREPARE SVIX", "comentario": "Pánico se agota + contango vuelve."}

    return {"estado": "NEUTRAL", "accion": "NO NEW POSITION", "comentario": "Régimen mixto / transición."}


def compute_states(df_feat: pd.DataFrame, cfg: Optional[VixConfig] = None) -> pd.DataFrame:
    cfg = cfg or DEFAULT_CFG

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
        "svix_open", "svix_high", "svix_low", "svix_close",
        "uvix_open", "uvix_high", "uvix_low", "uvix_close",
    ]
    w = w[[c for c in keep_cols if c in w.columns]].copy()

    records: List[Dict[str, Any]] = w.to_dict(orient="records")
    records = _json_sanitize_records(records)

    resp = supabase.table("vix_daily").upsert(records, on_conflict="fecha").execute()
    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)

    return len(records)


def fetch_vix_daily() -> pd.DataFrame:
    data = _supabase_select_all(table="vix_daily", order_col="fecha", asc=True, batch_size=1000)
    df = pd.DataFrame(data)
    if not df.empty and "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


# -----------------------------
# Órdenes VIX (Supabase)
# -----------------------------

def fetch_vix_orders(limit: int = 300) -> pd.DataFrame:
    if limit <= 1000:
        resp = supabase.table("vix_orders").select("*").order("fecha", desc=True).limit(limit).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(resp.error)
        data = getattr(resp, "data", None) or []
        df = pd.DataFrame(data)
        if not df.empty and "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        return df

    data = _supabase_select_all(table="vix_orders", order_col="fecha", asc=False, batch_size=1000)
    df = pd.DataFrame(data)
    if not df.empty and "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


def insert_vix_order(
    fecha,
    ticker: str,
    side: str,
    qty: float,
    price: Optional[float] = None,
    status: str = "PLANNED",
    notes: Optional[str] = None,
    estado_signal: Optional[str] = None,
) -> int:
    payload: Dict[str, Any] = {
        "fecha": pd.to_datetime(fecha, errors="coerce").date().isoformat() if fecha is not None else None,
        "ticker": ticker,
        "side": side,
        "qty": float(qty),
        "price": float(price) if price is not None else None,
        "status": status,
        "notes": notes,
        "estado_signal": estado_signal,
    }
    payload = {k: _json_sanitize_value(v) for k, v in payload.items()}

    resp = supabase.table("vix_orders").insert(payload).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)

    data = getattr(resp, "data", None) or []
    if data and isinstance(data, list) and "id" in data[0]:
        return int(data[0]["id"])
    return 1


def update_vix_order_status(
    order_id: int,
    status: str,
    price: Optional[float] = None,
    notes: Optional[str] = None,
) -> None:
    patch: Dict[str, Any] = {"status": status}
    if price is not None:
        patch["price"] = float(price)
    if notes is not None:
        patch["notes"] = notes

    patch = {k: _json_sanitize_value(v) for k, v in patch.items()}

    resp = supabase.table("vix_orders").update(patch).eq("id", int(order_id)).execute()
    if getattr(resp, "error", None):
        raise RuntimeError(resp.error)


# -----------------------------
# Pipeline 1-click
# -----------------------------

def run_vix_pipeline(start: str, end: str, cfg: Optional[VixConfig] = None) -> pd.DataFrame:
    cfg = cfg or DEFAULT_CFG

    raw = download_yahoo_daily(start=start, end=end)
    feat = compute_features(raw, cfg=cfg)
    out = compute_states(feat, cfg=cfg)

    ohlc = download_trade_ohlc(start=start, end=end)
    if ohlc is not None and not ohlc.empty:
        out = out.merge(ohlc, on="date", how="left")

    upsert_vix_daily(out)
    return out
