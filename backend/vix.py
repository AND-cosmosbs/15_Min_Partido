# backend/vix.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Iterable, Tuple
import re

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

    # =========================
    # A) Confirmación anti-ruido
    # =========================
    confirm_days: int = 2  # requiere N días seguidos del mismo "raw_estado" para cambiar

    # =========================
    # B) Cooldown anti-latigazos
    # =========================
    cooldown_days: int = 3  # tras un cambio "real" de régimen, no volver a cambiar durante N días

    # =========================
    # C) Anti-flip SVIX<->UVIX
    # =========================
    anti_flip: bool = True  # si True, impide pasar directo SVIX<->UVIX; obliga a pasar por NEUTRAL

    # UVIX solo pánico real
    uvix_spy_panic_thresh: float = -0.012  # -1.2% (spy_ret <= -0.012)
    uvix_min_score: int = 3  # requiere score >= 3
    uvix_require_vix_extreme: bool = True  # requiere vix > p85


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
    # último recurso
    return pd.Series(x)


def _json_sanitize_value(x: Any) -> Any:
    if x is None:
        return None

    if pd.isna(x):
        return None

    # pandas Timestamp
    if isinstance(x, pd.Timestamp):
        return x.date().isoformat()

    # datetime/date de python
    if hasattr(x, "isoformat") and ("date" in str(type(x)).lower() or "datetime" in str(type(x)).lower()):
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

        # Si devolvió menos de batch_size, ya no hay más.
        if len(data) < batch_size:
            break

        start += batch_size

    return out


def _extract_missing_column_from_postgrest_error(msg: str) -> Optional[str]:
    """
    Ej:
    "Could not find the 'vxx' column of 'vix_daily' in the schema cache"
    """
    if not msg:
        return None
    m = re.search(r"Could not find the '([^']+)' column", msg)
    if m:
        return m.group(1)
    return None


def _upsert_with_schema_fallback(table: str, records: List[Dict[str, Any]], on_conflict: str) -> None:
    """
    Si Supabase/PostgREST se queja de una columna que no existe en el schema cache,
    reintentamos quitando esa columna de todos los records.
    """
    if not records:
        return

    # Reintento en bucle por si hay varias columnas conflictivas
    max_tries = 8
    current = records

    for _ in range(max_tries):
        resp = supabase.table(table).upsert(current, on_conflict=on_conflict).execute()
        err = getattr(resp, "error", None)
        if not err:
            return

        err_str = str(err)
        missing = _extract_missing_column_from_postgrest_error(err_str)
        if not missing:
            raise RuntimeError(err)

        # quitamos esa col y reintentamos
        new_records: List[Dict[str, Any]] = []
        for r in current:
            if missing in r:
                r = dict(r)
                r.pop(missing, None)
            new_records.append(r)

        current = new_records

    # si agota reintentos
    raise RuntimeError("Upsert falló tras reintentos por columnas inexistentes en schema cache.")


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
        "VXX": "vixy",   # OJO: guardamos VXX en columna vixy (para no tocar schema)
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
    Descarga OHLC de los tickers operables (SVIX, UVIX) para backtest de stops intradía.
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
            auto_adjust=False,  # OHLC real
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
# Señales y estado (RAW)
# -----------------------------

def compute_features(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    w = df.copy()
    _ensure_expected_columns(w, ["date", "vix", "vxn", "vixy", "spy"])

    w["vix"] = _safe_num_series(w["vix"])
    w["vxn"] = _safe_num_series(w["vxn"])
    w["vixy"] = _safe_num_series(w["vixy"])
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

    # MA sobre vixy (que ahora contiene VXX como proxy)
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

    # SVIX (calma + contango + sin macro mañana)
    cond_svix = (
        (vix < p25)
        and (pd.notna(ratio) and ratio < cfg.ratio_ok)
        and contango_ok
        and (macro_tomorrow is False)
    )
    if cond_svix:
        return {"estado": "SVIX", "accion": "OPEN/HOLD SVIX", "comentario": "Calma + contango + sin macro mañana."}

    # UVIX: condiciones base (score)
    uvix_cond1 = vix > p65
    uvix_cond2 = (pd.notna(ratio) and ratio > cfg.ratio_alert and ratio_up)
    uvix_cond3 = (
        pd.notna(row.get("vixy_ma_3"))
        and pd.notna(row.get("vixy_ma_10"))
        and (row.get("vixy_ma_3") > row.get("vixy_ma_10"))
    )
    uvix_cond4 = (pd.notna(spy_ret) and float(spy_ret) < -0.008)

    uvix_score = sum([bool(uvix_cond1), bool(uvix_cond2), bool(uvix_cond3), bool(uvix_cond4)])

    # ✅ UVIX "solo pánico real"
    spy_panic = (pd.notna(spy_ret) and float(spy_ret) <= float(cfg.uvix_spy_panic_thresh))
    vix_extremo = True
    if cfg.uvix_require_vix_extreme:
        vix_extremo = (pd.notna(vix) and pd.notna(p85) and float(vix) > float(p85))

    if (uvix_score >= int(cfg.uvix_min_score)) and vix_extremo and spy_panic:
        return {
            "estado": "UVIX",
            "accion": "OPEN/HOLD UVIX",
            "comentario": f"PANIC UVIX: score={uvix_score}, vix>p85 y spy_ret<={cfg.uvix_spy_panic_thresh*100:.1f}%.",
        }

    # PREP_SVIX
    cond_prep = (vix > p85) and (ratio_up is False) and contango_ok
    if cond_prep:
        return {"estado": "PREP_SVIX", "accion": "WAIT / PREPARE SVIX", "comentario": "Pánico se agota + contango vuelve."}

    return {"estado": "NEUTRAL", "accion": "NO NEW POSITION", "comentario": "Régimen mixto / transición."}


def compute_states(df_feat: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    """
    Devuelve RAW:
      - raw_estado / raw_accion / raw_comentario
    y luego aplica filtros A+B+C para producir:
      - estado / accion / comentario (operable)
    """
    w = df_feat.copy()

    macro = fetch_macro_events()
    w["macro_tomorrow"] = w["date"].apply(
        lambda d: macro_tomorrow_flag(pd.to_datetime(d), macro) if pd.notna(d) else False
    )

    raw_estados, raw_acciones, raw_comentarios = [], [], []
    for _, r in w.iterrows():
        res = decide_state_row(r, cfg=cfg)
        raw_estados.append(res["estado"])
        raw_acciones.append(res["accion"])
        raw_comentarios.append(res["comentario"])

    w["raw_estado"] = raw_estados
    w["raw_accion"] = raw_acciones
    w["raw_comentario"] = raw_comentarios

    # Aplicamos A+B+C
    w = apply_signal_filters(w, cfg=cfg)

    return w


# -----------------------------
# A + B + C: filtros anti-ruido
# -----------------------------

def _desired_state_from_raw(raw_estado: str) -> str:
    # Mantenemos como "régimen operable" SVIX / UVIX / NEUTRAL
    if raw_estado in ("SVIX", "UVIX"):
        return raw_estado
    return "NEUTRAL"


def apply_signal_filters(df: pd.DataFrame, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    """
    Implementa:
      A) Confirmación (confirm_days)
      B) Cooldown (cooldown_days)
      C) Anti-flip SVIX<->UVIX (anti_flip)
    Output:
      - estado / accion / comentario (final)
    """
    w = df.copy()
    if w.empty:
        w["estado"] = pd.NA
        w["accion"] = pd.NA
        w["comentario"] = pd.NA
        return w

    # ordenar por fecha por seguridad
    if "date" in w.columns:
        w = w.sort_values("date").reset_index(drop=True)

    confirm_n = max(1, int(cfg.confirm_days))
    cooldown_n = max(0, int(cfg.cooldown_days))

    final_estado: List[str] = []
    final_accion: List[str] = []
    final_com: List[str] = []

    current_state = "NEUTRAL"
    last_change_idx = -10**9  # índice del último cambio real

    # tracking para confirmación
    run_state = None
    run_len = 0

    for i, row in w.iterrows():
        raw_state = str(row.get("raw_estado", "NEUTRAL") or "NEUTRAL")
        desired = _desired_state_from_raw(raw_state)

        # --- Confirmación (A) ---
        if desired == run_state:
            run_len += 1
        else:
            run_state = desired
            run_len = 1

        confirmed_desired = desired if run_len >= confirm_n else current_state

        # --- Anti-flip (C) ---
        if cfg.anti_flip:
            if (current_state in ("SVIX", "UVIX")) and (confirmed_desired in ("SVIX", "UVIX")) and (confirmed_desired != current_state):
                # obligamos a pasar por NEUTRAL
                confirmed_desired = "NEUTRAL"

        # --- Cooldown (B) ---
        if cooldown_n > 0:
            in_cooldown = (i - last_change_idx) < cooldown_n
            # en cooldown, no permitimos cambios de estado
            if in_cooldown and confirmed_desired != current_state:
                confirmed_desired = current_state

        # aplicamos cambio si procede
        changed = (confirmed_desired != current_state)
        if changed:
            last_change_idx = i
            current_state = confirmed_desired

        # construimos acción/comentario finales
        if current_state == "SVIX":
            accion = "OPEN/HOLD SVIX"
        elif current_state == "UVIX":
            accion = "OPEN/HOLD UVIX"
        else:
            accion = "NO NEW POSITION"

        # comentario final: explicamos filtros si han intervenido
        raw_acc = str(row.get("raw_accion", "") or "")
        raw_coment = str(row.get("raw_comentario", "") or "")

        note_parts = []
        if confirm_n > 1:
            note_parts.append(f"confirm={confirm_n}")
        if cooldown_n > 0:
            note_parts.append(f"cooldown={cooldown_n}")
        if cfg.anti_flip:
            note_parts.append("anti_flip=on")

        filt_note = (" | ".join(note_parts)).strip()

        if current_state == _desired_state_from_raw(raw_state):
            # señal raw y final coinciden
            comentario = raw_coment
        else:
            # filtros han modulado la señal
            comentario = f"[FILTRADA] raw={raw_state} ({raw_acc}). {raw_coment} | {filt_note}"

        final_estado.append(current_state)
        final_accion.append(accion)
        final_com.append(comentario)

    w["estado"] = final_estado
    w["accion"] = final_accion
    w["comentario"] = final_com
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

        # FINAL
        "estado", "accion", "comentario",

        # RAW (si existen en tu schema; si no, el upsert hará fallback y las quitará)
        "raw_estado", "raw_accion", "raw_comentario",

        # OHLC operables
        "svix_open", "svix_high", "svix_low", "svix_close",
        "uvix_open", "uvix_high", "uvix_low", "uvix_close",
    ]

    w = w[[c for c in keep_cols if c in w.columns]].copy()

    records: List[Dict[str, Any]] = w.to_dict(orient="records")
    records = _json_sanitize_records(records)

    # ✅ upsert con fallback si hay columnas que tu schema cache no conoce
    _upsert_with_schema_fallback(table="vix_daily", records=records, on_conflict="fecha")

    return len(records)


def fetch_vix_daily() -> pd.DataFrame:
    # ✅ IMPORTANTE: paginamos para no quedarnos en 1000 filas
    data = _supabase_select_all(table="vix_daily", order_col="fecha", asc=True, batch_size=1000)
    df = pd.DataFrame(data)
    if not df.empty and "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


# -----------------------------
# Órdenes VIX (Supabase)
# -----------------------------

def fetch_vix_orders(limit: int = 300) -> pd.DataFrame:
    # Si pides pocas (limit), usamos el endpoint normal.
    if limit <= 1000:
        resp = supabase.table("vix_orders").select("*").order("fecha", desc=True).limit(limit).execute()
        if getattr(resp, "error", None):
            raise RuntimeError(resp.error)
        data = getattr(resp, "data", None) or []
        df = pd.DataFrame(data)
        if not df.empty and "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        return df

    # Si algún día quieres más de 1000, también paginamos:
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

def run_vix_pipeline(start: str, end: str, cfg: VixConfig = DEFAULT_CFG) -> pd.DataFrame:
    raw = download_yahoo_daily(start=start, end=end)
    feat = compute_features(raw, cfg=cfg)
    out = compute_states(feat, cfg=cfg)

    # merge OHLC operables (SVIX/UVIX)
    ohlc = download_trade_ohlc(start=start, end=end)
    if ohlc is not None and not ohlc.empty:
        out = out.merge(ohlc, on="date", how="left")

    upsert_vix_daily(out)
    return out
