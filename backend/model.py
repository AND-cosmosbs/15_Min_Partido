import re
from pathlib import Path
from typing import Tuple

import pandas as pd


# -----------------------------------------------------
# 1. Carga de histórico
# -----------------------------------------------------


def load_historical_data(data_dir: str | Path) -> pd.DataFrame:
    """
    Lee todos los ficheros all-euro-data-*.xls* de la carpeta data_dir
    y devuelve un único DataFrame unificado.

    Espera ficheros tipo:
      - all-euro-data-2023-2024.xlsm
      - all-euro-data-2024-2025.xlsm
      - all-euro-data-2025-2026.xlsm
    con una hoja por liga y una hoja 'VARIABLES' que se ignora.
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob("all-euro-data-*.xls*"))

    if not files:
        raise FileNotFoundError(
            f"No se han encontrado ficheros 'all-euro-data-*.xls*' en {data_path.resolve()}"
        )

    dfs: list[pd.DataFrame] = []

    for f in files:
        # Extraer temporada del nombre de archivo, ej. 2023-2024
        m = re.search(r"(\d{4}-\d{4})", f.name)
        season = m.group(1) if m else "unknown"

        xl = pd.ExcelFile(f, engine="openpyxl")
        for sh in xl.sheet_names:
            if sh.upper() == "VARIABLES":
                continue

            df = xl.parse(sh)

            # Fecha en formato europeo dd/mm/yy
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(
                    df["Date"], dayfirst=True, errors="coerce"
                )

            df["Season"] = season

            # Si no viene columna Div, usamos el nombre de la hoja
            if "Div" not in df.columns:
                df["Div"] = sh

            dfs.append(df)

    hist = pd.concat(dfs, ignore_index=True)

    # Forzamos a numérico lo mínimo necesario
    for c in ["HTHG", "HTAG"]:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")

    return hist


# -----------------------------------------------------
# 2. Cálculo de stats por equipo y liga
# -----------------------------------------------------


def _compute_team_stats(hist: pd.DataFrame) -> pd.DataFrame:
    home = hist.copy()
    home["Team"] = home["HomeTeam"]

    away = hist.copy()
    away["Team"] = away["AwayTeam"]

    long = pd.concat([home, away], ignore_index=True)

    long["is_00_HT"] = (
        (long["HTHG"] == 0) & (long["HTAG"] == 0)
    ).astype(int)

    team_stats = (
        long.groupby("Team")["is_00_HT"]
        .agg(["mean", "count"])
        .reset_index()
    )
    team_stats.columns = ["Team", "p_00_HT", "matches"]

    def team_score_row(row) -> int:
        p = row["p_00_HT"]
        n = row["matches"]
        pts = 0

        # Probabilidad de 0-0 al descanso
        if p >= 0.40:
            pts += 3
        elif p >= 0.33:
            pts += 2
        elif p >= 0.27:
            pts += 1

        # Volumen de partidos
        if n >= 40:
            pts += 1

        return pts

    team_stats["T_score"] = team_stats.apply(
        team_score_row, axis=1
    )

    return team_stats


def _compute_div_stats(hist: pd.DataFrame) -> pd.DataFrame:
    div_stats = (
        hist.groupby("Div")
        .apply(
            lambda g: pd.Series(
                {
                    "avg_g_ht": g["HTHG"].mean()
                    + g["HTAG"].mean(),
                    "p_00_HT": (
                        (g["HTHG"] == 0)
                        & (g["HTAG"] == 0)
                    ).mean(),
                }
            )
        )
        .reset_index()
    )

    def league_score_row(row) -> int:
        p = row["p_00_HT"]
        g = row["avg_g_ht"]
        pts = 0

        # Probabilidad de 0-0 al descanso
        if p >= 0.32:
            pts += 2
        elif p >= 0.28:
            pts += 1

        # Goles medios en 1ª parte
        if g <= 1.05:
            pts += 2
        elif g <= 1.25:
            pts += 1

        return pts

    div_stats["L_score"] = div_stats.apply(
        league_score_row, axis=1
    )

    def league_tier(score: int) -> str:
        if score >= 3:
            return "Oro"
        elif score >= 2:
            return "Plata"
        else:
            return "Bronce"

    div_stats["LeagueTier"] = div_stats["L_score"].apply(
        league_tier
    )

    return div_stats


def compute_team_and_league_stats(
    hist: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - team_stats  (Team, p_00_HT, matches, T_score)
      - div_stats   (Div, avg_g_ht, p_00_HT, L_score, LeagueTier)
    """
    team_stats = _compute_team_stats(hist)
    div_stats = _compute_div_stats(hist)
    return team_stats, div_stats


# -----------------------------------------------------
# 3. Scoring de fixtures
# -----------------------------------------------------


def score_fixtures(
    hist: pd.DataFrame,
    team_stats: pd.DataFrame,
    div_stats: pd.DataFrame,
    fixtures: pd.DataFrame,
) -> pd.DataFrame:
    """
    Asigna score a cada partido de 'fixtures' y devuelve un DataFrame
    con columnas extra:
      - L_score, LeagueTier
      - H_T_score, A_T_score
      - MatchScore, MatchClass
      - FavOdd, NoClearFav
      - IsBuenaFiltrada, PickType

    El parámetro 'hist' se incluye solo para ser compatible con main.py,
    pero NO se usa dentro de la función.
    """

    # Aseguramos que las columnas clave existen
    needed_cols = [
        "Div",
        "Date",
        "Time",
        "HomeTeam",
        "AwayTeam",
        "B365H",
        "B365D",
        "B365A",
    ]
    missing = [c for c in needed_cols if c not in fixtures.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas en fixtures: {missing}"
        )

    df = fixtures.copy()

    # Parseo de fecha por si acaso
    df["Date"] = pd.to_datetime(
        df["Date"], dayfirst=True, errors="coerce"
    )

    # Merge de stats de liga
    df = df.merge(
        div_stats[["Div", "L_score", "LeagueTier"]],
        on="Div",
        how="left",
    )

    # Merge de stats de equipo local
    df = df.merge(
        team_stats[["Team", "T_score"]].rename(
            columns={"Team": "HomeTeam", "T_score": "H_T_score"}
        ),
        on="HomeTeam",
        how="left",
    )

    # Merge de stats de equipo visitante
    df = df.merge(
        team_stats[["Team", "T_score"]].rename(
            columns={"Team": "AwayTeam", "T_score": "A_T_score"}
        ),
        on="AwayTeam",
        how="left",
    )

    # MatchScore = L_score + H_T_score + A_T_score
    df["MatchScore"] = (
        df["L_score"].fillna(0)
        + df["H_T_score"].fillna(0)
        + df["A_T_score"].fillna(0)
    )

    def classify_match(score: float) -> str:
        if score >= 9:
            return "Ideal"
        elif score >= 7:
            return "Buena"
        elif score >= 5:
            return "Borderline"
        else:
            return "Descartar"

    df["MatchClass"] = df["MatchScore"].apply(
        classify_match
    )

    # Filtro de favorito claro (cuota < 2.0 en local o visitante)
    df["FavOdd"] = df[["B365H", "B365A"]].min(axis=1)
    df["NoClearFav"] = df["FavOdd"] >= 2.0

    # Buena filtrada: Buena + MatchScore>=8 + liga Oro/Plata + sin favorito claro
    df["IsBuenaFiltrada"] = (
        (df["MatchClass"] == "Buena")
        & (df["MatchScore"] >= 8)
        & (df["LeagueTier"].isin(["Oro", "Plata"]))
        & (df["NoClearFav"])
    )

    # PickType: lo que la app deberá mostrar como seleccionable
    df["PickType"] = None
    # Ideal sin favorito claro
    df.loc[
        (df["MatchClass"] == "Ideal")
        & (df["NoClearFav"]),
        "PickType",
    ] = "Ideal"
    # Buena filtrada solo si aún no hay PickType
    df.loc[
        df["IsBuenaFiltrada"]
        & df["PickType"].isna(),
        "PickType",
    ] = "Buena filtrada"

    return df
