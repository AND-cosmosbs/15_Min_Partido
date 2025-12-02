# backend/model.py

from pathlib import Path
from typing import Tuple

import pandas as pd


# ---------- 1. Carga de históricos ----------

def load_historical_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Lee todos los ficheros all-euro-data-*.xls/xlsm del directorio data/
    y devuelve un DataFrame unificado con todas las temporadas y ligas.
    """
    data_path = Path(data_dir)
    files = sorted(list(data_path.glob("all-euro-data-*.xls*")))

    if not files:
        raise FileNotFoundError(
            f"No se han encontrado ficheros 'all-euro-data-*.xls*' en {data_path.resolve()}"
        )

    dfs = []
    for f in files:
        xl = pd.ExcelFile(f, engine="openpyxl")
        for sh in xl.sheet_names:
            if sh.upper() == "VARIABLES":
                continue
            df = xl.parse(sh)

            # Fecha en formato europeo
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

            # División
            if "Div" not in df.columns:
                df["Div"] = sh

            # Temporada deducida del nombre de fichero
            season = f.stem.replace("all-euro-data-", "")
            df["Season"] = season

            dfs.append(df)

    unified = pd.concat(dfs, ignore_index=True)

    # Campos numéricos mínimos
    for c in ["HTHG", "HTAG"]:
        if c in unified.columns:
            unified[c] = pd.to_numeric(unified[c], errors="coerce")

    return unified


# ---------- 2. Stats por equipo y por liga ----------

def compute_team_and_league_stats(unified: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A partir del histórico unificado calcula:
      - team_stats: p(0-0 HT) y T_score por equipo.
      - div_stats: p(0-0 HT), goles HT y L_score por liga.
    """

    # --- Equipos ---
    home = unified.copy()
    home["Team"] = home["HomeTeam"]

    away = unified.copy()
    away["Team"] = away["AwayTeam"]

    long = pd.concat([home, away], ignore_index=True)

    long["is_00_HT"] = ((long["HTHG"] == 0) & (long["HTAG"] == 0)).astype(int)

    team_stats = long.groupby("Team")["is_00_HT"].agg(["mean", "count"]).reset_index()
    team_stats.columns = ["Team", "p_00_HT", "matches"]

    def team_score_row(row):
        p = row["p_00_HT"]
        n = row["matches"]
        pts = 0
        if p >= 0.40:
            pts += 3
        elif p >= 0.33:
            pts += 2
        elif p >= 0.27:
            pts += 1
        if n >= 40:
            pts += 1
        return pts

    team_stats["T_score"] = team_stats.apply(team_score_row, axis=1)

    # --- Ligas ---
    div_stats = unified.groupby("Div").apply(
        lambda g: pd.Series(
            {
                "avg_g_ht": g["HTHG"].mean() + g["HTAG"].mean(),
                "p_00_HT": ((g["HTHG"] == 0) & (g["HTAG"] == 0)).mean(),
            }
        )
    ).reset_index()

    def league_score_row(row):
        p = row["p_00_HT"]
        g = row["avg_g_ht"]
        pts = 0
        if p >= 0.32:
            pts += 2
        elif p >= 0.28:
            pts += 1
        if g <= 1.05:
            pts += 2
        elif g <= 1.25:
            pts += 1
        return pts

    div_stats["L_score"] = div_stats.apply(league_score_row, axis=1)

    def league_tier(s):
        if s >= 3:
            return "Oro"
        elif s >= 2:
            return "Plata"
        return "Bronce"

    div_stats["LeagueTier"] = div_stats["L_score"].apply(league_tier)

    return team_stats, div_stats


# ---------- 3. Scoring de un fixture ----------

def score_fixtures(fixtures: pd.DataFrame,
                   team_stats: pd.DataFrame,
                   div_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe un DataFrame de fixtures con al menos:
      Div, Date, Time, HomeTeam, AwayTeam, B365H, B365D, B365A
    y devuelve el mismo DataFrame con:
      L_score, LeagueTier, H_T_score, A_T_score, MatchScore, MatchClass,
      FavOdd, NoClearFav, IsBuenaFiltrada, PickType.
    """

    df = fixtures.copy()

    # Asegurar fecha
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Unir scores de liga
    df = df.merge(
        div_stats[["Div", "L_score", "LeagueTier"]],
        on="Div",
        how="left",
    )

    # Unir scores de equipo local
    df = df.merge(
        team_stats[["Team", "T_score"]].rename(
            columns={"Team": "HomeTeam", "T_score": "H_T_score"}
        ),
        on="HomeTeam",
        how="left",
    )

    # Unir scores de equipo visitante
    df = df.merge(
        team_stats[["Team", "T_score"]].rename(
            columns={"Team": "AwayTeam", "T_score": "A_T_score"}
        ),
        on="AwayTeam",
        how="left",
    )

    # MatchScore
    df["MatchScore"] = (
        df["L_score"].fillna(0)
        + df["H_T_score"].fillna(0)
        + df["A_T_score"].fillna(0)
    )

    # Clasificación por score
    def classify(s):
        if s >= 9:
            return "Ideal"
        elif s >= 7:
            return "Buena"
        elif s >= 5:
            return "Borderline"
        return "Descartar"

    df["MatchClass"] = df["MatchScore"].apply(classify)

    # Filtro de favorito claro (cuota favorita < 2.0)
    df["FavOdd"] = df[["B365H", "B365A"]].min(axis=1)
    df["NoClearFav"] = df["FavOdd"] >= 2.0

    # Buena filtrada: Buena + Score>=8 + liga Oro/Plata + sin favorito claro
    df["IsBuenaFiltrada"] = (
        (df["MatchClass"] == "Buena")
        & (df["MatchScore"] >= 8)
        & (df["LeagueTier"].isin(["Oro", "Plata"]))
        & (df["NoClearFav"])
    )

    # Tipo de pick final
    df["PickType"] = None
    # Ideal sin favorito claro
    df.loc[(df["MatchClass"] == "Ideal") & (df["NoClearFav"]), "PickType"] = "Ideal"
    # Buena filtrada si no había ya Ideal
    mask_bf = df["IsBuenaFiltrada"] & df["PickType"].isna()
    df.loc[mask_bf, "PickType"] = "Buena filtrada"

    return df
