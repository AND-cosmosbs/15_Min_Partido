import pandas as pd
from pathlib import Path

# -----------------------------
# 1. Carga de históricos
# -----------------------------

def load_historical_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Lee todos los ficheros all-euro-data-*.xlsm de la carpeta /data
    y los unifica en un solo DataFrame.
    """
    data_path = Path(data_dir)
    files = sorted([p for p in data_path.glob("all-euro-data-*.xlsm")])
    if not files:
        raise FileNotFoundError(
            "No se han encontrado ficheros all-euro-data-*.xlsm en la carpeta /data"
        )

    dfs = []
    for f in files:
        # temporada a partir del nombre de fichero, por si quieres usarla luego
        season = f.stem.replace("all-euro-data-", "")
        xl = pd.ExcelFile(f, engine="openpyxl")
        for sheet in xl.sheet_names:
            if sheet.upper() == "VARIABLES":
                continue
            df = xl.parse(sheet)
            # Parseo de fecha formato europeo
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df["Season"] = season
            if "Div" not in df.columns:
                df["Div"] = sheet
            dfs.append(df)

    unified = pd.concat(dfs, ignore_index=True)

    # Aseguramos numéricos mínimos para el modelo
    for c in ["HTHG", "HTAG"]:
        if c in unified.columns:
            unified[c] = pd.to_numeric(unified[c], errors="coerce")

    return unified


# -----------------------------
# 2. Cálculo de stats de ligas y equipos
# -----------------------------

def compute_team_and_league_stats(unified: pd.DataFrame):
    """
    A partir del histórico unificado calcula:
    - team_stats: p(0-0 HT) y T_score por equipo
    - div_stats: p(0-0 HT), goles HT y L_score por liga
    """
    # ---- Equipos ----
    home = unified.copy()
    home["Team"] = home["HomeTeam"]

    away = unified.copy()
    away["Team"] = away["AwayTeam"]

    long = pd.concat([home, away], ignore_index=True)
    long["is_00_HT"] = ((long["HTHG"] == 0) & (long["HTAG"] == 0)).astype(int)

    team_stats = long.groupby("Team")["is_00_HT"].agg(["mean", "count"]).reset_index()
    team_stats.columns = ["Team", "p_00_HT", "matches"]

    def team_score_row(r):
        p = r["p_00_HT"]
        n = r["matches"]
        pts = 0
        # prob 0-0 HT
        if p >= 0.40:
            pts += 3
        elif p >= 0.33:
            pts += 2
        elif p >= 0.27:
            pts += 1
        # tamaño muestra
        if n >= 40:
            pts += 1
        return pts

    team_stats["T_score"] = team_stats.apply(team_score_row, axis=1)

    # ---- Ligas ----
    div_stats = unified.groupby("Div").apply(
        lambda g: pd.Series(
            {
                "avg_g_ht": g["HTHG"].mean() + g["HTAG"].mean(),
                "p_00_HT": ((g["HTHG"] == 0) & (g["HTAG"] == 0)).mean(),
            }
        )
    ).reset_index()

    def league_score_row(r):
        p = r["p_00_HT"]
        g = r["avg_g_ht"]
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


# -----------------------------
# 3. Scoring de fixtures
# -----------------------------

def classify_match(score: float) -> str:
    if score >= 9:
        return "Ideal"
    elif score >= 7:
        return "Buena"
    elif score >= 5:
        return "Borderline"
    return "Descartar"


def score_fixtures(
    fixtures: pd.DataFrame, team_stats: pd.DataFrame, div_stats: pd.DataFrame
) -> pd.DataFrame:
    """
    Recibe un DataFrame de fixtures con columnas:
    Div, Date, Time, HomeTeam, AwayTeam, B365H, B365D, B365A
    y devuelve el mismo con MatchScore, MatchClass y PickType
    (Ideal / Buena filtrada / None).
    """

    fx = fixtures.copy()

    # Asegurar tipos numéricos en cuotas
    for c in ["B365H", "B365D", "B365A"]:
        if c in fx.columns:
            fx[c] = pd.to_numeric(fx[c], errors="coerce")

    # Merge con liga
    fx = fx.merge(
        div_stats[["Div", "L_score", "LeagueTier"]], on="Div", how="left"
    )

    # Merge con equipos
    fx = fx.merge(
        team_stats[["Team", "T_score"]].rename(
            columns={"Team": "HomeTeam", "T_score": "H_T_score"}
        ),
        on="HomeTeam",
        how="left",
    )
    fx = fx.merge(
        team_stats[["Team", "T_score"]].rename(
            columns={"Team": "AwayTeam", "T_score": "A_T_score"}
        ),
        on="AwayTeam",
        how="left",
    )

    fx["MatchScore"] = (
        fx["L_score"].fillna(0)
        + fx["H_T_score"].fillna(0)
        + fx["A_T_score"].fillna(0)
    )

    fx["MatchClass"] = fx["MatchScore"].apply(classify_match)

    # Filtro de favorito claro: min(B365H, B365A) >= 2.0
    fx["FavOdd"] = fx[["B365H", "B365A"]].min(axis=1)
    fx["NoClearFav"] = fx["FavOdd"] >= 2.0

    # Buena filtrada: Buena + score>=8 + liga Oro/Plata + sin favorito claro
    fx["IsBuenaFiltrada"] = (
        (fx["MatchClass"] == "Buena")
        & (fx["MatchScore"] >= 8)
        & (fx["LeagueTier"].isin(["Oro", "Plata"]))
        & (fx["NoClearFav"])
    )

    fx["PickType"] = None
    fx.loc[
        (fx["MatchClass"] == "Ideal") & (fx["NoClearFav"]), "PickType"
    ] = "Ideal"
    fx.loc[
        fx["IsBuenaFiltrada"] & fx["PickType"].isna(), "PickType"
    ] = "Buena filtrada"

    return fx
