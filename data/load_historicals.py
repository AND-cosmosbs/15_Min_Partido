import pandas as pd

def load_historicals(paths_and_seasons):
    dfs = []

    for path, season in paths_and_seasons:
        xl = pd.ExcelFile(path, engine="openpyxl")
        for sh in xl.sheet_names:
            if sh.upper() == "VARIABLES":
                continue
            df = xl.parse(sh)
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df["Season"] = season
            if "Div" not in df.columns:
                df["Div"] = sh
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
