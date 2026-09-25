
import requests
import pandas as pd
import os

# The historical seasons — start with 3 recent ones
SEASONS = ["2022-23", "2023-24", "2024-25"]
BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

os.makedirs("historical", exist_ok=True)

for season in SEASONS:
    url = f"{BASE}/{season}/gws/merged_gw.csv"
    out = f"historical/{season}.csv"
    print(f"Downloading {season} ...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(out, "wb") as f:
        f.write(r.content)
    df = pd.read_csv(out)
    print(f"  saved {out} — {df.shape[0]} rows, {df.shape[1]} cols")

print("\nAll historical data downloaded.")