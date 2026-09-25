import pandas as pd
import glob

# Columns to keep — features + target only. Drop the mng_*, defensive, and label columns.
KEEP = [
    'season', 'name', 'position', 'team', 'round', 'kickoff_time',
    'opponent_team', 'was_home', 'minutes',
    'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
    'saves', 'bonus', 'bps', 'yellow_cards', 'red_cards',
    'influence', 'creativity', 'threat', 'ict_index',
    'expected_goals', 'expected_assists', 'expected_goal_involvements',
    'expected_goals_conceded','starts',
    'value', 'selected', 'transfers_in', 'transfers_out', 'transfers_balance',
    'team_h_score', 'team_a_score',
    'total_points',   # <-- the TARGET
]

# 1. Load historical seasons
hist_dfs = []
for path in sorted(glob.glob("historical/*.csv")):
    season = path.split("/")[-1].replace(".csv", "")
    df = pd.read_csv(path)
    df['season'] = season
    hist_dfs.append(df)
    print(f"Loaded {season}: {df.shape}")
historical = pd.concat(hist_dfs, ignore_index=True)

# 2. Load current season and align its column names
current = pd.read_csv("fpl_gameweek_data.csv")
current = current.rename(columns={'web_name': 'name'})   # match historical
current['season'] = "2025-26"

# 3. Keep only the columns we agreed on (silently drop anything missing)
historical = historical[[c for c in KEEP if c in historical.columns]]
current    = current   [[c for c in KEEP if c in current.columns]]

# 4. Stack into one master table
master = pd.concat([historical, current], ignore_index=True)
master = master.sort_values(['season', 'name', 'round']).reset_index(drop=True)

# 5. Save
master.to_csv("fpl_master.csv", index=False)

print("\n== MASTER DATASET ==")
print(f"Shape: {master.shape}")
print("\nRows per season:")
print(master['season'].value_counts().sort_index())
print(f"\nUnique players across all seasons: {master['name'].nunique()}")
print(f"Rows where the player actually played (minutes > 0): {(master['minutes']>0).sum()}")