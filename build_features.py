import pandas as pd

df = pd.read_csv("fpl_master.csv")
df = df.sort_values(['season', 'name', 'round']).reset_index(drop=True)
df['was_home'] = df['was_home'].astype(int)

print(f"Starting shape: {df.shape}")
print("Building features...")

grp = df.groupby(['season', 'name'], sort=False)

def roll(col, w):
    """Rolling mean of window w, SHIFTED so it uses only prior gameweeks."""
    return grp[col].transform(lambda x: x.rolling(w, min_periods=1).mean().shift(1))

# === RECENT FORM ===
for w in [3, 5]:
    df[f'points_last{w}']  = roll('total_points', w)
    df[f'minutes_last{w}'] = roll('minutes',      w)
df['bps_last5']     = roll('bps', 5)
df['ict_last5']     = roll('ict_index', 5)
df['starts_last3']  = roll('starts', 3)

# === EXPECTED STATS ===
df['xg_last5']  = roll('expected_goals', 5)
df['xa_last5']  = roll('expected_assists', 5)
df['xgi_last5'] = roll('expected_goal_involvements', 5)

# === MOMENTUM ===
df['transfers_last3'] = roll('transfers_balance', 3)

# === SEASON-CUMULATIVE ===
df['season_points_avg']  = grp['total_points'].transform(lambda x: x.expanding().mean().shift(1))
df['season_minutes_avg'] = grp['minutes'].transform(lambda x: x.expanding().mean().shift(1))

# === POSITION (one-hot) ===
df = pd.concat([df, pd.get_dummies(df['position'], prefix='pos').astype(int)], axis=1)

# === FILTER TO TRAINING ROWS ===
# Rule: only rows where the player actually played, AND had at least some history
model_df = df[df['minutes'] > 0].copy()
model_df = model_df.dropna(subset=['points_last3'])

# === FINAL FEATURE LIST ===
FEATURES = [
    'points_last3', 'points_last5',
    'minutes_last3', 'minutes_last5',
    'bps_last5', 'ict_last5', 'starts_last3',
    'xg_last5', 'xa_last5', 'xgi_last5',
    'transfers_last3',
    'season_points_avg', 'season_minutes_avg',
    'was_home', 'value', 'round',
    'pos_DEF', 'pos_FWD', 'pos_GK', 'pos_MID',
]
TARGET = 'total_points'

# Keep only what we need + metadata
keep = ['season', 'name', 'position', 'round', TARGET] + FEATURES
model_df = model_df[keep]
model_df.to_csv("fpl_features.csv", index=False)

print(f"\nFinal shape: {model_df.shape}")
print(f"Features: {len(FEATURES)}")
print(f"\nRows per season:")
print(model_df['season'].value_counts().sort_index())
print(f"\nAny NaNs? {model_df[FEATURES].isna().sum().sum()}")
print(f"\nSample row:")
print(model_df.iloc[100][['name','season','round','total_points','points_last3','xg_last5','was_home','value']])