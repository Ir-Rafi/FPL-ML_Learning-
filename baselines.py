import pandas as pd

df = pd.read_csv("fpl_master.csv")
df = df.sort_values(['season', 'name', 'round']).reset_index(drop=True)

# For each (season, player) build predictions using ONLY prior rows in the same season
def add_predictions(g):
    g = g.sort_values('round')
    g['pred_zero']       = 0
    g['pred_season_avg'] = g['total_points'].expanding().mean().shift(1)
    g['pred_last3_avg']  = g['total_points'].rolling(3, min_periods=1).mean().shift(1)
    g['pred_last5_avg']  = g['total_points'].rolling(5, min_periods=1).mean().shift(1)
    return g

df = df.groupby(['season', 'name'], group_keys=False).apply(add_predictions)

# Drop rows where we couldn't form ANY prediction (a player's very first game)
eval_df = df.dropna(subset=['pred_last3_avg']).copy()

def mae(y, yhat):  return (y - yhat).abs().mean()
def rmse(y, yhat): return ((y - yhat) ** 2).mean() ** 0.5

print(f"Evaluation rows: {len(eval_df):,}\n")
print(f"{'Predictor':<38}{'MAE':>8}{'RMSE':>10}")
print("-" * 56)
for col, label in [
    ('pred_zero',       'Predict 0 for everyone'),
    ('pred_season_avg', 'Season-so-far average'),
    ('pred_last5_avg',  'Last-5-gameweek average'),
    ('pred_last3_avg',  'Last-3-gameweek average'),
]:
    m = mae(eval_df['total_points'], eval_df[col])
    r = rmse(eval_df['total_points'], eval_df[col])
    print(f"{label:<38}{m:>8.3f}{r:>10.3f}")

# Also — the "beat this or don't bother" number, evaluated only on
# players who actually played (to strip out the trivial 0-min rows)
played = eval_df[eval_df['minutes'] > 0]
print(f"\nOn PLAYED rows only ({len(played):,} rows):")
for col, label in [
    ('pred_last3_avg',  'Last-3-gameweek average'),
    ('pred_season_avg', 'Season-so-far average'),
]:
    print(f"  {label:<36} MAE={mae(played['total_points'], played[col]):.3f}")