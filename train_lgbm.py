import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("fpl_features.csv")

FEATURES = [
    'points_last5', 'minutes_last3',
    'bps_last5', 'ict_last5',
    'xg_last5', 'xa_last5',
    'transfers_last3',
    'season_points_avg', 'season_minutes_avg',
    'was_home', 'value', 'round',
    'pos_DEF', 'pos_FWD', 'pos_GK', 'pos_MID',
]
TARGET = 'total_points'

def rmse(y, yhat): return np.sqrt(mean_squared_error(y, yhat))

folds = [
    (['2022-23'],            '2023-24'),
    (['2022-23', '2023-24'], '2024-25'),
]

def make_model():
    return lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=30,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        objective='regression_l1',   # optimize MAE directly
        verbose=-1,
    )

print("=" * 60)
print("LIGHTGBM  (time-based CV)")
print("=" * 60)
print("Targets to beat:")
print("  Baseline (season-avg):  MAE 2.005")
print("  Linear regression:      MAE 1.994\n")

for train_seasons, test_season in folds:
    train = df[df['season'].isin(train_seasons)]
    test  = df[df['season'] == test_season]
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test,  y_test  = test[FEATURES],  test[TARGET]

    model = make_model()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"Train: {'+'.join(train_seasons):<20} ({len(train):>5,} rows)")
    print(f"Test:  {test_season:<20} ({len(test):>5,} rows)")
    print(f"  LightGBM  MAE={mean_absolute_error(y_test, pred):.3f}  "
          f"RMSE={rmse(y_test, pred):.3f}\n")

# ================== FINAL MODEL INSPECTION ==================
print("=" * 60)
print("FEATURE IMPORTANCES (final model 22-24 → 24-25)")
print("=" * 60)

train = df[df['season'].isin(['2022-23', '2023-24'])]
test  = df[df['season'] == '2024-25']
X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

model = make_model()
model.fit(X_train, y_train)
pred = model.predict(X_test)

imp = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False)
print("\nFeature importances (how often the model splits on each):")
print(imp.to_string(index=False))

# Worst 10 predictions
test_out = test.copy()
test_out['pred']  = pred.round(2)
test_out['error'] = (test_out[TARGET] - test_out['pred']).abs().round(2)
worst = test_out.nlargest(10, 'error')[
    ['name', 'position', 'round', 'points_last5', 'xg_last5', TARGET, 'pred', 'error']
]
print("\n10 WORST predictions:")
print(worst.to_string(index=False))