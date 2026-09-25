import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================================================
# 1. LOAD
# =========================================================
df = pd.read_csv("fpl_features.csv")

FEATURES = [
    'points_last5',
    'minutes_last3',
    'bps_last5', 'ict_last5',
    'xg_last5', 'xa_last5',
    'transfers_last3',
    'season_points_avg', 'season_minutes_avg',
    'was_home', 'value', 'round',
    'pos_DEF', 'pos_FWD', 'pos_GK', 'pos_MID',
]
TARGET = 'total_points'

def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))

# =========================================================
# 2. TIME-BASED CROSS-VALIDATION
# =========================================================
# Rule: only train on PAST seasons, test on FUTURE ones.
folds = [
    (['2022-23'],            '2023-24'),
    (['2022-23', '2023-24'], '2024-25'),
]

print("=" * 60)
print("LINEAR REGRESSION vs RIDGE  (time-based CV)")
print("=" * 60)
print("Baseline to beat: MAE 2.005  (season-avg predictor on played rows)\n")

for train_seasons, test_season in folds:
    train = df[df['season'].isin(train_seasons)]
    test  = df[df['season'] == test_season]

    # Scale — fit on TRAIN ONLY, then transform both (no leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES])
    X_test  = scaler.transform(test[FEATURES])
    y_train, y_test = train[TARGET], test[TARGET]

    print(f"Train: {'+'.join(train_seasons):<20} ({len(train):>5,} rows)")
    print(f"Test:  {test_season:<20} ({len(test):>5,} rows)")
    for name, model in [('Linear', LinearRegression()),
                        ('Ridge  ', Ridge(alpha=1.0))]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"  {name}  MAE={mean_absolute_error(y_test, pred):.3f}  "
              f"RMSE={rmse(y_test, pred):.3f}")
    print()

# =========================================================
# 3. INSPECT THE FINAL MODEL — coefficients + worst mistakes
# =========================================================
print("=" * 60)
print("MODEL INSPECTION (Ridge, trained 2022-24 → 2024-25)")
print("=" * 60)

train = df[df['season'].isin(['2022-23', '2023-24'])]
test  = df[df['season'] == '2024-25']
scaler = StandardScaler()
X_train = scaler.fit_transform(train[FEATURES])
X_test  = scaler.transform(test[FEATURES])
y_train, y_test = train[TARGET], test[TARGET]

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Coefficients — after scaling, magnitudes are directly comparable
coefs = pd.DataFrame({'feature': FEATURES, 'weight': model.coef_}) \
          .assign(abs_w=lambda d: d['weight'].abs()) \
          .sort_values('abs_w', ascending=False) \
          .drop(columns='abs_w')
print("\nTop features (larger |weight| = more influence):")
print(coefs.head(10).to_string(index=False))

# Worst 10 predictions
test_out = test.copy()
test_out['pred']  = pred.round(2)
test_out['error'] = (test_out[TARGET] - test_out['pred']).abs().round(2)
worst = test_out.nlargest(10, 'error')[
    ['name', 'position', 'round', 'points_last3', 'xg_last5',
     TARGET, 'pred', 'error']
]
print("\n10 WORST predictions:")
print(worst.to_string(index=False))