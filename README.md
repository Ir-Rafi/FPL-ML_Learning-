# FPL Player Points Prediction

Hi everyone! I'm a CS undergrad learning Machine Learning from the very beginning. To actually get good at it, I decided to build a real project using my course's knowledge  — so I picked Fantasy Premier League (FPL) because I already follow the game and the data is publicly available.

The goal is simple: **predict how many points a player will score in the next gameweek.**

## What I Did

### 1. Collected the Data

I pulled data from two sources:
- **FPL API** (`fantasy.premierleague.com/api/`) — current season gameweek-by-gameweek stats for all 667 players
- **vaastav/Fantasy-Premier-League GitHub repo** — historical data from 2022-23, 2023-24, and 2024-25 seasons

After cleaning and merging everything, I ended up with **87,051 rows** across 4 seasons and **2,036 unique players**.

### 2. Built Baselines First

Before touching any model, I computed simple baselines to know what "good" even means:

| Baseline | MAE (played rows) |
|----------|-------------------|
| Season average | 2.005 |
| Last 3 gameweeks avg | 2.210 |
| Last 5 gameweeks avg | 2.131 |

So any model I build has to beat **2.005 MAE** to be worth anything.

### 3. Feature Engineering

I created 16 features from the raw data — rolling averages (last 3 and 5 games), expected goals/assists, transfer activity, season cumulative stats, position encoding, home/away, and player value.

Every rolling feature uses `.shift(1)` so the model only sees past data, never the current gameweek. This prevents data leakage — something I learned the hard way is really important.

**Features used:**
```
points_last5, minutes_last3, bps_last5, ict_last5,
xg_last5, xa_last5, transfers_last3, season_points_avg,
season_minutes_avg, was_home, value, round,
pos_DEF, pos_FWD, pos_GK, pos_MID
```

### 4. Tried Linear Regression (It Basically Tied with Baseline)

My first real model was Linear Regression with StandardScaler. I used time-based cross-validation (train on older seasons, test on newer ones) to keep things honest.

Results:
- **Linear Regression MAE: ~1.99**
- That's barely better than the season-average baseline (2.005)

I also ran into **multicollinearity** — some features were basically measuring the same thing, which made the coefficients go haywire (negative weights on things that should be positive). I removed 3 redundant features (`points_last3`, `minutes_last5`, `xgi_last5`) and the coefficients started making sense.

But the prediction accuracy? Didn't really change. That's when I learned: **multicollinearity messes up interpretation, not prediction.**

The math behind it is the Normal Equation: **w = (X^T X)^(-1) X^T y** — it finds the weights that minimize squared error in one shot using linear algebra. Ridge regression adds a penalty term (L2 regularization) to handle correlated features, but it gave identical results here.

### 5. Tried LightGBM (This Actually Worked)

LightGBM is a gradient boosted tree model — instead of fitting one line, it builds hundreds of small decision trees where each tree tries to fix the mistakes of the previous ones.

```python
LGBMRegressor(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=30,
    objective='regression_l1',  # optimizes MAE directly
)
```

Results:
- **LightGBM MAE: 1.684** on the 2024-25 test set
- That's a **16% improvement** over the baseline (2.005)
- And **15% better** than linear regression (1.994)

Top 3 most important features: `transfers_last3`, `season_minutes_avg`, `minutes_last3`

### 6. What the Model Can't Do

The worst predictions are all big "haul" games — players scoring 15-23 points in a single gameweek. These are inherently unpredictable (a hat-trick, a last-minute goal). No model can reliably predict those, and that's fine. The model is good at predicting the typical 1-5 point range that most players score.

## Project Structure

```
FPL/
├── fpl_data.py           # Initial API exploration
├── build_dataset.py      # Pull gameweek data for all players
├── download_history.py   # Download 3 seasons of historical data
├── column_inspection.py  # Compare columns across datasets
├── merge_master.py       # Merge all seasons into one table
├── baselines.py          # Compute baseline predictions
├── build_features.py     # Feature engineering pipeline
├── train_linear.py       # Linear + Ridge regression
├── train_lgbm.py         # LightGBM model
├── fpl_master.csv        # Merged dataset (87K rows)
├── fpl_features.csv      # Final features (34K rows)
└── historical/           # Raw historical CSVs
```

## Results Summary

| Model | MAE | vs Baseline |
|-------|-----|-------------|
| Season-avg baseline | 2.005 | — |
| Linear Regression | 1.994 | -0.5% |
| Ridge Regression | 1.994 | -0.5% |
| **LightGBM** | **1.684** | **-16%** |

## What I Learned

- Always build a baseline before any model 
- Linear regression is a solid starting point but it assumes a linear relationship between features and target
- Feature engineering matters more than model complexity
- Data leakage is silent and deadly — `.shift(1)` on every rolling feature
- Multicollinearity affects interpretation (weird coefficients) but not prediction accuracy
- Time-based cross-validation is essential for time series data — random splits leak future info
- Tree models (LightGBM) handle non-linear patterns and feature interactions that linear models miss
- The biggest unpredictable factor is player "hauls" — those are noise, not signal

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn (LinearRegression, Ridge, StandardScaler, cross-validation)
- LightGBM
- FPL API + vaastav/Fantasy-Premier-League historical data
