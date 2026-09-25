import pandas as pd

hist = pd.read_csv("historical/2024-25.csv")
curr = pd.read_csv("fpl_gameweek_data.csv")

print("HISTORICAL columns:", sorted(hist.columns.tolist()))
print()
print("CURRENT columns:   ", sorted(curr.columns.tolist()))
print()
print("Common columns:", sorted(set(hist.columns) & set(curr.columns)))
print()
print("Only in historical:", sorted(set(hist.columns) - set(curr.columns)))
print("Only in current:   ", sorted(set(curr.columns) - set(hist.columns)))