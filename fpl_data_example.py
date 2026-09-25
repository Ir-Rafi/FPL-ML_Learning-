import requests
import pandas as pd

# 1. Get the master snapshot — all players, teams, fixtures
boot = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()

players = pd.DataFrame(boot['elements'])
teams   = pd.DataFrame(boot['teams'])
events  = pd.DataFrame(boot['events'])   # gameweeks

print(players.columns.tolist())   # see what's available — ~100 columns
print(players.shape)              # ~600–700 players

# 2. Get gameweek-by-gameweek history for ONE player (say, Salah)
salah_id = players.loc[players['web_name']=='Haaland','id'].iloc[0]
salah = requests.get(f"https://fantasy.premierleague.com/api/element-summary/{salah_id}/").json()

history = pd.DataFrame(salah['history'])
print(history.columns.tolist())
print(history[['round','opponent_team','was_home','minutes','goals_scored',
               'assists','clean_sheets','bonus','total_points']].head(10))