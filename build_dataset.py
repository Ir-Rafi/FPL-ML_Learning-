import requests
import pandas as pd
import time

boot = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
players = pd.DataFrame(boot['elements'])
teams   = pd.DataFrame(boot['teams'])

pos_map  = {1:'GK', 2:'DEF', 3:'MID', 4:'FWD'}
players['position'] = players['element_type'].map(pos_map)

# Build fast lookup dicts ONCE, before the loop
name_lookup = dict(zip(players['id'], players['web_name']))
pos_lookup  = dict(zip(players['id'], players['position']))
team_lookup = dict(zip(players['id'], players['team']))
team_name   = dict(zip(teams['id'],   teams['short_name']))

all_rows = []
start = time.time()
for i, pid in enumerate(players['id'], 1):
    try:
        r = requests.get(
            f"https://fantasy.premierleague.com/api/element-summary/{pid}/",
            timeout=10
        )
        hist = r.json()['history']
        for gw in hist:
            gw['player_id'] = pid
            gw['web_name']  = name_lookup[pid]
            gw['position']  = pos_lookup[pid]
            gw['team']      = team_lookup[pid]
            all_rows.append(gw)
    except Exception as e:
        print(f"  ! player {pid} failed: {e}")
    time.sleep(0.1)
    if i % 20 == 0:
        elapsed = time.time() - start
        print(f"  {i}/{len(players)} done  ({elapsed:.0f}s elapsed, {len(all_rows)} rows so far)")

df = pd.DataFrame(all_rows)
df['team_name']     = df['team'].map(team_name)
df['opponent_name'] = df['opponent_team'].map(team_name)
df.to_csv("fpl_gameweek_data.csv", index=False)
print(f"\nSaved {len(df)} rows to fpl_gameweek_data.csv")