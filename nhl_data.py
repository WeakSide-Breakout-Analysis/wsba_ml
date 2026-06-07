import pandas as pd
import wsba_hockey as wsba
from gspread_pandas import Spread
from wsba_hockey.tools.globals import *

### DATASETS ###
## Retrieve necessary datasets from local data inventory ##

DEFAULT_PARAMS = {
    'strength_state': ('not in', ['3v6', '4v6', '5v6', '6v3', '6v4', '6v5']),
    'season_type': ('==', 2)
}

FORMATS = {
    "read":{
        "csv": lambda path: pd.read_csv(path),
        "parquet": lambda path: pd.read_parquet(path)
    },
    "write":{
        "csv": lambda df, path: df.to_csv(path, index=False),
        "parquet": lambda df, path: df.to_parquet(path, index=False)
    }
}

DEFAULT_SEASONS = [s for s in wsba.SEASON_NAMES.keys() if s < 20262027][-5:]
DATA_INVENTORY = 'data'
SENSE_DATA = 'data/datasets'

def skater_dataset(seasons=DEFAULT_SEASONS, data_path=SENSE_DATA, scrape=False, format='parquet'):
    #Qualified skaters
    #A qualified skater is one that plays a over a time-on-ice threshold at a given strength state in a single given season
    #EV - 150 minutes; PP/SH - 50 minutes
    #Split by player, season, team, and position

    print('WSBA HS Skater Dataset...')
    export = f'{data_path}/nhl_skater_dataset.{format}'

    if scrape:
        data = nhl_build_dataset(seasons, True)
        data['season_type'] = data['season_type'].astype(int)
        
        FORMATS['write'][format](data, export)
    else:
        data = FORMATS['read'][format](export).query(f"season in @seasons")

    wsba.utility_get_schema(data).to_csv(f'{data_path}/schema/nhl_skater_schema.csv', index=False)

    return data

def team_dataset(seasons=DEFAULT_SEASONS, data_path=SENSE_DATA, scrape=False, format='parquet'):
    #Teams
    #The teams dataset will assist the weighting and conversion of skill values to impact on team goal differential
    
    export = f'{data_path}/nhl_team_dataset.{format}'

    print('WSBA HS Team Dataset...')
    if scrape:
        data = pd.concat([pd.read_parquet(f'{DATA_INVENTORY}/games/team_games/wsba_nhl_{season}_team_games.parquet') for season in seasons])

        FORMATS['write'][format](data, export)

    else:
        data = FORMATS['read'][format](export).query(f"season in @seasons")
    
    wsba.utility_get_schema(data).to_csv(f'{data_path}/schema/nhl_team_schema.csv', index=False)

    return data

def consolidate_dataset(df, seasons=DEFAULT_SEASONS[-3:]):
    return wsba.nhl_agg_stats(df, ['player_id','position'], params={'time_on_ice': ('>=', 450, 'after'), 'season': ('in', seasons), 'strength_state': ('==', '5v5')})

def combine_datasets(skater, team):
    #Combine skater and team dfs

    df = pd.merge(skater, team[
            [
                'team_abbr',
                'season',
                'regulation_wins',
                'regulation_losses',
                'overtime_wins',
                'overtime_losses',
                'shootout_wins',
                'shootout_losses'
            ]
        ], 
        how='left'
    )
    
    return df

def publish_data(df, group):
    spreadsheet = {
        'skater': Spread('1R6PKZl6UmrsWmWnP2k3IoGGaD26ILvwpm-25_MlM8aQ'),
        'team': Spread('1BFXA_Q8n0zRMKhyIXXxqrx6UfbcknRwRdtTaWIg8u3o'),
        'goalie': None,
    }

    #Tables
    spread = spreadsheet[group] 

    if spread:
        if group == 'skater':
            front_col = wsba.FRONT_COL

            df = df[[col for col in front_col if col in df.columns]+[col for col in df.columns if col not in front_col]]

            df['player_name'] = df['player_name'].str.title()
            df = df.sort_values(by=['player_name','season','team_abbr','player_id'])

        spread.df_to_sheet(df,index=False,sheet=f'{group}sDB')
    else:
        pass

    print('Done.')

def nhl_build_dataset(seasons:int | list, edge:bool = False) -> pd.DataFrame:
    if isinstance(seasons, int):
        seasons = [seasons]

    merge_cols = ['player_id', 'season', 'team_abbr', 'position', 'season_type', 'strength_state']

    # Load datasets
    wsba_stats = pd.concat([
        pd.read_parquet(f)
        for f in [f'data/games/skater_games/wsba_nhl_{season}_skater_games.parquet' for season in seasons]
        if os.path.exists(f)
    ], ignore_index=True)

    a3z_stats = pd.concat([
        pd.read_csv(f)
        for f in [f'data/a3z/games/skater_games/wsba_nhl_{season}_a3z_skater_games.csv' for season in seasons]
        if os.path.exists(f)
    ], ignore_index=True)

    schedule = pd.read_csv(wsba.SCHEDULE_PATH)

    # Keep only desired columns
    wsba_stats = wsba_stats[['game_id']+[col for col in merge_cols + DATASET_COL['wsba'] if col in wsba_stats.columns]]
    a3z_stats = a3z_stats[['game_id']+[col for col in merge_cols + DATASET_COL['a3z'] if col in a3z_stats.columns]]
    
    # Add the game date to select the latest team when aggregating and standing points accumulated (to a single strength only to avoid inflating the value)
    schedule[[
        'game_id','game_date','away_team_abbr','home_team_abbr',
        'period_type_last','away_score', 'home_score'
    ]]
    
    schedule['team_abbr'] = schedule[['home_team_abbr','away_team_abbr']].values.tolist()
    team_schedule = schedule.explode('team_abbr').copy()

    team_schedule['venue'] = np.where(
        team_schedule['team_abbr'] == team_schedule['home_team_abbr'],
        'home',
        'away'
    )

    team_schedule['team_score'] = np.where(
        team_schedule['venue']=='home',
        team_schedule['home_score'],
        team_schedule['away_score']
    )

    team_schedule['opponent_score'] = np.where(
        team_schedule['venue']=='away',
        team_schedule['home_score'],
        team_schedule['away_score']
    )

    team_schedule = team_schedule[['game_id','game_date','period_type_last','team_abbr','venue','team_score', 'opponent_score']]

    team_schedule['win'] = (team_schedule['team_score'] > team_schedule['opponent_score']).astype(int)
    team_schedule['overtime_loss'] = ((team_schedule['team_score'] < team_schedule['opponent_score']) & (team_schedule['period_type_last'] != 'REG')).astype(int)
    team_schedule['standings_points'] = (2*team_schedule['win']) + team_schedule['overtime_loss']

    team_schedule['strength_state'] = '5v5'

    wsba_stats = pd.merge(
        wsba_stats, team_schedule[['game_id','team_abbr','strength_state','game_date','standings_points']],
        how='left'
    )

    # Aggregate WSBA and A3Z independently
    wsba_stats = wsba_stats.sort_values(by=['player_id','season','game_date','game_id'])

    wsba_stats = wsba.nhl_agg_stats(wsba_stats, manual_agg={'game_date': 'last'})
    a3z_stats = wsba.nhl_agg_stats(a3z_stats, ['player_id', 'season', 'team_abbr', 'season_type', 'strength_state'])

    # Merge WSBA + A3Z
    stats = pd.merge(
        wsba_stats,
        a3z_stats.rename(columns={
            'time_on_ice': 'a3z_time_on_ice',
            'time_on_ice_per_games_played': 'a3z_time_on_ice_per_games_played'
        }).drop(columns=[
            'time_on_ice_per_games_played_percentile',
            'position', 
            'games_played',
            'season_type'
        ], errors='ignore'),
        how='left'
    )

    # Standardize formats
    stats['strength_state'] = stats['strength_state'].map(STRENGTH_MAP)
    stats['position'] = stats['position'].str[0].map({'C':'F','L':'F', 'R':'F', 'D':'D'})

    # Re-aggregate to combine similar strength states (mapped above)
    stats = wsba.nhl_agg_stats(stats, manual_agg={'game_date': 'last'})

    return stats.replace([-np.inf, np.inf], 0)

#Create datasets if ran in this script
if __name__ == '__main__':
    skater_dataset(scrape=True)
