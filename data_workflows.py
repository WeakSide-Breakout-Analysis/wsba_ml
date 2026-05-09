import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = (PROJECT_ROOT / 'data').as_posix()
WSBA_ML_ROOT = (PROJECT_ROOT / 'wsba_ml').as_posix()

import numpy as np
import pandas as pd
import wsba_hockey as wsba
from functools import reduce
from wsba_ml.game_score import calculate_game_score_metrics, wsba_game_score
from wsba_ml.gar import wsba_gar
from wsba_ml.nhl_data import publish_data, skater_dataset as _skater_dataset, team_dataset as _team_dataset
from wsba_ml.qoc_qot import qoc_qot
from wsba_ml.rapm import wsba_rapm
from wsba_hockey.tools.globals import (
    DATASET_COL,
    SEASON_NAMES,
    STRENGTHS,
)
from wsba_ml.microstat import (
    nhl_calculate_microstats,
    nhl_join_plays,
    nhl_scrape_a3z_pbp
)

### DATA PIPELINES ###
def skater_dataset(seasons=None, data_path=None, scrape=False, format='parquet'):
    if seasons is None:
        return _skater_dataset(data_path=data_path or f'{DATA_ROOT}/datasets', scrape=scrape, format=format)

    return _skater_dataset(seasons, data_path or f'{DATA_ROOT}/datasets', scrape, format)


def team_dataset(seasons=None, data_path=None, scrape=False, format='parquet'):
    if seasons is None:
        return _team_dataset(data_path=data_path or f'{DATA_ROOT}/datasets', scrape=scrape, format=format)

    return _team_dataset(seasons, data_path or f'{DATA_ROOT}/datasets', scrape, format)


def _parquet_safe_value(value):
    if isinstance(value, (list, tuple, set, dict)):
        return str(value)
    return value


def _normalize_parquet_columns(df):
    out = df.copy()
    out.columns = out.columns.astype(str)

    for col in out.columns:
        if out[col].dtype != 'object':
            continue

        series = out[col].map(_parquet_safe_value)
        non_null = series.dropna()
        if non_null.empty:
            out[col] = series
            continue

        types = {type(value) for value in non_null}
        if types.issubset({str}):
            out[col] = series.astype(object).where(series.notna(), np.nan)
        elif len(types) > 1:
            out[col] = series.where(series.isna(), series.astype(str))
        else:
            out[col] = series

    return out


def parquet_buffer(df, export_name):
    os.makedirs(os.path.dirname(export_name), exist_ok=True)
    data = _normalize_parquet_columns(df)

    try:
        data = data.drop_duplicates()
    except TypeError:
        data = _normalize_parquet_columns(data.astype({col: 'string' for col in data.select_dtypes(include='object').columns}))
        data = data.drop_duplicates()

    try:
        data.to_parquet(export_name, index=False)
    except (TypeError, ValueError, OverflowError):
        data = data.copy()
        for col in data.select_dtypes(include='object').columns:
            data[col] = data[col].where(data[col].isna(), data[col].astype(str))
        data.to_parquet(export_name, index=False)

def pbp(seasons='now', start=None, end=None, local=False, append=False):
    errors=[]
    if seasons == 'now':
        seasons = [seasons]

    for season in seasons:
        try:
            if season == 'now':
                scores = wsba.nhl_scrape_schedule().query("season_type in [2,3]")
                games = scores['game_id'].drop_duplicates().to_list()
                current = scores['season'].iloc[0]
                
                data, roster = wsba.nhl_scrape_game(games,remove=[],sources=True,errors=True,xg=True,export_roster=True)
            else:
                data, roster = wsba.nhl_scrape_season(season,start=start,end=end,local=local,remove=[],xg=True,sources=True,errors=True,export_roster=True)
                current = season

            errors.extend(data['errors'])
            scraped = data['pbp']

            scraped["game_id"] = scraped["game_id"].astype('int64')
            games = scraped['game_id'].drop_duplicates().to_list()
            
            if append:
                dataset = pd.read_parquet(f'{DATA_ROOT}/pbp/nhl_pbp_{current}.parquet')
                rosterset = pd.read_parquet(f'{DATA_ROOT}/games/rosters/nhl_rosters_{current}.parquet')

                dataset['game_id'] = dataset['game_id'].astype('int64')
                rosterset['game_id'] = rosterset['game_id'].astype('int64')

                #Remove games already in the dataset to avoid duplicate or conflicting entries
                dataset = dataset.loc[~(dataset['game_id'].isin(games))]
                rosterset = rosterset.loc[~(rosterset['game_id'].isin(games))]

            else:
                dataset = pd.DataFrame()
                rosterset = pd.DataFrame()

            combined = pd.concat([dataset,scraped]).sort_values(by=['season','game_id','period','seconds_elapsed','event_num'])
            parquet_buffer(combined, f'{DATA_ROOT}/pbp/nhl_pbp_{current}.parquet')

            combined = pd.concat([rosterset,roster]).sort_values(by=['game_id'])
            parquet_buffer(combined, f'{DATA_ROOT}/games/rosters/nhl_rosters_{current}.parquet')

            for group in ['skater','team','goalie']:
                scraped_games = gen_stats(scraped, group)

                if append:
                    games_df = pd.read_parquet(f'{DATA_ROOT}/games/{group}_games/wsba_nhl_{current}_{group}_games.parquet')

                    games_df['game_id'] = games_df['game_id'].astype('int64')

                    #Remove games already in the dataset to avoid duplicate or conflicting entries
                    games_df = games_df.loc[~(games_df['game_id'].isin(games))]
                else:
                    games_df = pd.DataFrame()

                combined_games = pd.concat([games_df,scraped_games])
                parquet_buffer(combined_games, f'{DATA_ROOT}/games/{group}_games/wsba_nhl_{current}_{group}_games.parquet')
            
        except Exception as e:
            print(f'No data found for selected season or date... Exception: {e}')
                
    print(f'Errors: {errors}')

def roster(seasons):
    if isinstance(seasons, int):
        seasons = [seasons]

    for season in seasons:
        games = pd.read_parquet(f'{DATA_ROOT}/tools/schedule/schedule.parquet')
        games = games.loc[(games['season']==season)&(games['season_type']>1), 'game_id'].to_list()
        df = wsba.nhl_scrape_game_roster(games)
        
        parquet_buffer(df, f'{DATA_ROOT}/games/rosters/nhl_rosters_{season}.parquet')

def load_pbp(seasons):
    if isinstance(seasons, int):
        seasons = [seasons]

    return pd.concat([pd.read_parquet(f'{DATA_ROOT}/pbp/nhl_pbp_{season}.parquet') for season in seasons])

def load_network(seasons):
    if isinstance(seasons, int):
        seasons = [seasons]
        
    return pd.concat([pd.read_csv(f'{DATA_ROOT}/network/wsba_nhl_{season}_network.csv') for season in seasons])

def load_games(seasons, group):
    if isinstance(seasons, int):
        seasons = [seasons]
        
    return pd.concat([pd.read_parquet(f'{DATA_ROOT}/games/{group}_games/wsba_nhl_{season}_{group}_games.parquet') for season in seasons])

def load_rosters(seasons):
    if isinstance(seasons, int):
        seasons = [seasons]
        
    return pd.concat([pd.read_parquet(f'{DATA_ROOT}/games/rosters/nhl_rosters_{season}.parquet') for season in seasons])

def load_rapm(seasons, group):
    if isinstance(seasons, int):
        seasons = [seasons]

    if group == 'skater':
        suf = ''
    else:
        suf = f'_{group}'

    shifts = pd.concat([
        pd.read_csv(f'{DATA_ROOT}/rapm/nhl_rapm{suf}_{s}.csv') for s in seasons
    ]) if group in ['skater', 'coach'] else pd.DataFrame()
    shots = pd.concat([
        pd.read_csv(f'{DATA_ROOT}/shot_rapm/nhl_shot_rapm{suf}_{s}.csv') for s in seasons
    ]) if group in ['skater', 'goalie'] else pd.DataFrame()
    pens = pd.concat([
        pd.read_csv(f'{DATA_ROOT}/penalty_rapm/nhl_penalty_rapm{suf}_{s}.csv') for s in seasons
    ]) if group in ['skater', 'coach'] else pd.DataFrame()

    return shifts, shots, pens

def gen_stats(df,group):
    dfs = []
    for strength in STRENGTHS:
        for season_type in [2,3]:
            data = wsba.nhl_calculate_stats(df,group,strength,season_type)
            dfs.append(data)

    stat = pd.concat(dfs)

    return stat

def build_stats(arg,seasons):
    #Stats building
    for group in arg:
        for season in seasons:
            df = pd.read_parquet(f'{DATA_ROOT}/pbp/nhl_pbp_{season}.parquet')
            stat = gen_stats(df, group)
            
            subdir = 'games'
            groupdir = f'{group}_games'
            parquet_buffer(stat, f'{DATA_ROOT}/{subdir}/{groupdir}/wsba_nhl_{season}_{groupdir}.parquet')    

def build_edge(seasons):
    for season in seasons:
        if season < 20212022:
            pass
        else:
            roster = pd.read_parquet(f'{DATA_ROOT}/stats/skater/wsba_nhl_{season}_skater.parquet')
            players = roster['player_id'].drop_duplicates().astype(int).to_list()

            df = wsba.nhl_scrape_edge(season, 'skater', players)

            parquet_buffer(df, f'{DATA_ROOT}/stats/edge/wsba_nhl_{season}_edge.parquet')

def build_a3z(seasons, tasks=['pbp', 'join', 'stats'], simple=False):
    if simple:
        roster = load_rosters(seasons)
        a3z = nhl_scrape_a3z_pbp(f'{DATA_ROOT}/a3z', simple=True)

        stats = pd.merge(
            a3z, roster[['game_id', 'sweater_number', 'team_abbr', 'player_id']],
            how='left', 
            left_on=['game_id', 'player_num', 'team_abbr'],
            right_on=['game_id', 'sweater_number', 'team_abbr']
        )

        for season in seasons:
            stats.query("season == @season").to_csv(f'{DATA_ROOT}/a3z/games/skater_games/wsba_nhl_{season}_a3z_skater_games.csv', index=False)
    else:
        for season in seasons:
            roster = load_rosters(season)

            if 'pbp' in tasks:
                a3z_pbp = nhl_scrape_a3z_pbp(f'{DATA_ROOT}/a3z', season)
                a3z_pbp.to_csv(f'{DATA_ROOT}/a3z/raw/a3z_{season}.csv', index=False)
                
            if 'join' in tasks:
                a3z_pbp = pd.read_csv(f'{DATA_ROOT}/a3z/raw/a3z_{season}.csv')

                print('Rosters for raw A3Z data...')
                a3z_pbp = pd.merge(
                    a3z_pbp, roster[['game_id','team_abbr', 'sweater_number', 'player_id']], how='left',
                    left_on=['game_id', 'event_team_abbr', 'event_player_1_num'],
                    right_on=['game_id', 'team_abbr', 'sweater_number']
                ).drop(columns=['team_abbr', 'sweater_number'], errors='ignore').rename(columns={'player_id': 'event_player_1_id'})
                
                print('Joining dataframes...')
                pbp = load_pbp(season)
                data = nhl_join_plays(pbp, a3z_pbp)
                data.to_csv(f'{DATA_ROOT}/a3z/pbp/nhl_a3z_{season}.csv', index=False)
            else:
                data = pd.read_csv(f'{DATA_ROOT}/a3z/pbp/nhl_a3z_{season}.csv')

            if 'stats' in tasks:
                data['season'] = season
                stats = pd.concat([nhl_calculate_microstats(data, strength, 2, f'{DATA_ROOT}/games/rosters') for strength in ['5v5', '5v4', '4v5']])

                stats.to_csv(f'{DATA_ROOT}/a3z/games/skater_games/wsba_nhl_{season}_a3z_skater_games.csv', index=False)

def assess_a3z(seasons):
    for season in seasons:
        df = pd.read_csv(f'{DATA_ROOT}/a3z/pbp/nhl_a3z_{season}.csv')

        xg_shots = (df['xG']>0).sum()
        shots = df['event_type'].isin(wsba.FENWICK_EVENTS).sum()
        non_shots = (df['event_type']=='entry-sequence').sum()

        print(f'{season}\nxG Shots: {xg_shots}\nShots: {shots}\nShot Coverage: {((xg_shots/shots)*100).round(2)}%\nNon-Shots: {non_shots}\n')

def build_game_score(seasons):
    for season in seasons:
        df = load_games(season, 'skater')
        gs = wsba_game_score(df, train=False)
        
        gs.to_csv(f'{DATA_ROOT}/games/game_score/wsba_nhl_{season}_game_score.csv',index=False)

def build_rapm(seasons, targets=['expected_goals', 'goals', 'fenwick', 'corsi'], tasks=['shifts', 'shots', 'penalties']):
    for season in seasons:
        pbp = load_pbp(season).query("season_type == 2")

        if 'shifts' in tasks:
            ev_rapm, ev_coach = wsba_rapm(
                pbp, 
                wsba.STRENGTH_GROUPS['even_strength'], 
                targets=targets, 
                model_path=f'{WSBA_ML_ROOT}/rapm/shift_by_shift/shot_rates/wsba_reg_shots_ev.csv',
                params={
                    'expected_goals': {
                        'alpha': 50000,
                        'param_grid': {'alpha': np.logspace(np.log10(3000), np.log10(200000), 300)},
                        'cv': 10,
                        'n_jobs': 4
                    },
                    'fenwick': {
                        'alpha': 100000,
                        'param_grid': {'alpha': np.logspace(np.log10(3000), np.log10(200000), 300)},
                        'cv': 10,
                        'n_jobs': 4
                    },
                    'corsi': {
                        'alpha': 150000,
                        'param_grid': {'alpha': np.logspace(np.log10(3000), np.log10(200000), 300)},
                        'cv': 10,
                        'n_jobs': 4
                    },
                    'goals': {
                        'alpha': 20000,
                        'param_grid': {'alpha': np.logspace(np.log10(3000), np.log10(200000), 300)},
                        'cv': 10,
                        'n_jobs': 4
                    }
                }
            )

            st_rapm, st_coach = wsba_rapm(
                pbp, 
                wsba.STRENGTH_GROUPS['power_play'], 
                targets=targets, 
                model_path=f'{WSBA_ML_ROOT}/rapm/shift_by_shift/shot_rates/wsba_reg_shots_st.csv',
                params={
                    'expected_goals': {
                        'alpha': 10000,
                        'param_grid': {'alpha': np.logspace(np.log10(3000), np.log10(200000), 300)},
                        'cv': 10,
                        'n_jobs': 4
                    },
                    'fenwick': {
                        'alpha': 20000,
                        'param_grid': {'alpha': np.logspace(np.log10(3000), np.log10(200000), 300)},
                        'cv': 10,
                        'n_jobs': 4
                    },
                    'corsi': {
                        'alpha': 30000,
                        'param_grid': {'alpha': np.logspace(np.log10(3000), np.log10(200000), 300)},
                        'cv': 10,
                        'n_jobs': 4
                    },
                    'goals': {
                        'alpha': 20000,
                        'param_grid': {'alpha': np.logspace(np.log10(3000), np.log10(200000), 300)},
                        'cv': 10,
                        'n_jobs': 4
                    }
                }
            )

            rapm = pd.concat([ev_rapm, st_rapm])
            coach = pd.concat([ev_coach, st_coach])

            rapm.to_csv(f'{DATA_ROOT}/rapm/nhl_rapm_{season}.csv', index=False)
            coach.to_csv(f'{DATA_ROOT}/rapm/nhl_rapm_coach_{season}.csv', index=False)

        if 'shots' in tasks:
            skaters = []
            goalies = []
            #Shot RAPM
            
            for s in ['even_strength']:
                skater, goalie = wsba_rapm(pbp, wsba.STRENGTHS, ['goals'], True, f'{WSBA_ML_ROOT}/rapm/shot_by_shot/goals/wsba_reg_goals.csv', params={
                    'goals': {
                        'C': 0.025,
                        'cv': 10,
                        'max_iter': 5000,
                        'n_iter': 100,
                        'n_jobs': 4
                    }
                })
                
                skaters.append(skater)
                goalies.append(goalie)

            pd.concat(skaters).to_csv(f'{DATA_ROOT}/shot_rapm/nhl_shot_rapm_{season}.csv', index=False)
            pd.concat(goalies).to_csv(f'{DATA_ROOT}/shot_rapm/nhl_shot_rapm_goalie_{season}.csv', index=False)

        if 'penalties' in tasks:
            pens_rapm, coach_pens = wsba_rapm(
                pbp, 
                wsba.STRENGTHS, 
                targets=['penalty_minutes'], 
                model_path=f'{WSBA_ML_ROOT}/rapm/shift_by_shift/penalties/wsba_reg_pens.csv',
                params={
                    'penalty_minutes': {
                        'alpha': 10000,
                        'param_grid': {'alpha': np.logspace(np.log10(3000), np.log10(200000), 100)},
                        'cv': 10,
                        'n_jobs': 4
                    }
                }
            )

            pens_rapm.to_csv(f'{DATA_ROOT}/penalty_rapm/nhl_penalty_rapm_{season}.csv', index=False)
            coach_pens.to_csv(f'{DATA_ROOT}/penalty_rapm/nhl_penalty_rapm_coach_{season}.csv', index=False)

def build_network(seasons, network=['for','against']):
    missing = [
        'nhl_create_network_edges',
        'nhl_calculate_network',
        'nhl_create_defense_network',
    ]
    raise NotImplementedError(
        "build_network requires network helper functions that are not defined in wsba_ml: "
        + ", ".join(missing)
    )

    if isinstance(network, str):
        network = [network]
        
    for season in seasons:
        pbp = load_pbp(season)

        if 'for' in network:
            edges_df = nhl_create_network_edges(pbp)
            network_df = nhl_calculate_network(edges_df, 'for')
            
            parquet_buffer(edges_df, f'{DATA_ROOT}/network/edges/nhl_network_edges_{season}.parquet')
            network_df.to_csv(f'{DATA_ROOT}/network/wsba_nhl_{season}_network.csv',index=False)
        
        if 'against' in network:
            edges_df = nhl_create_defense_network(pbp)
            network_df = nhl_calculate_network(edges_df, 'against')
            
            parquet_buffer(edges_df, f'{DATA_ROOT}/network/defense/edges/nhl_network_defense_edges_{season}.parquet')
            network_df.to_csv(f'{DATA_ROOT}/network/defense/wsba_nhl_{season}_defense_network.csv',index=False)

def fix_stats(arg,seasons,split_game=False):
    #Stats building
    for group in arg:
        for season in seasons:
            print(f'Fixing stats for {group} {'game' if split_game else ''} stats in {season}...')

            subdir = 'games'
            groupdir = f'{group}_games'

            path = f'{DATA_ROOT}/{subdir}/{groupdir}/wsba_nhl_{season}_{groupdir}.parquet'
            stats = pd.read_parquet(path)

            if group != 'team':
                group_name = 'player_name'
                
                missing  = stats.loc[stats[group_name].astype(str)=='0','player_id'].drop_duplicates().to_list()
            else:
                missing = None

            if missing:
                info = wsba.nhl_scrape_player_info(missing)

                info = info[[
                    'player_id',
                    'player_name',
                    'position',
                    'headshot',
                    'height_in',
                    'weight_lbs',
                    'birth_date'
                ]]

                complete = pd.merge(stats,info,how='left',on='player_id', suffixes=('','_info')).replace({'0':np.nan})
                for col in info.columns[1:]:
                    complete[col] = complete[col].combine_first(complete[f'{col}_info'])
                    complete = complete.drop(columns=f'{col}_info')

                #Add player age
                complete['birth_date'] = pd.to_datetime(complete['birth_date'],format='mixed')
                complete['season_year'] = complete['season'].astype(str).str[0:4].astype(int)+1
                complete['age'] = complete['season_year'] - complete['birth_date'].dt.year

                season_names = {v: k for k, v in SEASON_NAMES.items()}
                complete['wsba_id'] = complete['player_id'].astype(str).str.replace('.0','')+complete['season'].replace(season_names).astype(str)+complete['team_abbr']
                complete = complete.sort_values(by=[group_name,'season','team_abbr','player_id']).drop(columns=['season_year'])  
            else:
                complete = stats

            complete['strength_state'] = complete['strength_state'].replace('a, l, l', 'all')
            complete = complete.loc[complete['strength_state'].isin(STRENGTHS+['all'])]
            
            front_col = ['player_name', 'player_id', 'season', 'game_id', 'team_abbr', 'headshot','position','handedness','height_in','weight_lbs','birth_date','nationality','age','wsba_id','season_year']
            complete = complete[[col for col in front_col if col in complete.columns] + [col for col in complete.columns if col not in front_col]]

            parquet_buffer(complete, path)

def build_gar(seasons, append=True):
    #Build GAR datasets

    # Skater
    bio = skater_dataset(seasons)[[
        'player_name', 'player_id', 'season', 'position'
    ]].drop_duplicates(subset=['player_id', 'season'])

    shifts, shots, pens = load_rapm(seasons, 'skater')

    shifts = shifts.merge(bio, how='left')
    shots = shots.merge(bio, how='left')
    pens = pens.merge(bio, how='left')

    skater = wsba_gar('skater', shifts, shots, pens)    

    # Goalie
    df = load_games(seasons, 'goalie')
    df = wsba.nhl_agg_stats(df, ['player_id', 'season', 'team_abbr'], comparison=False)[[
        'player_name', 'player_id', 'season', 'team_abbr', 'time_on_ice'
    ]]

    _, goalie_shots, _ = load_rapm(seasons, 'goalie')
    goalie_shots = df.merge(goalie_shots)

    goalie = wsba_gar('goalie', shots=goalie_shots)

    # Coach
    coach_shifts, _, coach_pens = load_rapm(seasons, 'coach')

    coach = wsba_gar('coach', shifts=coach_shifts, pens=coach_pens)

    # Team
    skater_team = wsba.nhl_agg_stats(skater, ['team_abbr', 'season', 'position'], rates=False, comparison=False).drop(columns=[col for col in skater.columns if 'player' in col])
    forward = skater_team.query("position == 'F'").rename(columns={'goals_above_replacement': 'forward_goals_above_replacement'}).drop(columns=['position'])
    dman = skater_team.query("position == 'D'").rename(columns={'goals_above_replacement': 'defenseman_goals_above_replacement'}).drop(columns=['position'])
    skater_team = wsba.nhl_agg_stats(pd.concat([forward, dman]), ['team_abbr', 'season'], rates=False, comparison=False).drop(columns=[col for col in skater_team.columns if 'player' in col])
    skater_team['skater_goals_above_replacement'] = skater_team['forward_goals_above_replacement']+skater_team['defenseman_goals_above_replacement']
    
    goalie_team = wsba.nhl_agg_stats(goalie, ['team_abbr', 'season'], rates=False, comparison=False).rename(
        columns={'goals_above_replacement': 'goalie_goals_above_replacement'}
    ).drop(columns=[col for col in goalie.columns if 'player' in col])

    coach_team = wsba.nhl_agg_stats(coach, ['team_abbr', 'season'], rates=False, comparison=False).rename(
        columns={'goals_above_replacement': 'coach_goals_above_replacement'}
    ).drop(columns=[col for col in coach.columns if 'coach' in col])

    shared_cols = [col for col in skater_team.columns if col in coach_team.columns and col not in [
        'team_abbr',
        'season'
    ]]

    skater_team = skater_team.rename(columns={col: f'skater_{col}' for col in shared_cols})
    coach_team = coach_team.rename(columns={col: f'coach_{col}' for col in shared_cols})

    dfs = [skater_team, goalie_team, coach_team]

    team = reduce(lambda left, right: pd.merge(left, right, on=['team_abbr', 'season']), dfs)

    for col in shared_cols:
        team[col] = team[f'skater_{col}']+team[f'coach_{col}']
    
    team['goals_above_replacement'] = team['skater_goals_above_replacement']+team['goalie_goals_above_replacement']+team['coach_goals_above_replacement']
    team = team.drop(columns=['position'], errors='ignore')

    standings = wsba.nhl_scrape_standings(seasons)[[
        'team_abbr', 'season', 'points', 'league_sequence'
    ]].rename(columns={'points': 'standings_points', 'league_sequence': 'standings_points_rank'})
    
    team = pd.merge(team, standings)

    if append:
        skater = pd.concat([
            pd.read_csv(f'{DATA_ROOT}/gar/nhl_gar_dataset.csv').query("season not in @seasons"),
            skater
        ])

        goalie = pd.concat([
            pd.read_csv(f'{DATA_ROOT}/gar/nhl_gar_dataset_goalie.csv').query("season not in @seasons"),
            goalie
        ])

        coach = pd.concat([
            pd.read_csv(f'{DATA_ROOT}/gar/nhl_gar_dataset_coach.csv').query("season not in @seasons"),
            coach
        ])

        team = pd.concat([
            pd.read_csv(f'{DATA_ROOT}/gar/nhl_gar_dataset_team.csv').query("season not in @seasons"),
            team
        ])

    skater.to_csv(f'{DATA_ROOT}/gar/nhl_gar_dataset.csv', index=False)
    goalie.to_csv(f'{DATA_ROOT}/gar/nhl_gar_dataset_goalie.csv', index=False)
    coach.to_csv(f'{DATA_ROOT}/gar/nhl_gar_dataset_coach.csv', index=False)
    team.to_csv(f'{DATA_ROOT}/gar/nhl_gar_dataset_team.csv', index=False)

def build_qoct(seasons, append=True):
    #Add quality of teammates and quality of competiton
    gar = pd.read_csv(f'{DATA_ROOT}/gar/nhl_gar_dataset.csv')

    ctdf = pd.concat([qoc_qot(load_pbp(season).query("season_type == 2"), gar) for season in seasons])

    if append:
        ctdf = pd.concat([
            pd.read_csv(f'{DATA_ROOT}/gar/nhl_qoc_qot_dataset.csv').query("season not in @seasons"),
            ctdf
        ])

    ctdf.to_csv(f'{DATA_ROOT}/gar/nhl_qoc_qot_dataset.csv', index=False)

def hs_workflow(seasons, scrape=False, fmt='parquet'):
    #Retrieve the dataset and select desired features
    params = {
        'season_type': ('==', 2),
        'strength_state': ('==', 'even_strength')
    }

    front_matter = [
        'player_name', 'player_id', 'season', 
        'team_abbr', 'position', 'strength_state',
        'handedness', 'height_in', 'weight_lbs',
        'age', 'birth_country', 'headshot',
        'games_played',
        'time_on_ice', 'time_on_ice_per_games_played',
        'season_type'
    ]

    add = [
        'a3z_time_on_ice',
        'offensive_zone_faceoffs_percentage',
        'defensive_zone_faceoffs_percentage',
        'neutral_zone_faceoffs_percentage',

        'fenwick_per_sixty',
        'expected_goals_per_fenwick',
        'fenwick_assists_per_sixty',
        'expected_assists_per_fenwick_assists',
        'giveaways_per_sixty',
        'takeaways_per_sixty',
        'blocked_shots_per_sixty',
        'faceoff_percentage',
        'hits_for_per_sixty',
        'hits_against_per_sixty',
        'game_score_consistency',

        'entries_per_sixty',
        'exits_per_sixty',
        'entries_against_per_sixty',
        'retrievals_per_sixty',
        'carries_per_sixty',
        'dump_ins_per_sixty',
        'pass_exits_per_sixty',
        'carry_exits_per_sixty',
        'pass_entries_per_sixty',
        'carry_entries_per_sixty',
        'carry_entries_against_per_sixty',
        'disrupted_entries_per_sixty',
        'recovered_dump_ins_per_sixty',
        'forechecking_pressures_per_sixty'
    ]

    SKATER = skater_dataset(seasons, scrape=scrape, format=fmt)
    TEAM = team_dataset(seasons, scrape=scrape, format=fmt)

    a3z_cols = [col for col in SKATER.columns if col in DATASET_COL['a3z']] + ['player_id','team_abbr','position','season','strength_state','season_type']
    non_a3z_cols = [col for col in SKATER.columns if col not in DATASET_COL['a3z']] + ['time_on_ice']

    a3z_df = SKATER[a3z_cols]
    non_a3z_df = SKATER[non_a3z_cols]

    a3z_df = a3z_df.rename(columns={'a3z_time_on_ice':'time_on_ice'})

    non_a3z_df = non_a3z_df.sort_values(by=['player_id','season','game_date'])

    a3z_df = wsba.nhl_agg_stats(a3z_df, group_by=['player_id','season','position','season_type','strength_state'], manual_agg={'team_abbr': 'last'}, params=params).rename(columns={'time_on_ice':'a3z_time_on_ice'})
    non_a3z_df = wsba.nhl_agg_stats(non_a3z_df, group_by=['player_id','season','position','season_type','strength_state'], manual_agg={'team_abbr': 'last', 'game_date': 'last', 'games_played': 'sum'}, params=params)
    
    skater = pd.merge(non_a3z_df.drop(columns=['a3z_time_on_ice']), a3z_df.drop(columns=[col for col in a3z_df.columns if ('time_on_ice' in col and 'a3z' not in col)]+['season_type','team_abbr']), how='left')
    
    #Add consistency metric
    gs = pd.concat([pd.read_csv(f'{DATA_ROOT}/games/game_score/wsba_nhl_{s}_game_score.csv') for s in seasons])
    gs = calculate_game_score_metrics(gs, ['player_id', 'season'])[['player_id','season','game_score_consistency']]

    skater = pd.merge(skater, gs, how='left')

    skater.to_csv(f'{DATA_ROOT}/datasets/nhl_skater_agg_dataset.csv', index=False)
    
    gar = pd.read_csv(f'{DATA_ROOT}/gar/nhl_gar_dataset.csv')
    load = gar.query("season in @seasons")
    gar = wsba.nhl_agg_stats(load[[col for col in [
        'player_id','season','position',
        'even_strength_time_on_ice_per_games_played',
        'power_play_time_on_ice_per_games_played',
        'penalty_kill_time_on_ice_per_games_played',
        'even_strength_generation',
        'even_strength_suppression',
        'power_play_generation',
        'power_play_suppression',
        'penalty_kill_generation',
        'penalty_kill_suppression',
        'generation', 'suppression',
        'exit_offense', 'entry_offense',
        'exit_defense', 'entry_defense',
        'advancement', 'prevention',
        'finishing', 'induction',
        'taken_penalties','drawn_penalties',
        'even_strength','power_play','penalty_kill',
        'offense', 'defense', 'transition',
        'production', 'penalties',
        'expected_goals_above_replacement',
        'goals_above_replacement'
    ] if col in load.columns]], ['player_id', 'season', 'position'], rates=False, comparison=False, manual_agg={
        f'{s}_time_on_ice_per_games_played': 'mean' for s in wsba.STRENGTH_GROUPS.keys()
    })
    
    total = pd.merge(
        skater[front_matter+add], gar, how='left'
    ).dropna(subset=['goals_above_replacement'])
    
    ctdf = pd.read_csv(f'{DATA_ROOT}/gar/nhl_qoc_qot_dataset.csv')

    total = pd.merge(total, ctdf, how='left')

    pp_ineligible = (total['power_play_time_on_ice_per_games_played']<1 | total['power_play_time_on_ice_per_games_played'].isna())
    pk_ineligible = (total['penalty_kill_time_on_ice_per_games_played']<1 | total['penalty_kill_time_on_ice_per_games_played'].isna())

    total.loc[pp_ineligible, 'power_play_generation'] = np.nan
    total.loc[pk_ineligible, 'penalty_kill_suppression'] = np.nan

    total.to_csv(f'{DATA_ROOT}/datasets/nhl_hs_dataset.csv', index=False)

    #Weighted average of past three seasons of data for cumulative 
    last_three = seasons[-3:]
    proj = total.query("season in @last_three").sort_values(['player_id', 'season'])
    proj['weight'] = 2 ** proj.groupby('player_id').cumcount()

    weighted_cols = [
        c for c in proj.columns
        if (c not in front_matter or 'time_on_ice' in c) and c not in ['birth_date','weight']
    ]
    
    proj = (
        proj.assign(**{
            f'{c}_w': proj[c] * proj['weight']
            for c in weighted_cols
        })
        .groupby(['player_id', 'position'], as_index=False)
        .agg({
            **{c: 'last' for c in proj.columns if c not in weighted_cols},
            **{f'{c}_w': 'sum' for c in weighted_cols},
            'weight': 'sum',
        })
    )

    for c in weighted_cols:
        proj[c] = proj[f'{c}_w'] / proj['weight']

    proj['season'] = '2023-2026'
    proj['strength_state'] = 'all'
    
    proj = proj[total.columns]
    proj.to_csv(f'{DATA_ROOT}/datasets/nhl_agg_dataset.csv',index=False)

    final = pd.concat([total, proj])
    final['wsba_id'] = final['player_id'].astype(str)+final['season'].astype(str)

    pp_ineligible = (final['power_play_time_on_ice_per_games_played']<1 | final['power_play_time_on_ice_per_games_played'].isna())
    pk_ineligible = (final['penalty_kill_time_on_ice_per_games_played']<1 | final['penalty_kill_time_on_ice_per_games_played'].isna())

    final.loc[pp_ineligible, 'power_play_generation'] = np.nan
    final.loc[pk_ineligible, 'penalty_kill_suppression'] = np.nan

    publish_data(final, 'skater')

    tgar = pd.read_csv(f'{DATA_ROOT}/gar/nhl_gar_dataset_team.csv')
    strengths = ['5v5', '4v4', '3v3']
    tdf = TEAM.query('strength_state in @strengths and season_type == 2')
    tdf = wsba.nhl_agg_stats(tdf, ['team_abbr', 'season'])[[
        'team_abbr',
        'season',
        'expected_goals_for_per_sixty',
        'expected_goals_against_per_sixty',
        'goals_for_per_sixty',
        'goals_against_per_sixty',
        'fenwick_for_per_sixty',
        'fenwick_against_per_sixty',
        'expected_goals_for_per_fenwick_for',
        'expected_goals_against_per_fenwick_against',
        'goals_for_per_expected_goals_for',
        'goals_against_per_expected_goals_against'
    ]]

    tdf[[col for col in tdf.columns if 'against' in col]] *= -1

    tdf['wsba_id'] = tdf['team_abbr']+tdf['season'].astype(str)

    tdf = pd.merge(tgar, tdf, how='inner')

    publish_data(tdf, 'team')

def repair_pbp(seasons):
    if isinstance(seasons, int):
        seasons = [seasons]

    for season in seasons:
        df = load_pbp(season)
        roster = load_rosters(season)

        df['game_id'] = df['game_id'].astype(int)
        roster['game_id'] = roster['game_id'].astype(int)
        roster['player_id'] = roster['player_id'].astype(float)

        roster = roster.drop_duplicates(['game_id','player_id'])

        num_dict = roster.set_index(['game_id','player_id'])['sweater_number']

        for i in range(3):
            keys = pd.MultiIndex.from_arrays([
                df['game_id'],
                df[f'event_player_{i+1}_id']
            ])
            df[f'event_player_{i+1}_num'] = num_dict.reindex(keys).values

        keys = pd.MultiIndex.from_arrays([
            df['game_id'],
            df['event_goalie_id']
        ])
        df['event_goalie_num'] = num_dict.reindex(keys).values

        pos_dict = roster.set_index(['game_id','player_id'])['position']

        for venue in ['away', 'home']:
            for i in range(6):

                col_id = f'{venue}_on_{i+1}_id'
                col_pos = f'{venue}_on_{i+1}_pos'

                if col_id in df.columns:

                    df[col_id] = (
                        df[col_id]
                        .replace(r'^\s*$', np.nan, regex=True)
                        .astype(float)
                    )

                    keys = pd.MultiIndex.from_arrays([
                        df['game_id'],
                        df[col_id]
                    ])

                    df[col_pos] = pos_dict.reindex(keys).values

                else:
                    df[col_pos] = np.nan


        pbp = wsba.nhl_apply_xG(df)
        parquet_buffer(pbp, f'{DATA_ROOT}/pbp/nhl_pbp_{season}.parquet')

