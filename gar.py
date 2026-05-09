import pandas as pd
import wsba_hockey as wsba
from functools import reduce
from wsba_hockey.tools.agg import *
from wsba_hockey.tools.globals import *

### GAR FUNCTIONS ###
# Provided in this file are functions for assessing player value through on-ice contributions with the WSBA Goals Above Replacement Model #

#Load globals

COMPONENTS = [
    'even_strength_generation',
    'even_strength_suppression',
    'power_play_generation',
    'penalty_kill_suppression',
    'finishing',
    'induction',
    'taken_penalties',
    'drawn_penalties'
]

RATINGS = {
    'generation': [
        'even_strength_generation',
        'power_play_generation'
    ],
    'suppression': [
        'even_strength_suppression',
        'penalty_kill_suppression'
    ],
    'production': [
        'finishing',
        'induction'
    ],
    'even_strength': [
        'even_strength_generation',
        'even_strength_suppression'
    ],
    'penalties': [
        'taken_penalties',
        'drawn_penalties'
    ],
    'offense': [
        'generation',
        'finishing',
        'induction'
    ],
    'defense': [
        'suppression'
    ],
    'expected_goals_above_replacement': [
        'even_strength_generation',
        'even_strength_suppression',
        'power_play_generation',
        'penalty_kill_suppression',
        'taken_penalties',
        'drawn_penalties'
    ],
    'goals_above_replacement': [
        'even_strength_generation',
        'even_strength_suppression',
        'power_play_generation',
        'penalty_kill_suppression',
        'finishing',
        'induction',
        'taken_penalties',
        'drawn_penalties'
    ]
}

REPLACEMENT_THRESHOLD = {
    'even_strength': {'F': 13, 'D': 7},
    'power_play': {'F': 9, 'D': 4},
    'penalty_kill': {'F': 8 ,'D': 6},
    'all': {'F': 13, 'D': 7, 'G': 3}
}

def calc_replacement(df, col):
    if 'even_strength' in col:
        situation = 'even_strength'
        toi_col = 'even_strength_time_on_ice'
    elif 'power_play' in col:
        situation = 'power_play'
        toi_col = 'power_play_time_on_ice'
    elif 'penalty_kill' in col:
        situation = 'penalty_kill'
        toi_col = 'penalty_kill_time_on_ice'
    else:
        situation = 'all'
        toi_col = 'time_on_ice'

    # TOI per game + rank (season, position)
    toi_pg = df[f'{toi_col}_per_games_played']
    rank = toi_pg.groupby([df['team_abbr'], df['season'], df['position']]).rank(method='first', ascending=False)

    # threshold per position
    threshold = df['position'].map(REPLACEMENT_THRESHOLD[situation])

    # replacement pool mask
    replacement_mask = rank >= threshold

    # compute per-60 replacement level
    rep = (
        df[replacement_mask]
        .groupby(['season','position'])
        .apply(lambda g: (g[col].sum() / g[toi_col].sum()) * 60)
        .rename('replacement_avg')
        .reset_index()
    )

    # map back to players
    df_rep = df[['season','position']].merge(rep, on=['season','position'], how='left')

    # apply replacement adjustment
    return df[col] + (df_rep['replacement_avg'] / 60 * df[toi_col])

def wsba_gar(group='skater', shifts = pd.DataFrame(), shots = pd.DataFrame(), pens = pd.DataFrame()):
    #Main function for the WSBA Goals Above Replacement Model
    if group == 'skater':
        shifts = shifts.query("season_type == 2")
        shots = shots.query("season_type == 2")
        pens = pens.query("season_type == 2")

        result = shifts[[
            'player_name', 'player_id', 'season', 'team_abbr', 'position'
        ]].drop_duplicates(subset=['player_id', 'season', 'team_abbr'], keep='last')

        #Iterate through each strength, finding per-sixty playdriving components
        dfs = []
        shifts['time_on_ice_per_games_played'] = shifts['time_on_ice'] / shifts['games_played']

        for strength in wsba.STRENGTH_GROUPS.keys():
            df = shifts.query(
                "strength_state == @strength "
                f"and time_on_ice >= {150 if strength == 'even_strength' else 50}"
            )

            df[f'{strength}_time_on_ice'] = df['time_on_ice']
            df[f'{strength}_time_on_ice_per_games_played'] = df['time_on_ice_per_games_played']

            df[f'{strength}_generation'] = df['rapm_expected_goals_for_per_sixty']
            df[f'{strength}_suppression'] = -df['rapm_expected_goals_against_per_sixty']

            dfs.append(df[[
                'player_id', 'season', 'team_abbr',
                f'{strength}_time_on_ice', f'{strength}_time_on_ice_per_games_played',
                f'{strength}_generation', f'{strength}_suppression'
            ]].dropna(how='all', axis=1))

        rapms = reduce(lambda left, right: pd.merge(left, right, how='left'), dfs)
        result = pd.merge(result, rapms, how='right')
        
        all_dfs = pd.merge(shots, pens, how='right')
        all_dfs['time_on_ice_per_games_played'] = all_dfs['time_on_ice']/all_dfs['games_played']

        all_dfs['finishing'] = all_dfs['rapm_goals'] * all_dfs['fenwick'] / all_dfs['time_on_ice'] * 60
        all_dfs['induction'] = all_dfs['rapm_primary_assists'] * all_dfs['primary_fenwick_assists'] / all_dfs['time_on_ice'] * 60
        all_dfs['secondary_induction'] = all_dfs['rapm_secondary_assists'] * all_dfs['secondary_fenwick_assists'] / all_dfs['time_on_ice'] * 60
            
        all_dfs['taken_penalties'] = -all_dfs['rapm_penalty_minutes_for_per_sixty'] * 0.02
        all_dfs['drawn_penalties'] = all_dfs['rapm_penalty_minutes_against_per_sixty'] * 0.02

        result = pd.merge(result, all_dfs[['player_id', 'season', 'team_abbr', 'time_on_ice', 'time_on_ice_per_games_played', 'finishing', 'induction', 'taken_penalties', 'drawn_penalties']], how='left')
        
        for col in COMPONENTS:
            if 'even_strength' in col:
                prefix = 'even_strength_'
            elif 'power_play' in col:
                prefix = 'power_play_'
            elif 'penalty_kill' in col:
                prefix = 'penalty_kill_'
            else:
                prefix = ''

            result[col] = (result[col]/60) * result[f'{prefix}time_on_ice']
            #result[col] = calc_replacement(result, col)

        for comp, cols in RATINGS.items():
            result[comp] = result[cols].sum(axis=1)
        
        return result
    
    elif group == 'goalie':
        result = shots[[
            'player_name', 'player_id', 'season', 'team_abbr'
        ]].drop_duplicates(subset=['player_id', 'season', 'team_abbr'], keep='last')

        shots['rapm_goals_against_per_sixty'] = shots['rapm_goals_against'] * (shots['fenwick_against'] / shots['time_on_ice']) * 60

        all_dfs = shots[[
            'player_id','season','team_abbr',
            'time_on_ice',
            'rapm_goals_against_per_sixty'
        ]]

        all_dfs['goals_above_replacement'] = -all_dfs['rapm_goals_against_per_sixty'] * all_dfs['time_on_ice'] / 60
        
        result = pd.merge(result, all_dfs[['player_id', 'season', 'team_abbr', 'time_on_ice', 'goals_above_replacement']], how='inner')
        
        #result['time_on_ice_per_games_played'] = result['time_on_ice']
        result['position'] = 'G'
        #result['goals_above_replacement'] = calc_replacement(result, 'goals_above_replacement')
        #result = result.drop(columns=['time_on_ice_per_games_played', 'position'])

        return result

    elif group == 'coach':
        shifts = shifts.query("season_type == 2")
        pens = pens.query("season_type == 2")

        result = shifts[[
            'coach_name', 'season', 'team_abbr'
        ]].drop_duplicates(subset=['coach_name', 'season', 'team_abbr'], keep='last')

        #Iterate through each strength, finding per-sixty playdriving components
        dfs = []

        for strength in wsba.STRENGTH_GROUPS.keys():
            df = shifts.query(
                "strength_state == @strength "
                f"and time_on_ice >= {150 if strength == 'even_strength' else 50}"
            )

            df[f'{strength}_time_on_ice'] = df['time_on_ice']
            df[f'{strength}_time_on_ice_per_games_played'] = df['time_on_ice'] / df['games_played']

            df[f'{strength}_generation'] = df['rapm_expected_goals_for_per_sixty']
            df[f'{strength}_suppression'] = -df['rapm_expected_goals_against_per_sixty']

            dfs.append(df[[
                'coach_name', 'season', 'team_abbr',
                f'{strength}_time_on_ice', f'{strength}_time_on_ice_per_games_played',
                f'{strength}_generation', f'{strength}_suppression'
            ]].dropna(how='all', axis=1))

        rapms = reduce(lambda left, right: pd.merge(left, right, how='left'), dfs)
        result = pd.merge(result, rapms, how='right')
                            
        pens['taken_penalties'] = -pens['rapm_penalty_minutes_for_per_sixty'] * 0.02
        pens['drawn_penalties'] = pens['rapm_penalty_minutes_against_per_sixty'] * 0.02

        result = pd.merge(result, pens[['coach_name', 'season', 'team_abbr', 'time_on_ice', 'taken_penalties', 'drawn_penalties']], how='left')

        components = [col for col in COMPONENTS if col not in ['finishing', 'induction']]
        for col in components:
            if 'even_strength' in col:
                prefix = 'even_strength_'
            elif 'power_play' in col:
                prefix = 'power_play_'
            elif 'penalty_kill' in col:
                prefix = 'penalty_kill_'
            else:
                prefix = ''

            result[col] = (result[col]/60) * result[f'{prefix}time_on_ice']

        for comp, cols in RATINGS.items():
            filtered_cols = [col for col in cols if col not in ['finishing', 'induction']]
            result[comp] = result[filtered_cols].sum(axis=1)

        return result
    else:
        raise ValueError('Invalid group type...')
