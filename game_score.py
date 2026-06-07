import pickle
import pandas as pd
import numpy as np
import wsba_hockey as wsba
from functools import reduce
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = Path(__file__).resolve().parent / 'game_score' / 'wsba_game_score.pkl'

BIO = ['game_id','player_id','season','team_abbr']
GS_TARGET = 'goal_differential'
GS_VARIABLES = [
    'even_strength_expected_goals_for',
    'even_strength_expected_goals_against',
    'power_play_expected_goals_for',
    'power_play_expected_goals_against',
    'penalty_kill_expected_goals_for',
    'penalty_kill_expected_goals_against',
    'goals',
    'primary_assists',
    'secondary_assists',
    'expected_goals',
    'primary_expected_assists',
    'secondary_expected_assists',
    'takeaways',
    'giveaways',
    'hits_applied',
    'hits_received',
    'blocked_shots',
    'penalties_taken',
    'penalties_drawn',
    'faceoff_wins',
    'faceoff_losses'
]

def normalize_game_score_schema(stats):
    stats = stats.copy()

    for col in BIO + [GS_TARGET] + GS_VARIABLES:
        if col not in stats.columns and col != GS_TARGET:
            stats[col] = 0

    return stats

def prepare_game_score(stats):
    stats = normalize_game_score_schema(stats)

    #Condense strengths to EV/PP/PK
    stats['strength_state'] = stats['strength_state'].map(
        wsba.STRENGTH_MAP
    )
    
    #Iterate through each strength, collecting on-ice metrics
    dfs = []
    for strength in wsba.STRENGTH_GROUPS.keys():
        str_df = stats.query(f"strength_state == '{strength}'")

        measures = [
            'time_on_ice',
            'expected_goals_for',
            'expected_goals_against'
        ]

        agg = wsba.nhl_agg_stats(str_df[BIO+measures], BIO, rates=False, comparison=False).rename(columns={
            col: f'{strength}_{col}' for col in measures
        })

        dfs.append(agg[BIO+[f'{strength}_{col}' for col in measures]])

    split_df = reduce(lambda left, right: pd.merge(left, right, how='left'), dfs)

    #Individual metrics and target variable are at all-strengths
    measures = [
        'time_on_ice',
        'goals_for',
        'goals_against',
        'goals',
        'primary_assists',
        'secondary_assists',
        'expected_goals',
        'primary_expected_assists',
        'secondary_expected_assists',
        'takeaways',
        'giveaways',
        'hits_applied',
        'hits_received',
        'blocked_shots',
        'penalties_taken',
        'penalties_drawn',
        'faceoff_wins',
        'faceoff_losses'
    ]
 
    all_df = wsba.nhl_agg_stats(stats[BIO+measures], BIO, rates=False, comparison=False)
    df = pd.merge(split_df, all_df[BIO+measures], how='left')

    #Add target
    df['goal_differential'] = df['goals_for']-df['goals_against']

    return df

def wsba_game_score(stats, train=False):
    #WSBA Game Score Model

    df = prepare_game_score(stats)

    df = df.loc[df['time_on_ice'] > 0].fillna(0)
    
    if train:
        target = df[GS_TARGET]
        weight = df['time_on_ice']
        predictors = df[GS_VARIABLES]

        #cv = RidgeCV(alphas=np.logspace(-6, 6, 200), cv=5)
        #cv.fit(predictors, target, sample_weight=weight)

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=4,
            random_state=42
        )
        model.fit(predictors, target, sample_weight=weight)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MODEL_PATH.open('wb') as f:
            pickle.dump(model, f)
    else:
        with MODEL_PATH.open('rb') as f:
            model = pickle.load(f)
    
    predictors = df[GS_VARIABLES].values
    df['game_score'] = model.predict(predictors)

    return df

def calculate_game_score_metrics(gs_stats, group_by=['player_id','season','team_abbr']):
    def std(gs):
        x = np.array(gs)
        sd = np.std(x, ddof=1)

        return abs(sd)
    
    gs_stats['game_score_consistency'] = gs_stats['game_score']
    df = wsba.nhl_agg_stats(
        gs_stats, group_by, manual_agg={
            'game_score_consistency': std
        }, rates=False, comparison=False
    )[['player_id','season','team_abbr','time_on_ice']+[GS_TARGET]+GS_VARIABLES+['game_score','game_score_consistency']]

    return df
