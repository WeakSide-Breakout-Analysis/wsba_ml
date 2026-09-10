import pickle
import pandas as pd
import numpy as np
import wsba_hockey as wsba
from wsba_ml.wsba_hockey_bridge import nhl_agg_stats_pandas
import matplotlib.pyplot as plt
from functools import reduce
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

MODEL_PATH = Path(__file__).resolve().parent / 'game_score' / 'wsba_game_score.pkl'
CV_PATH = Path(__file__).resolve().parent / 'game_score' / 'testing' / 'game_score_cv.csv'
METRIC_PATH = Path(__file__).resolve().parent / 'game_score' / 'metrics'

BIO = ['game_id','player_id','season','team_abbr']
GS_TARGET = 'goal_differential'
GS_VARIABLES = [
    'even_strength_time_on_ice',
    'power_play_time_on_ice',
    'penalty_kill_time_on_ice',
    'offensive_zone_faceoffs',
    'neutral_zone_faceoffs',
    'defensive_zone_faceoffs',
    'even_strength_expected_goals_for',
    'even_strength_expected_goals_against',
    'power_play_expected_goals_for',
    'power_play_expected_goals_against',
    'penalty_kill_expected_goals_for',
    'penalty_kill_expected_goals_against',
    'even_strength_goals_for',
    'even_strength_goals_against',
    'power_play_goals_for',
    'power_play_goals_against',
    'penalty_kill_goals_for',
    'penalty_kill_goals_against',
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
GS_MEASURES = {
    'deployment': [
        'even_strength_time_on_ice',
        'power_play_time_on_ice',
        'penalty_kill_time_on_ice',
        'offensive_zone_faceoffs',
        'neutral_zone_faceoffs',
        'defensive_zone_faceoffs',
    ],
    'on_ice_offense': [
        'even_strength_expected_goals_for',
        'power_play_expected_goals_for',
        'penalty_kill_expected_goals_for',
        'even_strength_goals_for',
        'power_play_goals_for',
        'penalty_kill_goals_for',
    ],
    'on_ice_defense': [
        'even_strength_expected_goals_against',
        'power_play_expected_goals_against',
        'penalty_kill_expected_goals_against',
        'even_strength_goals_against',
        'power_play_goals_against',
        'penalty_kill_goals_against',
    ],
    'individual_offense': [
        'goals',
        'primary_assists',
        'secondary_assists',
        'expected_goals',
        'primary_expected_assists',
        'secondary_expected_assists'
    ],
    'faceoffs': [
        'faceoff_wins',
        'faceoff_losses'
    ],
    'change_in_possession': [
        'takeaways',
        'giveaways'
    ],
    'physicality': [
        'hits_applied',
        'hits_received',
        'blocked_shots'
    ],
    'penalties': [
        'penalties_taken',
        'penalties_drawn'
    ],
    'even_strength': [
        'even_strength_time_on_ice',
        'even_strength_expected_goals_for',
        'even_strength_expected_goals_against',
    ],
    'power_play': [
        'power_play_time_on_ice',
        'power_play_expected_goals_for',
        'power_play_expected_goals_against',
    ],
    'penalty_kill': [
        'penalty_kill_time_on_ice',
        'penalty_kill_expected_goals_for',
        'penalty_kill_expected_goals_against',
    ]
}

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
            'expected_goals_against',
            'goals_for',
            'goals_against',
        ]

        agg = nhl_agg_stats_pandas(str_df[BIO+measures], BIO, rates=False, comparison=False).rename(columns={
            col: f'{strength}_{col}' for col in measures
        })

        dfs.append(agg[BIO+[f'{strength}_{col}' for col in measures]])

    split_df = reduce(lambda left, right: pd.merge(left, right, how='left'), dfs)

    #Individual metrics and target variable are at all-strengths
    measures = [
        'time_on_ice',
        'offensive_zone_faceoffs',
        'neutral_zone_faceoffs',
        'defensive_zone_faceoffs',
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
 
    all_df = nhl_agg_stats_pandas(stats[BIO+measures], BIO, rates=False, comparison=False)
    df = pd.merge(split_df, all_df[BIO+measures], how='left')
    strength_toi_cols = [f'{strength}_time_on_ice' for strength in wsba.STRENGTH_GROUPS.keys()]
    df['time_on_ice'] = pd.to_numeric(df['time_on_ice'], errors='coerce').fillna(0)
    df['time_on_ice'] = df['time_on_ice'].where(
        df['time_on_ice'] > 0,
        df[strength_toi_cols].sum(axis=1)
    )

    #Add target
    df['goal_differential'] = df['goals_for']-df['goals_against']

    return df

def wsba_game_score(stats, train=False, cv=5):
    #WSBA Game Score Model

    df = prepare_game_score(stats)

    df = df.loc[df['time_on_ice'] > 0].fillna(0)
    
    if train:
        target = df[GS_TARGET]
        predictors = df[GS_VARIABLES]

        groups = df['game_id']
        n_groups = groups.nunique()
        cv = min(cv, n_groups) if cv else 0
        model = LinearRegression()

        if cv >= 2:
            folds = GroupKFold(n_splits=cv)
            results = []

            for fold, (train_idx, test_idx) in enumerate(folds.split(predictors, target, groups), start=1):
                fold_model = LinearRegression()
                fold_model.fit(
                    predictors.iloc[train_idx],
                    target.iloc[train_idx],
                )
                pred = fold_model.predict(predictors.iloc[test_idx])
                y = target.iloc[test_idx]
                results.append({
                    'fold': fold,
                    'n_obs': len(test_idx),
                    'r2': r2_score(y, pred),
                    'mae': mean_absolute_error(y, pred),
                    'rmse': np.sqrt(mean_squared_error(y, pred)),
                })

            CV_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(CV_PATH, index=False)

        model.fit(predictors, target)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MODEL_PATH.open('wb') as f:
            pickle.dump(model, f)
    else:
        with MODEL_PATH.open('rb') as f:
            model = pickle.load(f)
    
    predictors = df[GS_VARIABLES]
    df['game_score'] = model.predict(predictors)

    # Add Game Score components
    coefficients = dict(zip(GS_VARIABLES, model.coef_))
    for component, measures in GS_MEASURES.items():
        col = f'game_score_{component}'
        values = pd.Series(0.0, index=df.index)
        for measure in measures:
            measure_values = df[measure] if measure in df.columns else pd.Series(0.0, index=df.index)
            values += pd.to_numeric(measure_values, errors='coerce').fillna(0) * coefficients.get(measure, 0)
        df[col] = values

    return df

def gs_coefficients(model_path=MODEL_PATH, metric_path=METRIC_PATH):
    with Path(model_path).open('rb') as f:
        model = pickle.load(f)
    fi = pd.DataFrame({
        'feature': GS_VARIABLES,
        'coefficient': model.coef_,
        'importance': np.abs(model.coef_)
    }).sort_values('coefficient', ascending=False)

    Path(metric_path).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.barh(fi['feature'], fi['coefficient'])
    plt.gca().invert_yaxis()
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel('Coefficient')
    plt.title('WSBA Game Score Coefficients')
    plt.savefig(Path(metric_path) / 'feature_importance.png', bbox_inches='tight')
    plt.close()

    return fi

def gs_model_fit(stats, metric_path=METRIC_PATH):
    print('Fit Measures for WSBA Game Score Model...')

    data = wsba_game_score(stats)
    Path(metric_path).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 7))
    plt.scatter(data[GS_TARGET], data['game_score'], s=10, alpha=0.35)
    lims = [
        min(data[GS_TARGET].min(), data['game_score'].min()),
        max(data[GS_TARGET].max(), data['game_score'].max()),
    ]
    plt.plot(lims, lims, linestyle='--')
    plt.xlabel('Actual Goal Differential')
    plt.ylabel('Predicted Game Score')
    plt.title('WSBA Game Score Fit')
    plt.savefig(Path(metric_path) / 'model_fit.png', bbox_inches='tight')
    plt.close()

    residual = data[GS_TARGET] - data['game_score']
    plt.figure(figsize=(8, 7))
    plt.scatter(data['game_score'], residual, s=10, alpha=0.35)
    plt.axhline(0, linestyle='--')
    plt.xlabel('Predicted Game Score')
    plt.ylabel('Residual')
    plt.title('WSBA Game Score Residuals')
    plt.savefig(Path(metric_path) / 'residuals.png', bbox_inches='tight')
    plt.close()

    metrics = pd.DataFrame([{
        'n_obs': len(data),
        'r2': r2_score(data[GS_TARGET], data['game_score'], sample_weight=data['time_on_ice']),
        'mae': mean_absolute_error(data[GS_TARGET], data['game_score'], sample_weight=data['time_on_ice']),
        'rmse': np.sqrt(mean_squared_error(data[GS_TARGET], data['game_score'], sample_weight=data['time_on_ice'])),
    }])
    metrics.to_csv(Path(metric_path) / 'model_fit_metrics.csv', index=False)

    return metrics

def cv_metrics(cv_path=CV_PATH, metric_path=METRIC_PATH):
    cv = pd.read_csv(cv_path)
    means = cv[['r2', 'mae', 'rmse']].mean().reset_index()
    means.columns = ['metric', 'value']

    Path(metric_path).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.bar(means['metric'], means['value'])
    plt.title('WSBA Game Score Cross-Validation Metrics')
    plt.savefig(Path(metric_path) / 'cross_validation_metrics.png', bbox_inches='tight')
    plt.close()

    return cv

def calculate_game_score_metrics(gs_stats, group_by=['player_id','season','team_abbr']):
    def std(gs):
        x = np.array(gs)
        sd = np.std(x, ddof=1)

        return abs(sd)
    
    component_cols = [f'game_score_{component}' for component in GS_MEASURES.keys() if f'game_score_{component}' in gs_stats.columns]
    gs_stats['game_score_consistency'] = gs_stats['game_score']
    df = nhl_agg_stats_pandas(
        gs_stats, group_by, manual_agg={
            'game_score_consistency': std
        }, rates=False, comparison=False
    )[['player_id','season','team_abbr','time_on_ice']+[GS_TARGET]+GS_VARIABLES+component_cols+['game_score','game_score_consistency']]

    return df
