import os
import json as json_lib
import pandas as pd
import numpy as np
import xgboost as xgb
import scipy.sparse as sp
import matplotlib.pyplot as plt
import wsba_hockey as wsba
from wsba_hockey.tools.scraping import adjust_coords
from typing import Literal
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GroupKFold

### XG_MODEL FUNCTIONS ###
# Provided in this file are functions vital to the goal prediction model in the WSBA Hockey Python package. #

def fix_players(pbp):
    #Add/fix player info for shooters and goaltenders

    #Find players that don't have a handness
    try:
        find = pbp.loc[(pbp['event_type'].isin(wsba.FENWICK_EVENTS))&(pbp['event_player_1_hand'].isna()),'event_player_1_id'].drop_duplicates().to_list()
    except:
        #If this fails then the column doesn't exist and all players are included in the search
        find = pbp['event_player_1_id'].drop_duplicates().to_list()
    
    roster = pd.read_csv(wsba.DEFAULT_ROSTER)
    roster = roster.loc[roster['player_id'].isin(find)].drop_duplicates(['player_id'])[['player_name','player_id','position','handedness']]

    #Load roster and locate players
    if not find:
        pass
    else:
        print('Adding player info to pbp...')

        #Some players are missing from the roster file (generally in newer seasons); add these manually
        miss = pbp.loc[((pbp['event_player_1_id'].isin(find))&(~pbp['event_player_1_id'].isin(roster['player_id']))),'event_player_1_id'].drop_duplicates().dropna().to_list()
        if miss:
            add = wsba.nhl_scrape_player_info(miss)[['player_name','player_id','handedness']]
            roster = pd.concat([roster,add]).reset_index(drop=True)

        #Conversion dict
        roster['player_id'] = roster['player_id'].astype('Int64')
        hand_dict = roster.set_index('player_id').to_dict()['handedness']

        #Fix event goalies
        pbp['event_goalie_id'] = np.where(pbp['event_team_venue']=='away',pbp['home_goalie_id'],pbp['away_goalie_id'])
            
        #Add hands
        pbp['event_player_1_hand'] = pbp['event_player_1_id'].astype('Int64').map(hand_dict).where(lambda x: x.notna(), None)
    
    return pbp

def apply_passing_imputation(pbp):
    #Apply the imputation scheme to estimate player passing (or setting) impacts on shot attempts
    #The process is described by Micah Blake McCurdy in his xG model writeup
    #https://hockeyviz.com/txt/xg7#setterImputation

    goals = pbp['event_type'] == 'goal'
    non_goals = pbp['event_type'].isin(wsba.FENWICK_EVENTS) & (pbp['event_type'] != 'goal')

    #Iterate through each venue and apply to venue on ice players
    for venue in ['away', 'home']:
        team_mask = pbp['event_team_venue'] == venue
        
        # Get all on-ice player and position columns at once
        player_cols = [f'{venue}_on_{j}_id' for j in range(1, 7)]
        pos_cols = [f'{venue}_on_{j}_pos' for j in range(1, 7)]
        
        # Initialize all probability columns at once
        prob_cols = {
            'primary': [f'{venue}_on_{j}_primary_fenwick_assist_probability' for j in range(1, 7)],
            'secondary': [f'{venue}_on_{j}_secondary_fenwick_assist_probability' for j in range(1, 7)],
            'tertiary': [f'{venue}_on_{j}_tertiary_fenwick_assist_probability' for j in range(1, 7)]
        }
        
        for assist_type in prob_cols:
            pbp[prob_cols[assist_type]] = 0.0
        
        # GOALS: Set known assist probabilities
        goal_team_mask = team_mask & goals
        
        # Vectorized assignment for goals
        for j in range(1, 7):
            player_col = f'{venue}_on_{j}_id'
            is_scorer = pbp[player_col] == pbp['event_player_1_id']
            is_primary_assist = pbp[player_col] == pbp['event_player_2_id']
            is_secondary_assist = pbp[player_col] == pbp['event_player_3_id']
            
            pbp.loc[goal_team_mask & is_primary_assist, f'{venue}_on_{j}_primary_fenwick_assist_probability'] = 1.0
            pbp.loc[goal_team_mask & is_secondary_assist, f'{venue}_on_{j}_secondary_fenwick_assist_probability'] = 1.0
        
        # GOALS: Compute tertiary assist probabilities
        if goal_team_mask.any():
            goal_indices = pbp.index[goal_team_mask]
            goal_on_ice_ids = pbp.loc[goal_indices, player_cols].values
            goal_on_ice_pos = pbp.loc[goal_indices, pos_cols].values
            
            # Map positions to tertiary probabilities
            tertiary_probs_goals = np.vectorize(lambda pos: wsba.POS_BASE_PROB['tertiary'].get(pos, 0))(goal_on_ice_pos)
            
            # Zero out scorer and assist players
            goal_player_1 = pbp.loc[goal_indices, 'event_player_1_id'].values[:, None]
            goal_player_2 = pbp.loc[goal_indices, 'event_player_2_id'].values[:, None]
            goal_player_3 = pbp.loc[goal_indices, 'event_player_3_id'].values[:, None]
            
            involved_mask = (
                (goal_on_ice_ids == goal_player_1) |
                (goal_on_ice_ids == goal_player_2) |
                (goal_on_ice_ids == goal_player_3)
            )
            tertiary_probs_goals[involved_mask] = 0
            
            # Normalize
            tertiary_sums = tertiary_probs_goals.sum(axis=1, keepdims=True)
            tertiary_sums[tertiary_sums == 0] = 1
            tertiary_probs_goals = (tertiary_probs_goals / tertiary_sums) * 0.8
            
            # Assign back
            pbp.loc[goal_indices, prob_cols['tertiary']] = tertiary_probs_goals
        
        # NON-GOALS: Compute all assist probabilities
        if non_goals.any():
            non_goal_team_mask = team_mask & non_goals
            non_goal_indices = pbp.index[non_goal_team_mask]
            
            on_ice_ids = pbp.loc[non_goal_indices, player_cols].values
            on_ice_pos = pbp.loc[non_goal_indices, pos_cols].values
            shooter_ids = pbp.loc[non_goal_indices, 'event_player_1_id'].values[:, None]
            
            # Map positions to probabilities for all assist types
            probs = {}
            for assist_type in ['primary', 'secondary', 'tertiary']:
                probs[assist_type] = np.vectorize(lambda pos: wsba.POS_BASE_PROB[assist_type].get(pos, 0))(on_ice_pos)
                # Zero out shooter
                probs[assist_type][on_ice_ids == shooter_ids] = 0
                # Normalize
                sums = probs[assist_type].sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                probs[assist_type] = (probs[assist_type] / sums) * 0.8
            
            # Assign all probabilities at once
            for assist_type in ['primary', 'secondary', 'tertiary']:
                pbp.loc[non_goal_indices, prob_cols[assist_type]] = probs[assist_type]
    
    return pbp

def apply_time_on_ice(pbp):
    #Apply time on ice (in shift) for all on-ice skaters and first event player (shooter)

    #Collect on ice columns
    home_cols = [f'home_on_{i}_id' for i in range(1, 7)]
    away_cols = [f'away_on_{i}_id' for i in range(1, 7)]
    pbp = pbp.copy()
    pbp['_toi_row'] = np.arange(len(pbp))
    home_long = pbp[['season', 'game_id', 'period', 'seconds_elapsed', '_toi_row'] + home_cols].melt(
        id_vars=['season', 'game_id', 'period', 'seconds_elapsed', '_toi_row'],
        value_vars=home_cols,
        value_name='player_id'
    )
    home_long['venue'] = 'home'
    away_long = pbp[['season', 'game_id', 'period', 'seconds_elapsed', '_toi_row'] + away_cols].melt(
        id_vars=['season', 'game_id', 'period', 'seconds_elapsed', '_toi_row'],
        value_vars=away_cols,
        value_name='player_id'
    )
    away_long['venue'] = 'away'
    on_ice = pd.concat([home_long, away_long], ignore_index=True)
    on_ice = on_ice.loc[on_ice['player_id'].notna()].sort_values(['season', 'game_id', 'period', 'player_id', '_toi_row'])
    on_ice['_prev_row'] = on_ice.groupby(['season', 'game_id', 'period', 'player_id'], sort=False)['_toi_row'].shift(1)
    on_ice['_new_shift'] = (on_ice['_prev_row'].isna() | (on_ice['_toi_row'] != on_ice['_prev_row'] + 1)).astype(int)
    on_ice['_shift_id'] = on_ice.groupby(['season', 'game_id', 'period', 'player_id'], sort=False)['_new_shift'].cumsum()
    on_ice['_shift_start'] = on_ice.groupby(['season', 'game_id', 'period', 'player_id', '_shift_id'], sort=False)['seconds_elapsed'].transform('first')
    on_ice['shift_time_on_ice'] = on_ice['seconds_elapsed'] - on_ice['_shift_start']

    #Shooter
    shooter_col = 'event_player_1_shift_time_on_ice'
    shooter_toi = on_ice[['_toi_row', 'player_id', 'shift_time_on_ice']].merge(
        pbp[['_toi_row', 'event_player_1_id']],
        left_on=['_toi_row', 'player_id'],
        right_on=['_toi_row', 'event_player_1_id'],
        how='inner'
    )[['_toi_row', 'shift_time_on_ice']]
    pbp = pbp.merge(shooter_toi.rename(columns={'shift_time_on_ice': shooter_col}), on='_toi_row', how='left')

    #Opponent TOI
    event_venue = pbp[['_toi_row', 'event_team_venue']]
    against = on_ice[['_toi_row', 'venue', 'shift_time_on_ice']].merge(event_venue, on='_toi_row', how='left')
    against = against.loc[
        ((against['event_team_venue'] == 'away') & (against['venue'] == 'home')) |
        ((against['event_team_venue'] == 'home') & (against['venue'] == 'away'))
    ]
    against = against.groupby('_toi_row', as_index=False)['shift_time_on_ice'].agg(
        event_on_ice_against_min_shift_time_on_ice='min',
        event_on_ice_against_avg_shift_time_on_ice='mean',
        event_on_ice_against_max_shift_time_on_ice='max'
    )
    pbp = pbp.merge(against, on='_toi_row', how='left')
    pbp = pbp.drop(columns=['_toi_row'])

    return pbp
        
def prep_xG_data(pbp):
    #Prep data for xG training and calculation
    pbp = fix_players(pbp)

    #Informal groupby
    pbp = pbp.sort_values(by=['season','game_id','period','seconds_elapsed','event_num'])
    shift_group = pbp.groupby(['season', 'game_id', 'period'], sort=False)

    #Recalibrate times series data with current data
    pbp['seconds_since_last'] = pbp['seconds_elapsed'] - shift_group['seconds_elapsed'].shift(1)
    pbp["seconds_since_last"] = pbp["seconds_since_last"].fillna(0)

    #Create last event columns
    pbp["event_team_last"] = shift_group['event_team_abbr'].shift(1)
    pbp["event_type_last"] = shift_group['event_type'].shift(1)
    pbp["x_adj_last"] = shift_group['x_adj'].shift(1)
    pbp["y_adj_last"] = shift_group['y_adj'].shift(1)
    pbp["zone_code_last"] = shift_group['zone_code'].shift(1)

    pbp = pbp.sort_values(['season','game_id','period','seconds_elapsed','event_num'])    

    #Contextual Data (for score state minimize the capture to four goals)
    pbp['score_state'] = np.where(pbp['away_team_abbr']==pbp['event_team_abbr'],pbp['away_score']-pbp['home_score'],pbp['home_score']-pbp['away_score'])
    pbp['score_state'] = np.where(pbp['score_state']>4,4,pbp['score_state'])
    pbp['score_state'] = np.where(pbp['score_state']<-4,-4,pbp['score_state'])

    pbp['strength_diff'] = np.where(pbp['away_team_abbr']==pbp['event_team_abbr'],pbp['away_skaters']-pbp['home_skaters'],pbp['home_skaters']-pbp['away_skaters'])
    pbp['strength_state_venue'] = pbp['away_skaters'].astype(str)+'v'+pbp['home_skaters'].astype(str)
    pbp['distance_from_last'] = np.sqrt((pbp['x_adj'] - pbp['x_adj_last'])**2 + (pbp['y_adj'] - pbp['y_adj_last'])**2)
    pbp['angle_from_last'] = np.degrees(np.arctan2(abs(pbp['y_adj'] - pbp['y_adj_last']), abs(89 - (pbp['x_adj']-pbp['x_adj_last']))))

    #Event speeds
    pbp['speed_from_last'] = np.where(pbp['seconds_since_last']==0,0,pbp['distance_from_last']/pbp['seconds_since_last'])
    pbp['speed_of_angle_from_last'] = np.where(pbp['seconds_since_last']==0,0,pbp['angle_from_last']/pbp['seconds_since_last'])

    #Rush, in-zone, and rebound shots are labelled
    pbp['rush'] = np.where((pbp['event_type'].isin(wsba.FENWICK_EVENTS))&(pbp['zone_code_last'].isin(['N','D']))&(pbp['zone_code']=='O')&(pbp['seconds_since_last']<=5),1,0)
    pbp['in_zone'] = np.where((pbp['event_type'].isin(wsba.FENWICK_EVENTS))&(pbp['zone_code_last']=='O')&(pbp['zone_code']=='O')&(pbp['seconds_since_last']<=5),1,0)
    pbp['rebound'] = np.where((pbp['event_type'].isin(wsba.FENWICK_EVENTS))&(pbp['event_type_last'].isin(wsba.FENWICK_EVENTS))&(pbp['seconds_since_last']<=2),1,0)

    #Create boolean variables
    pbp["is_goal"]=(pbp['event_type']=='goal').astype(int)
    pbp["is_home"]=(pbp['home_team_abbr']==pbp['event_team_abbr']).astype(int)

    #Boolean variables for shot types and prior events
    for shot in wsba.SHOT_TYPES:
        pbp[shot] = (pbp['shot_type']==shot).astype(int)
    for event in wsba.EVENTS[:-1]:
        pbp[f'prior_{event}'] = (pbp['event_type_last']==event).astype(int)

    pbp['other-shot'] = (~pbp['shot_type'].isin(wsba.SHOT_TYPES)).astype(int)
    pbp['prior_same'] = (pbp['event_team_last']==pbp['event_team_abbr']).astype(int)
    
    #Strength boolean (used instead of 'strength_diff' in order to more usefully distinguish between strength states)
    for strength in wsba.STRENGTHS:
        pbp[f'strength_{strength}'] = (pbp['strength_state']==strength).astype(int)
    
    #Flag special shot attempts added after 2021
    pbp['short'] = (pbp['event_reason']=='short').astype(int)
    pbp['failed_bank'] = (pbp['event_reason']=='failed-bank-attempt').astype(int)

    #Determine if the current play occured on the opposite side of the ice from the previous event
    pbp['cross_ice'] = (pbp['y_adj_last'] * pbp['y_adj'] < 0).astype(int)

    #Misc variables
    pbp['empty_net'] = np.where((pbp['event_type'].isin(wsba.FENWICK_EVENTS))&(pbp['event_goalie_id'].isna()),1,0)
    pbp['offwing'] = np.where(((pbp['y_adj']<0)&(pbp['event_player_1_hand']=='L'))|((pbp['y_adj']>=0)&(pbp['event_player_1_hand']=='R')),1,0)
    
    #Add shot assist probabilities
    pbp = apply_passing_imputation(pbp)

    #Return: pbp data prepared to train and calculate the xG model
    return pbp

def wsba_xG(pbp, model_type: Literal['bayesian', 'frequentist'] = 'frequentist', states = False, hypertune = False, train = False, test_path = wsba.TEST_PATH, cv_path = wsba.CV_PATH, model_path = wsba.XG_MODEL, train_runs = 20, cv_runs = 20, strengths = None, empty_net = False):
    #Train and calculate the WSBA Expected Goals model
    
    #Add index for future merging
    pbp['event_index'] = pbp.index

    #Initialize xG column for all events
    pbp['xG'] = 0.0

    #Recalibrate coordinates
    pbp = adjust_coords(pbp)

    #Recalculate stat states if speciifed
    if states:
        for venue in ['away','home']:
            pbp[f'{venue}_score'] = ((pbp['event_team_venue']==venue)&(pbp['event_type']=='goal')).groupby(pbp['game_id']).cumsum().groupby(pbp['game_id']).shift(1).fillna(0)
            pbp[f'{venue}_corsi'] = ((pbp['event_team_venue']==venue)&(pbp['event_type'].isin(['blocked-shot','missed-shot','shot-on-goal','goal']))).groupby(pbp['game_id']).cumsum().groupby(pbp['game_id']).shift(1).fillna(0)
            pbp[f'{venue}_fenwick'] = ((pbp['event_team_venue']==venue)&(pbp['event_type'].isin(['missed-shot','shot-on-goal','goal']))).groupby(pbp['game_id']).cumsum().groupby(pbp['game_id']).shift(1).fillna(0)
            pbp[f'{venue}_penalties'] = ((pbp['event_team_venue']==venue)&(pbp['event_type']=='penalty')).groupby(pbp['game_id']).cumsum().groupby(pbp['game_id']).shift(1).fillna(0)

    #Fix strengths
    pbp['strength_state'] = np.where((pbp['season_type']==3)&(pbp['period']>4),
                                    (np.where(pbp['event_team_abbr']==pbp['away_team_abbr'],
                                            pbp['away_skaters'].astype(str)+"v"+pbp['home_skaters'].astype(str),
                                            pbp['home_skaters'].astype(str)+"v"+pbp['away_skaters'].astype(str))),
                                    pbp['strength_state'])
    pbp = pbp.sort_values(by=['season','game_id','period','seconds_elapsed','event_num'])

    #Prep data and filter shot events
    data = prep_xG_data(pbp.loc[(pbp['event_type'].isin(wsba.EVENTS))&(pbp['strength_state'].isin(wsba.STRENGTHS))&(pbp['x'].notna())&(pbp['y'].notna())])
    data = data.loc[data['event_type'].isin(wsba.FENWICK_EVENTS)]
    if strengths is not None:
        data = data.loc[data['strength_state'].isin(strengths)]
    data = data.loc[data['empty_net'] == int(empty_net)]
    
    if model_type == 'bayesian':
        NotImplementedError('PyMC Model in Development...')
    else:
        training = data.copy()
        if training.empty:
            return None if train else pbp.sort_values(by=['game_id','period','seconds_elapsed','event_num'])

        #Convert to sparse
        feature_names = [
            col for col in wsba.CONTINUOUS + wsba.BOOLEAN
            if training[col].notna().any() and not training[col].fillna(0).eq(0).all()
        ]
        data_sparse = sp.csr_matrix(training[[wsba.TARGET] + feature_names])
        is_goal_vect = data_sparse[:,0].toarray()
        predictors = data_sparse[:,1:]

        #XGB DataModel
        xgb_matrix = xgb.DMatrix(data=predictors,label=is_goal_vect,feature_names=feature_names)

        if train:
            print('### XGBOOST MODEL TRAINING ###')
            groups = training['game_id'].to_numpy()
            folds = list(GroupKFold(n_splits=min(10, len(pd.unique(groups)))).split(predictors, is_goal_vect.ravel(), groups=groups))
            if hypertune:
                # DataFrames to store results
                best_df = pd.DataFrame(columns=["max_depth", "eta", "gamma", "subsample", "colsample_bytree", "min_child_weight", "max_delta_step"])
                best_ll = pd.DataFrame(columns=["ll", "ll_rounds", "auc", "auc_rounds", "seed"])

                print('### HYPERTUNING ###')
                for i in range(train_runs):
                    print(f"## LOOP: {i+1} ##")
                    param = {
                        "objective": "binary:logistic",
                        "eval_metric": ["logloss", "auc"],
                        "max_depth": 6,
                        "eta": np.random.uniform(0.06, 0.11),
                        "gamma": np.random.uniform(0.06, 0.12),
                        "subsample": np.random.uniform(0.76, 0.84),
                        "colsample_bytree": np.random.uniform(0.76, 0.8),
                        "min_child_weight": np.random.randint(5, 23),
                        "max_delta_step": np.random.randint(4, 9)
                    }
                    seed = np.random.randint(0, 10000)
                    np.random.seed(seed)
                    cv_results = xgb.cv(
                        params=param,
                        dtrain=xgb_matrix,
                        num_boost_round=1000,
                        folds=folds,
                        early_stopping_rounds=25,
                        metrics=["logloss", "auc"],
                        seed=seed
                    )
                    best_df.loc[i] = param
                    best_ll.loc[i] = [
                        cv_results["test-logloss-mean"].min(),
                        cv_results["test-logloss-mean"].idxmin(),
                        cv_results["test-auc-mean"].max(),
                        cv_results["test-auc-mean"].idxmax(),
                        seed
                    ]

                best_all = pd.concat([best_df, best_ll], axis=1).dropna().sort_values(by="auc", ascending=False)
                best_all.to_csv(test_path,index=False)
                params_grid = {
                    "objective": "binary:logistic",
                    "eval_metric": ["logloss", "auc"],
                    "eta": best_all['eta'].iloc[0],
                    "gamma": best_all['gamma'].iloc[0],
                    "subsample": best_all['subsample'].iloc[0],
                    "max_depth": int(best_all['max_depth'].iloc[0]),
                    "colsample_bytree": best_all['colsample_bytree'].iloc[0],
                    "min_child_weight": int(best_all['min_child_weight'].iloc[0]),
                    "max_delta_step": int(best_all['max_delta_step'].iloc[0]),
                }

                cv_test = pd.DataFrame(columns=["AUC_rounds", "AUC", "LL_rounds", "LL", "seed"])
                print('### CROSS-VALIDATION ###')
                for i in range(cv_runs):
                    print(f"## LOOP: {i+1} ##")
                    seed = np.random.randint(0, 10000)
                    np.random.seed(seed)
                    cv_rounds = xgb.cv(
                        params=params_grid,
                        dtrain=xgb_matrix,
                        num_boost_round=1000,
                        folds=folds,
                        early_stopping_rounds=25,
                        metrics=["logloss", "auc"],
                        seed=seed
                    )
                    cv_test.loc[i] = [
                        cv_rounds["test-auc-mean"].idxmax(),
                        cv_rounds["test-auc-mean"].max(),
                        cv_rounds["test-logloss-mean"].idxmin(),
                        cv_rounds["test-logloss-mean"].min(),
                        seed,
                    ]

                cv_final = cv_test.sort_values(by="AUC", ascending=False)
                cv_final.to_csv(cv_path,index=False)
            else:
                # Load previous parameters
                best_all = pd.read_csv(test_path)
                cv_final = pd.read_csv(cv_path)

                print('Loaded hyperparameters...')
                # Final parameters
                params_grid = {
                    "objective": "binary:logistic",
                    "eval_metric": ["logloss", "auc"],
                    "eta": best_all['eta'].iloc[0],
                    "gamma": best_all['gamma'].iloc[0],
                    "subsample": best_all['subsample'].iloc[0],
                    "max_depth": int(best_all['max_depth'].iloc[0]),
                    "colsample_bytree": best_all['colsample_bytree'].iloc[0],
                    "min_child_weight": int(best_all['min_child_weight'].iloc[0]),
                    "max_delta_step": int(best_all['max_delta_step'].iloc[0]),
                }

            print('Training model...')
            seed = int(cv_final['seed'].iloc[0])
            np.random.seed(seed)
            model = xgb.train(
                params=params_grid,
                dtrain=xgb_matrix,
                num_boost_round=int(cv_final['AUC_rounds'].iloc[0]),
                verbose_eval=2,
            )
            
            #Save model
            model.save_model(model_path)
            with open(model_path.replace('.json', '_features.json'), 'w') as f:
                json_lib.dump(feature_names, f)
            
        else:
            #Load model
            model = xgb.Booster()
            model.load_model(model_path)
            training['xG'] = model.predict(xgb_matrix)
            xg_data = training

            #Add xG columns
            for col in xg_data.columns:
                if col not in pbp.columns:
                    pbp[col] = np.nan
                    pbp[col] = pbp[col].astype('object')

            pbp.loc[xg_data.index, xg_data.columns] = xg_data

            #Return: PBP dataframe with xG columns
            pbp_xg = pbp.sort_values(by=['game_id','period','seconds_elapsed','event_num'])

            return pbp_xg

def feature_importance(model_path = wsba.XG_MODEL, metric_path = wsba.METRIC_PATH):
    print('Feature importance for WSBA xG Model...')

    model_dir = os.path.dirname(model_path)
    model_paths = {
        'ev': os.path.join(model_dir, 'wsba_xg_ev.json'),
        'pp': os.path.join(model_dir, 'wsba_xg_pp.json'),
        'pk': os.path.join(model_dir, 'wsba_xg_pk.json'),
        'en': os.path.join(model_dir, 'wsba_xg_en.json'),
    }
    if not all(os.path.exists(path) for path in model_paths.values()):
        model_paths = {
            'all': model_path,
            'en': model_path.replace('.json', '_en.json')
        }

    os.makedirs(metric_path, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for ax, (name, path) in zip(axes, model_paths.items()):
        model = xgb.Booster()
        model.load_model(path)

        fi = pd.DataFrame(model.get_score(importance_type='gain').items(), columns=['feature','gain']).sort_values("gain", ascending=False).head(20)
        fi['gain'] /= fi['gain'].sum()

        ax.barh(fi['feature'], fi['gain'])
        ax.invert_yaxis()
        ax.set_xlabel('Gain')
        ax.set_title(f'WSBA xG {name.upper()} Feature Importance')

        single_fig, single_ax = plt.subplots(figsize=(10, 7))
        single_ax.barh(fi['feature'], fi['gain'])
        single_ax.invert_yaxis()
        single_ax.set_xlabel('Gain')
        single_ax.set_title(f'WSBA xG {name.upper()} Feature Importance')
        single_fig.savefig(os.path.join(metric_path, f'feature_importance_{name}.png'), bbox_inches='tight')
        plt.close(single_fig)

    for ax in axes[len(model_paths):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(metric_path, 'feature_importance.png'), bbox_inches='tight')
    plt.close(fig)

def roc_auc_curve(pbp, metric_path = wsba.METRIC_PATH):
    print('ROC-AUC Curve for WSBA xG Model...')

    if 'xG' in pbp.columns:
        data = pbp.loc[(pbp['event_type'].isin(wsba.FENWICK_EVENTS))&(pbp['strength_state'].isin(wsba.STRENGTHS))&(pbp['x'].notna())&(pbp['y'].notna())].copy()
        data['is_goal'] = (data['event_type'] == 'goal').astype(int)
        data['xG'] = pd.to_numeric(data['xG'], errors='coerce').fillna(0.0)
        empty_mask = data['event_goalie_id'].isna()
        modeled_mask = (
            ((~empty_mask) & data['strength_state'].isin(wsba.STRENGTH_GROUPS['even_strength'])) |
            ((~empty_mask) & data['strength_state'].isin(wsba.STRENGTH_GROUPS['power_play'] + ['6v5', '6v4'])) |
            ((~empty_mask) & data['strength_state'].isin(wsba.STRENGTH_GROUPS['penalty_kill'])) |
            empty_mask
        )
        data = data.loc[modeled_mask]

        os.makedirs(metric_path, exist_ok=True)
        plt.figure(figsize=(8, 7))
        fpr, tpr, _ = roc_curve(data['is_goal'], data['xG'])
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr,tpr,label=f"Total (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"WSBA xG ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(metric_path, f'roc_auc_curve.png'), bbox_inches='tight')
        plt.close()
    else:
        print('No xG found for provided play-by-play data.  Apply xG model to the play-by-play data first.')

def calibration(pbp, metric_path = wsba.METRIC_PATH):
    print('Reliability for WSBA xG Model...')

    if 'xG' in pbp.columns: 
        data = pbp.loc[(pbp['event_type'].isin(wsba.FENWICK_EVENTS))&(pbp['strength_state'].isin(wsba.STRENGTHS))&(pbp['x'].notna())&(pbp['y'].notna())].copy()
        data['is_goal'] = (data['event_type'] == 'goal').astype(int)
        data['xG'] = pd.to_numeric(data['xG'], errors='coerce').fillna(0.0).clip(0, 1)
        empty_mask = data['event_goalie_id'].isna()
        modeled_mask = (
            ((~empty_mask) & data['strength_state'].isin(wsba.STRENGTH_GROUPS['even_strength'])) |
            ((~empty_mask) & data['strength_state'].isin(wsba.STRENGTH_GROUPS['power_play'] + ['6v5', '6v4'])) |
            ((~empty_mask) & data['strength_state'].isin(wsba.STRENGTH_GROUPS['penalty_kill'])) |
            empty_mask
        )
        data = data.loc[modeled_mask].copy()
        data['bin'] = pd.cut(data['xG'], bins=np.linspace(0, 1, 21), include_lowest=True, duplicates='drop')
        curve = data.groupby('bin', observed=True).agg(
            predicted=('xG', 'mean'),
            observed=('is_goal', 'mean'),
            shots=('is_goal', 'size')
        ).reset_index()

        os.makedirs(metric_path, exist_ok=True)
        plt.figure(figsize=(8, 7))
        plt.plot(curve['predicted'], curve['observed'], "s-", label="Total")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        plt.title(f"WSBA xG Calibration")
        plt.xlabel("Predicted Probability (mean)")
        plt.ylabel("Fraction of positives")
        plt.legend(loc="best")
        plt.savefig(os.path.join(metric_path, f'calibration.png'), bbox_inches='tight')
        plt.close()

    else:
        print('No xG found for provided play-by-play data.  Apply xG model to the play-by-play data first.')
