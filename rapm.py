import base64
import json
import zlib
import pandas as pd
import numpy as np
import scipy.sparse as sp
import wsba_hockey as wsba
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from functools import reduce
from wsba_hockey.tools.plotting import wsba_rink
from sklearn.linear_model import LogisticRegression
from scipy.sparse.linalg import cg, splu
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

DEFAULT_FEATURES = {
    'shifts':{
        ('expected_goals', 'goals', 'corsi', 'fenwick', 'shots', 'penalty_minutes'): [
            'period_2', 'period_3', 'period_4_plus',
            'offensive_faceoff', 'neutral_faceoff', 'defensive_faceoff',
            'lead_1', 'lead_2', 'lead_3_plus',
            'behind_1', 'behind_2', 'behind_3_plus',
            'offense_back_to_back', 'defense_back_to_back',
            'power_play_expiry',
            'penalty_kill_expiry',
            'is_home',
            'strength_3v3', 'strength_3v4', 'strength_3v5',
            'strength_4v3', 'strength_4v4', 'strength_4v5',
            'strength_5v3', 'strength_5v4'
        ],
    },
    'shots':{
        ('goals', ): ['expected_goals']
    }
}

RAPM_STATS = [
    'corsi',
    'fenwick',
    'shots',
    'goals',
    'expected_goals',
    'penalty_minutes'
]

RINK_Y = np.arange(-42.0, 43.0, 1.0, dtype=np.float32)
MAP_X = np.arange(0.0, 100.0, 1.0, dtype=np.float32)
MAP_Y = RINK_Y
MAP_SHAPE = (len(MAP_X), len(MAP_Y))
MAP_SIZE = MAP_SHAPE[0] * MAP_SHAPE[1]
MAP_TARGETS = {'fenwick', 'shots', 'goals', 'expected_goals'}
MAP_BLOCK_SIZE = 512

def parse_strength(strength):
    try:
        left, right = str(strength).split('v')
        return int(left), int(right)
    except (TypeError, ValueError):
        return None

def reference_strength(strengths):
    candidates = []
    for strength in strengths:
        parsed = parse_strength(strength)
        if parsed is None:
            continue
        left, right = parsed
        if left > 5 or right > 5:
            continue
        candidates.append((left + right, min(left, right), left, right, strength))

    if not candidates:
        return None
    return max(candidates)[-1]

def encode_rapm_map(map_array):
    data = np.asarray(map_array, dtype=np.float16)
    return base64.b64encode(zlib.compress(data.tobytes(), level=9)).decode('ascii')

def decode_rapm_map(encoded_map):
    data = np.frombuffer(zlib.decompress(base64.b64decode(encoded_map)), dtype=np.float16).astype(np.float32)
    return data.reshape(MAP_SHAPE)

def season_values(season):
    if season is None:
        return None
    if isinstance(season, (list, tuple, set, np.ndarray, pd.Series)):
        values = [int(value) for value in season if pd.notna(value)]
        return tuple(sorted(set(values)))
    if pd.isna(season):
        return None
    return (int(season),)

def center_series(frame, column):
    if column not in frame.columns:
        return
    values = pd.to_numeric(frame[column], errors='coerce')
    mean = values.mean()
    if pd.isna(mean):
        return
    frame[column] = values - mean
    for suffix in ['ci_lower', 'ci_upper']:
        interval_col = f'{column}_{suffix}'
        if interval_col in frame.columns:
            frame[interval_col] = pd.to_numeric(frame[interval_col], errors='coerce') - mean
    se_col = f'{column}_standard_error'
    if se_col in frame.columns:
        frame[f'{column}_p_value'] = coefficient_p_values(frame[column], frame[se_col])


def center_rapm_columns(frame, value_columns):
    frame = frame.copy()
    for column in value_columns:
        center_series(frame, column)
    return frame

def transform_rapm_map(encoded_map, reverse_x=False, reverse_y=False):
    grid = decode_rapm_map(encoded_map)
    if reverse_x:
        grid = grid[::-1, :]
    if reverse_y:
        grid = grid[:, ::-1]
    return encode_rapm_map(grid)

def encode_raw_maps(raw_maps):
    encoded_maps = []
    for start in range(0, len(raw_maps), MAP_BLOCK_SIZE):
        end = min(start + MAP_BLOCK_SIZE, len(raw_maps))
        batch = raw_maps[start:end].reshape(end - start, MAP_SHAPE[0], MAP_SHAPE[1]).astype(np.float64)
        encoded_maps.extend([encode_rapm_map(grid) for grid in batch])
    return encoded_maps

def shift_cols():
    return [
        'season', 'game_id', 'shift_id', 'score_state',
        'offense_team_abbr', 'defense_team_abbr',
        *[f'offense_on_{i}_id' for i in range(1,7)],
        *[f'defense_on_{i}_id' for i in range(1,7)],
        'offense_goalie_id', 'defense_goalie_id',
        'offense_coach', 'defense_coach',
        *[f'{stat}_for' for stat in RAPM_STATS],
        *[f'{stat}_for_per_sixty' for stat in RAPM_STATS],
        'period_2', 'period_3', 'period_4_plus',
        'offensive_faceoff', 'neutral_faceoff', 'defensive_faceoff', 'on_the_fly',
        'tied',
        'lead_1', 'lead_2', 'lead_3_plus',
        'behind_1', 'behind_2', 'behind_3_plus',
        'offense_back_to_back', 'defense_back_to_back',
        'power_play_expiry', 'penalty_kill_expiry',
        'scoreless', 'last_penalty',
        'shift_length', 'is_home'
    ]

def select_target_predictors(sparse, predictor_names, target, feature_filters=None):
    predictor_names = list(predictor_names)
    # Treat only encoded entity dummy columns as entity terms. Contextual
    # features like offense_back_to_back / defense_back_to_back should remain
    # scalar context predictors and be preserved in saved coefficient CSVs.
    context_entity_exceptions = {'offense_back_to_back', 'defense_back_to_back'}
    entity_mask = np.array([
        (
            name.startswith((
                'offense_',
                'defense_',
                'offense_coach_',
                'defense_coach_',
                'shooter_',
                'goalie_',
                'primary_passer_',
                'secondary_passer_',
                'tertiary_passer_',
            ))
            and name not in context_entity_exceptions
        )
        for name in predictor_names
    ], dtype=bool)

    matched_filter = False
    selected_features = []
    if feature_filters:
        for target_group, features in feature_filters.items():
            if isinstance(target_group, str):
                matches_target = (target_group == target)
            else:
                matches_target = target in target_group

            if matches_target:
                matched_filter = True
                selected_features.extend(features)

    if not matched_filter:
        selected_mask = np.ones(len(predictor_names), dtype=bool)
    else:
        selected_feature_set = set(selected_features)
        selected_mask = np.array(
            [name in selected_feature_set for name in predictor_names],
            dtype=bool
        )
        selected_mask |= entity_mask

    if not selected_mask.any():
        raise ValueError('Target feature filtering removed every encoded predictor.')

    return {
        'predictors': sparse if selected_mask.all() else sparse[:, selected_mask],
        'features': [name for name, keep in zip(predictor_names, selected_mask) if keep],
        'entity_mask': entity_mask[selected_mask],
        'context_mask': (~entity_mask)[selected_mask]
    }


def columns_to_csr(frame, columns):
    if not columns:
        return sp.csr_matrix((len(frame), len(columns)), dtype=np.float64)
    dense = frame.loc[:, columns].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    return sp.csr_matrix(dense, dtype=np.float64)


def entity_labels(ids, seasons, teams):
    ids = np.asarray(ids)
    seasons = np.asarray(seasons, dtype=str)
    teams = np.asarray(teams, dtype=str)
    return np.char.add(
        np.char.add(np.char.add(ids.astype(np.int64).astype(str), '_'), seasons),
        np.char.add('_', teams)
    )


def indicator_sparse(player_arr, seasons, teams):
    player_arr = np.asarray(player_arr)
    n_rows, n_slots = player_arr.shape
    flat_ids = player_arr.reshape(-1)
    valid = flat_ids != 0
    if not np.any(valid):
        return sp.csr_matrix((n_rows, 0), dtype=np.float64), np.array([], dtype=object)

    row_idx = np.repeat(np.arange(n_rows, dtype=np.int32), n_slots)[valid]
    season_idx = np.repeat(np.asarray(seasons, dtype=str), n_slots)[valid]
    team_idx = np.repeat(np.asarray(teams, dtype=str), n_slots)[valid]
    labels = entity_labels(flat_ids[valid], season_idx, team_idx)
    col_idx, classes = pd.factorize(labels, sort=True)
    values = np.ones(len(col_idx), dtype=np.float64)
    sparse = sp.csr_matrix((values, (row_idx, col_idx)), shape=(n_rows, len(classes)), dtype=np.float64)
    sparse.sum_duplicates()
    return sparse, np.asarray(classes, dtype=object)


def coach_sparse(coach_arr, seasons, teams, score_state_frame):
    coach_arr = np.asarray(coach_arr, dtype=str)
    seasons = np.asarray(seasons, dtype=str)
    teams = np.asarray(teams, dtype=str)
    n_rows = len(coach_arr)

    base_labels = np.char.add(
        np.char.add(np.char.add(coach_arr, '_'), seasons),
        np.char.add('_', teams)
    )
    row_parts = [np.arange(n_rows, dtype=np.int32)]
    label_parts = [base_labels]

    for state in ['tied', 'lead_1', 'lead_2', 'lead_3_plus', 'behind_1', 'behind_2', 'behind_3_plus']:
        active = score_state_frame[state].to_numpy(dtype=np.int8, copy=False) == 1
        if not np.any(active):
            continue
        state_labels = np.char.add(f'{state}_', base_labels[active])
        row_parts.append(np.flatnonzero(active).astype(np.int32))
        label_parts.append(state_labels)

    rows = np.concatenate(row_parts)
    labels = np.concatenate(label_parts)
    col_idx, classes = pd.factorize(labels, sort=True)
    values = np.ones(len(col_idx), dtype=np.float64)
    sparse = sp.csr_matrix((values, (rows, col_idx)), shape=(n_rows, len(classes)), dtype=np.float64)
    sparse.sum_duplicates()
    return sparse, np.asarray(classes, dtype=object)

def build_shift_map_targets(events, sbs):
    keys = sbs[['season', 'game_id', 'shift_id', 'is_home']].copy().reset_index(drop=True)
    keys['row_id'] = np.arange(len(keys))

    if events.empty:
        return {target: sp.csr_matrix((len(keys), MAP_SIZE), dtype=np.float32) for target in MAP_TARGETS}

    events = events.copy()
    events['is_home'] = (events['event_team_venue'] == 'home').astype(int)

    orientation = np.where(events['is_home'] == 1, 1.0, -1.0)
    x_oriented = orientation * events['x_adj'].to_numpy(dtype=np.float64)
    y_oriented = orientation * events['y_adj'].to_numpy(dtype=np.float64)

    x_idx = np.clip(np.rint(x_oriented).astype(int), 0, MAP_SHAPE[0] - 1)
    y_idx = np.clip(np.rint(y_oriented - MAP_Y[0]).astype(int), 0, MAP_SHAPE[1] - 1)

    events['coord_index'] = x_idx * MAP_SHAPE[1] + y_idx
    events['corsi'] = events['event_type'].isin(wsba.FENWICK_EVENTS + ['blocked-shot']).astype(np.float32)
    events['fenwick'] = events['event_type'].isin(wsba.FENWICK_EVENTS).astype(np.float32)
    events['shots'] = events['event_type'].isin(['shot-on-goal', 'goal']).astype(np.float32)
    events['goals'] = (events['event_type'] == 'goal').astype(np.float32)
    events['expected_goals'] = events['xG'].fillna(0).astype(np.float32)

    events = events.merge(keys, on=['season', 'game_id', 'shift_id', 'is_home'], how='inner')

    if events.empty:
        return {target: sp.csr_matrix((len(keys), MAP_SIZE), dtype=np.float32) for target in MAP_TARGETS}

    grouped = events.groupby(['row_id', 'coord_index'], as_index=False)[list(MAP_TARGETS)].sum()
    rows = grouped['row_id'].to_numpy()
    cols = grouped['coord_index'].to_numpy()

    responses = {}
    for target in MAP_TARGETS:
        values = grouped[target].to_numpy(dtype=np.float32)
        mask = values != 0
        responses[target] = sp.csr_matrix(
            (values[mask], (rows[mask], cols[mask])),
            shape=(len(keys), MAP_SIZE),
            dtype=np.float32
        )

    return responses

def penalized_system(sparse, weights, penalty):
    weights = np.asarray(weights, dtype=np.float64)
    weighted_sparse = sparse.multiply(weights[:, None])
    xtwx = (sparse.T @ weighted_sparse).tocsc()
    xw = np.asarray(sparse.T @ weights).ravel()
    return penalized_system_from_components(xtwx, xw, weights.sum(), penalty)


def normal_components(sparse, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weighted_sparse = sparse.multiply(weights[:, None])
    xtwx = (sparse.T @ weighted_sparse).tocsc()
    xw = np.asarray(sparse.T @ weights).ravel()
    return xtwx, xw, weights.sum()


def penalized_system_from_components(xtwx, xw, weight_sum, penalty):
    xtwx = xtwx.copy()
    diag = xtwx.diagonal()
    xtwx.setdiag(diag + penalty)

    return sp.bmat(
        [
            [sp.csc_matrix([[weight_sum]]), sp.csc_matrix(xw.reshape(1, -1))],
            [sp.csc_matrix(xw.reshape(-1, 1)), xtwx]
        ],
        format='csc'
    )


def normalize_workers(workers):
    if workers is None:
        return 1
    return max(int(workers), 1)


def solve_map_blocks(system, sparse, response, block_ranges, group_indices):
    solver = splu(system)
    solved = []
    for start, end in block_ranges:
        block = response[:, start:end]
        rhs = sp.vstack(
            [
                sp.csr_matrix(block.sum(axis=0)),
                sparse.T @ block,
            ],
            format='csc'
        )
        solution = solver.solve(rhs.toarray() * 3600.0)
        solved.append((
            start,
            end,
            {
                key: np.asarray(solution[np.where(idx < 0, 0, idx + 1), :], dtype=np.float32)
                for key, idx in group_indices.items()
                if len(idx) > 0
            }
        ))
    return solved


def fit_shift_map_model(sparse, weights, response, predictor_names, alpha, selected_masks, system=None, solver=None, solve_workers=1):
    weights = np.asarray(weights, dtype=np.float64)
    if system is None:
        system = penalized_system(sparse, weights, alpha)
    solve_workers = normalize_workers(solve_workers)
    if solver is None and solve_workers == 1:
        solver = splu(system)

    groups = {
        'intercept': {
            'idx': np.array([-1], dtype=int),
            'names': ['(intercept)'],
            'raw_maps': np.zeros((1, MAP_SIZE), dtype=np.float32)
        }
    }
    for key, mask in selected_masks.items():
        selected_idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        groups[key] = {
            'idx': selected_idx,
            'names': [predictor_names[i] for i in selected_idx],
            'raw_maps': np.zeros((len(selected_idx), MAP_SIZE), dtype=np.float32)
        }

    block_ranges = [(start, min(start + MAP_BLOCK_SIZE, response.shape[1])) for start in range(0, response.shape[1], MAP_BLOCK_SIZE)]
    group_indices = {key: group['idx'] for key, group in groups.items()}
    if solver is not None:
        solved = []
        for start, end in block_ranges:
            block = response[:, start:end]
            rhs = sp.vstack(
                [
                    sp.csr_matrix(block.sum(axis=0)),
                    sparse.T @ block,
                ],
                format='csc'
            )
            solution = solver.solve(rhs.toarray() * 3600.0)
            solved.append((
                start,
                end,
                {
                    key: np.asarray(solution[np.where(idx < 0, 0, idx + 1), :], dtype=np.float32)
                    for key, idx in group_indices.items()
                    if len(idx) > 0
                }
            ))
        solved_groups = [solved]
    elif solve_workers == 1:
        solved_groups = [solve_map_blocks(system, sparse, response, block_ranges, group_indices)] if solver is None else []
    else:
        chunks = [chunk.tolist() for chunk in np.array_split(np.asarray(block_ranges, dtype=object), solve_workers) if len(chunk) > 0]
        with ThreadPoolExecutor(max_workers=solve_workers) as executor:
            solved_groups = list(
                executor.map(
                    lambda chunk: solve_map_blocks(system, sparse, response, chunk, group_indices),
                    chunks
                )
            )

    for solved in solved_groups:
        for start, end, values_by_group in solved:
            for key, values in values_by_group.items():
                groups[key]['raw_maps'][:, start:end] = values

    results = {}
    for key, group in groups.items():
        scalar_values = group['raw_maps'].sum(axis=1).astype(float).tolist()
        results[key] = (group['names'], scalar_values, encode_raw_maps(group['raw_maps']))

    return results

def interval_names(source, dest):
    return {
        source: dest,
        f'{source}_standard_error': f'{dest}_standard_error',
        f'{source}_margin_of_error': f'{dest}_margin_of_error',
        f'{source}_p_value': f'{dest}_p_value',
        f'{source}_ci_lower': f'{dest}_ci_lower',
        f'{source}_ci_upper': f'{dest}_ci_upper'
    }

def coefficient_p_values(values, standard_errors):
    values = np.asarray(values, dtype=np.float64)
    standard_errors = np.asarray(standard_errors, dtype=np.float64)
    z_scores = np.divide(
        values,
        standard_errors,
        out=np.full_like(values, np.nan, dtype=np.float64),
        where=standard_errors > 0
    )
    return (2.0 * norm.sf(np.abs(z_scores))).tolist()


def coef_vector(model, shot):
    intercept = float(model.intercept_[0]) if shot else float(model.intercept_)
    coefs = model.coef_.ravel() if shot else np.asarray(model.coef_, dtype=np.float64)
    return np.concatenate(([intercept], np.asarray(coefs, dtype=np.float64)))


class _SparseRidgeModel:
    def __init__(self, coef_vector):
        coef_vector = np.asarray(coef_vector, dtype=np.float64)
        self.intercept_ = float(coef_vector[0])
        self.coef_ = coef_vector[1:]

    def predict(self, sparse):
        return np.asarray(sparse @ self.coef_, dtype=np.float64) + self.intercept_


def shot_estimator(optimal, target_params):
    return LogisticRegression(
        C=optimal,
        penalty='l2',
        fit_intercept=True,
        max_iter=target_params['max_iter'],
        random_state=42
    )


def fit_sparse_ridge(sparse, target, sample_weight, alpha, factorize=True, tol=1e-8, maxiter=None, system_components=None):
    weights = np.ones(sparse.shape[0], dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if system_components is None:
        system = penalized_system(sparse, weights, alpha)
    else:
        system = penalized_system_from_components(*system_components, alpha)
    weighted_target = weights * target
    rhs = np.concatenate((
        [weighted_target.sum()],
        np.asarray(sparse.T @ weighted_target).ravel()
    ))

    if factorize:
        solver = splu(system)
        coefficients = np.asarray(solver.solve(rhs), dtype=np.float64)
    else:
        coefficients, info = cg(system, rhs, rtol=tol, atol=0.0, maxiter=maxiter)
        solver = None
        if info != 0:
            solver = splu(system)
            coefficients = np.asarray(solver.solve(rhs), dtype=np.float64)

    return _SparseRidgeModel(coefficients), system, solver


def inverse_diag_blocks(system, indices, block_size=512):
    solver = splu(system)
    diag = np.zeros(len(indices), dtype=np.float64)
    n = system.shape[0]
    for start in range(0, len(indices), block_size):
        end = min(start + block_size, len(indices))
        cols = indices[start:end]
        rhs = np.zeros((n, len(cols)), dtype=np.float64)
        rhs[cols, np.arange(len(cols), dtype=int)] = 1.0
        solution = solver.solve(rhs)
        diag[start:end] = np.asarray(solution)[cols, np.arange(len(cols), dtype=int)]
    return diag


def inverse_diag(system, indices=None, block_size=512, solver=None, solve_workers=1):
    solve_workers = normalize_workers(solve_workers)
    n = system.shape[0]
    indices = np.arange(n, dtype=int) if indices is None else np.asarray(indices, dtype=int)

    if solver is not None:
        diag = np.zeros(len(indices), dtype=np.float64)
        for start in range(0, len(indices), block_size):
            end = min(start + block_size, len(indices))
            cols = indices[start:end]
            rhs = np.zeros((n, len(cols)), dtype=np.float64)
            rhs[cols, np.arange(len(cols), dtype=int)] = 1.0
            solution = solver.solve(rhs)
            diag[start:end] = np.asarray(solution)[cols, np.arange(len(cols), dtype=int)]
        return diag

    if solve_workers > 1:
        chunks = [chunk for chunk in np.array_split(indices, solve_workers) if len(chunk) > 0]
        with ThreadPoolExecutor(max_workers=solve_workers) as executor:
            return np.concatenate(list(executor.map(lambda chunk: inverse_diag_blocks(system, chunk, block_size), chunks)))

    solver = splu(system)
    diag = np.zeros(len(indices), dtype=np.float64)

    for start in range(0, len(indices), block_size):
        end = min(start + block_size, len(indices))
        cols = indices[start:end]
        rhs = np.zeros((n, len(cols)), dtype=np.float64)
        rhs[cols, np.arange(len(cols), dtype=int)] = 1.0
        solution = solver.solve(rhs)
        diag[start:end] = np.asarray(solution)[cols, np.arange(len(cols), dtype=int)]

    return diag


def residual_coefficient_intervals(
    point_estimates,
    model,
    sparse,
    target,
    sample_weight,
    shot,
    optimal,
    system=None,
    solver=None,
    solve_workers=1
):
    point_estimates = np.asarray(point_estimates, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rows, columns = sparse.shape
    dof = max(rows - columns - 1, 1)

    if shot:
        prediction = model.predict_proba(sparse)[:, 1]
        residuals = target - prediction
        squared_error_sum = float(np.sum(residuals ** 2))
        hessian_weights = prediction * (1.0 - prediction)
        penalty = 1.0 / optimal
        if system is None:
            system = penalized_system(sparse, hessian_weights, penalty)
        if solver is None:
            solver = splu(system)
    else:
        prediction = model.predict(sparse)
        residuals = target - prediction
        if sample_weight is None:
            squared_error_sum = float(np.sum(residuals ** 2))
            weights = np.ones(rows, dtype=np.float64)
        else:
            weights = np.asarray(sample_weight, dtype=np.float64)
            squared_error_sum = float(np.sum(weights * residuals ** 2))
        if system is None:
            system = penalized_system(sparse, weights, optimal)
        if solver is None:
            solver = splu(system)

    diag_values = np.maximum(inverse_diag(system, solver=solver, solve_workers=solve_workers), 0.0)
    standard_errors = np.sqrt((squared_error_sum / dof) * diag_values)
    margin_of_errors = 1.96 * standard_errors
    p_values = np.asarray(coefficient_p_values(point_estimates, standard_errors), dtype=np.float64)
    ci_lower = point_estimates - margin_of_errors
    ci_upper = point_estimates + margin_of_errors

    return standard_errors, margin_of_errors, p_values, ci_lower, ci_upper


def model_fit_metrics(model, sparse, target, sample_weight, shot, regularization):
    target = np.asarray(target, dtype=np.float64)
    weights = np.ones(len(target), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    prediction = model.predict_proba(sparse)[:, 1] if shot else model.predict(sparse)
    prediction = np.asarray(prediction, dtype=np.float64)
    residuals = target - prediction
    weight_sum = float(np.sum(weights))
    target_mean = float(np.average(target, weights=weights)) if weight_sum > 0 else np.nan
    prediction_mean = float(np.average(prediction, weights=weights)) if weight_sum > 0 else np.nan
    sse = float(np.sum(weights * residuals ** 2))
    baseline_sse = float(np.sum(weights * (target - target_mean) ** 2)) if np.isfinite(target_mean) else np.nan
    mse = sse / weight_sum if weight_sum > 0 else np.nan
    mae = float(np.sum(weights * np.abs(residuals)) / weight_sum) if weight_sum > 0 else np.nan
    r2 = 1.0 - (sse / baseline_sse) if baseline_sse and baseline_sse > 0 else np.nan

    metrics = {
        "model_n_obs": int(len(target)),
        "model_n_predictors": int(sparse.shape[1]),
        "model_regularization": float(regularization),
        "model_weight_sum": weight_sum,
        "model_target_mean": target_mean,
        "model_prediction_mean": prediction_mean,
        "model_sse": sse,
        "model_mse": mse,
        "model_rmse": float(np.sqrt(mse)) if np.isfinite(mse) else np.nan,
        "model_mae": mae,
        "model_r2": float(r2) if np.isfinite(r2) else np.nan,
    }

    if shot:
        clipped = np.clip(prediction, 1e-15, 1.0 - 1e-15)
        metrics.update({
            "model_log_loss": float(-np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))),
            "model_brier_score": float(np.mean((target - prediction) ** 2)),
            "model_accuracy": float(np.mean((prediction >= 0.5) == target)),
        })

    return metrics


def scale_shot_columns(frame, scale):
    scaled = frame.copy()
    for col in scaled.columns:
        if not str(col).startswith('rapm_') or str(col).endswith('_p_value'):
            continue
        scaled[col] = scaled[col] * scale
    return scaled

### RAPM FUNCTIONS ###
# Provided in this file are functions vital to the Regularized Adjusted Plus-Minus model in the WSBA Hockey package #
# DISCLAIMER: The datasets generated by this module are incredibly large and may significantly hinder the performance of the machine executing the code #

def game_info(pbp):
    #Add game context (such as whether a team is on the second half of a back to back)
    pbp['game_date'] = pd.to_datetime(pbp['game_date'])
    
    teams = pbp.groupby('game_date').agg(home_team_yesterday=('home_team_abbr', 'first'),
                                            away_team_yesterday=('away_team_abbr', 'first')).reset_index()
    teams['game_date'] = teams['game_date'] + pd.Timedelta(days=1)
    pbp = pbp.merge(teams, on='game_date', how='left')

    #Assign back to back flags
    pbp['home_home_yesterday'] = (pbp['home_team_abbr'] == pbp['home_team_yesterday']).astype(int)
    pbp['home_away_yesterday'] = (pbp['home_team_abbr'] == pbp['away_team_yesterday']).astype(int)
    pbp['away_home_yesterday'] = (pbp['away_team_abbr'] == pbp['home_team_yesterday']).astype(int)
    pbp['away_away_yesterday'] = (pbp['away_team_abbr'] == pbp['away_team_yesterday']).astype(int)

    pbp['home_back_to_back'] = ((pbp['home_home_yesterday'] == 1) | (pbp['home_away_yesterday'] == 1)).astype(int)
    pbp['away_back_to_back'] = ((pbp['away_home_yesterday'] == 1) | (pbp['away_away_yesterday'] == 1)).astype(int)
    
    return pbp

def label_shifts(pbp):
    #Label shifts in the pbp dataframe
    #A new shift is logged when any of the on ice players changes

    home_cols = [f'home_on_{i}_id' for i in range(1, 7)]
    away_cols = [f'away_on_{i}_id' for i in range(1, 7)]

    pbp = pbp.sort_values(['game_id', 'period', 'seconds_elapsed', 'event_index'])

    # Recalculate score state to be the score at the time of the event rather than after the event
    for venue in ['away','home']:
        pbp[f'{venue}_score'] = ((pbp['event_team_venue']==venue)&(pbp['event_type']=='goal')).groupby(pbp['game_id']).cumsum().shift(1)

    # Create lineup key
    lineup_arr = pbp[home_cols + away_cols].astype(int).astype(str).to_numpy()
    lineup_key = np.array(['|'.join(row) for row in lineup_arr], dtype=object)
    
    # Pre-change detection
    non_change_mask = pbp['event_type'] != 'change'
    non_change_lineup = lineup_key[non_change_mask]
    non_change_rows = (
        pbp.loc[non_change_mask, ['game_id', 'period', 'seconds_elapsed']]
        .to_numpy()
    )
    
    # Vectorized match_flag
    lookup = set((g, p, s, l) for (g, p, s), l in zip(non_change_rows, non_change_lineup))
    match_flag = np.array([((g, p, s, l) in lookup) for g, p, s, l in zip(
        pbp['game_id'], pbp['period'], pbp['seconds_elapsed'], lineup_key
    )], dtype=int)

    is_pre_change = ((pbp['event_type'] == 'change') & (match_flag == 1)).astype(int)

    pbp['event_priority'] = 1
    pbp.loc[(pbp['event_type'] == 'change') & (is_pre_change == 1), 'event_priority'] = 0
    pbp.loc[(pbp['event_type'] == 'change') & (is_pre_change == 0), 'event_priority'] = 2
    pbp.loc[pbp['event_type'] == 'period-start', 'event_priority'] = 3
    pbp.loc[pbp['event_type'] == 'faceoff', 'event_priority'] = 4

    pbp = pbp.sort_values(
        ['game_id', 'period', 'seconds_elapsed', 'event_priority', 'event_index']
    )

    # Detect shifts
    home_sorted = np.sort(pbp[home_cols].to_numpy(), axis=1)
    away_sorted = np.sort(pbp[away_cols].to_numpy(), axis=1)

    same_home = (home_sorted == np.roll(home_sorted, 1, axis=0)).all(axis=1)
    same_away = (away_sorted == np.roll(away_sorted, 1, axis=0)).all(axis=1)

    same_goalie_home = pbp['home_goalie_id'].to_numpy() == np.roll(pbp['home_goalie_id'].to_numpy(), 1)
    same_goalie_away = pbp['away_goalie_id'].to_numpy() == np.roll(pbp['away_goalie_id'].to_numpy(), 1)

    same_as_prev = same_home & same_away & same_goalie_home & same_goalie_away

    pbp['shift_change'] = (~same_as_prev).astype(int)
    pbp['shift_id'] = pbp['shift_change'].cumsum()

    pbp = pbp.drop(columns=['event_priority'])

    return pbp

def orient_perspective(df, strengths, home=True):
    #Orient the perspective of a shift-by-shift dataframe from home and away to offense and defense
    df = df.copy()
    
    if home:
        for_team = 'home'
        against_team = 'away'
       
    else:
        for_team = 'away'
        against_team = 'home'
    
    offense_cols = [f'{for_team}_on_{i}_id' for i in range(1,7)]
    defense_cols = [f'{against_team}_on_{i}_id' for i in range(1,7)]

    for stat in RAPM_STATS:
        df[f'{stat}_for'] = df[f'{for_team}_{stat}']

    df['offense_team_abbr'] = df[f'{for_team}_team_abbr']
    df['defense_team_abbr'] = df[f'{against_team}_team_abbr']

    df['offense_goalie_id'] = df[f'{for_team}_goalie_id']
    df['defense_goalie_id'] = df[f'{against_team}_goalie_id']

    df['offense_coach'] = df[f'{for_team}_coach']
    df['defense_coach'] = df[f'{against_team}_coach']

    df['offensive_faceoff'] = df[f'{for_team}_offensive_zone_faceoff']
    df['defensive_faceoff'] = df[f'{against_team}_offensive_zone_faceoff']
    df['neutral_faceoff'] = df['neutral_zone_faceoff']
    df['on_the_fly'] = ((df['offensive_faceoff'] + df['defensive_faceoff'] + df['neutral_faceoff']) == 0).astype(int)

    for lead in ['1', '2', '3_plus']:
        df[f'lead_{lead}'] = df[f'{for_team}_lead_{lead}']
        df[f'behind_{lead}'] = df[f'{against_team}_lead_{lead}']

    for strength in strengths:
        #Strength state is relative to offensive team
        df[f'strength_{strength}'] = df[f'{for_team}_strength_{strength}']
    
    strength_cols = [f'strength_{s}' for s in strengths]

    df['offense_back_to_back'] = df[f'{for_team}_back_to_back']
    df['defense_back_to_back'] = df[f'{against_team}_back_to_back']
    df['power_play_expiry'] = df[f'{for_team}_power_play_expiry']
    df['penalty_kill_expiry'] = df[f'{against_team}_power_play_expiry']
    df['last_penalty'] = df[f'{for_team}_last_penalty']

    rename_dict = {}
    for i, col in enumerate(offense_cols):
        rename_dict.update({col: f'offense_on_{i+1}_id'})
    for i, col in enumerate(defense_cols):
        rename_dict.update({col: f'defense_on_{i+1}_id'})

    df = df.rename(columns=rename_dict)

    for stat in RAPM_STATS:
        df[f'{stat}_for_per_sixty'] = (df[f'{for_team}_{stat}']*3600)/df['shift_length']

    df['is_home'] = int(home)

    cols = shift_cols()

    return df[cols + strength_cols]

def encode_players(df):
    print('Encoding on-ice players to columns...')

    offense_cols = [f'offense_on_{i}_id' for i in range(1,7)]
    defense_cols = [f'defense_on_{i}_id' for i in range(1,7)]

    #Player grouping information
    offense_arr = df[offense_cols].values
    defense_arr = df[defense_cols].values
    season_arr = df['season'].astype(str).values
    offense_team_arr = df['offense_team_abbr'].astype(str).values
    defense_team_arr = df['defense_team_abbr'].astype(str).values

    offense_sparse, offense_classes = indicator_sparse(offense_arr, season_arr, offense_team_arr)
    defense_sparse, defense_classes = indicator_sparse(defense_arr, season_arr, defense_team_arr)
    offense_coach_sparse, offense_coach_classes = coach_sparse(df['offense_coach'].astype(str).to_numpy(), season_arr, offense_team_arr, df)
    defense_coach_sparse, defense_coach_classes = coach_sparse(df['defense_coach'].astype(str).to_numpy(), season_arr, defense_team_arr, df)
    
    #Trim input dataframe to sparse and combine with encoding sparse
    excluded_cols = (
        offense_cols + defense_cols + 
        ['season', 'game_id', 'shift_id', 'score_state','zone_code',
        'tied',
        'offense_goalie_id', 'defense_goalie_id',
        'offense_coach', 'defense_coach',
        'offense_team_abbr', 'defense_team_abbr'] +
        [f'{t}_for{m}' for t in RAPM_STATS for m in ['', '_per_sixty']]
    )
        
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    
    features_sparse = columns_to_csr(df, feature_cols)

    X_sparse = sp.hstack([
        features_sparse,
        offense_sparse,
        defense_sparse,
        offense_coach_sparse,
        defense_coach_sparse
    ], format='csr')

    offense_cols_encoded = [f'offense_{col}' for col in offense_classes]
    defense_cols_encoded = [f'defense_{col}' for col in defense_classes]
    offense_coach_cols_encoded = [f'offense_coach_{col}' for col in offense_coach_classes]
    defense_coach_cols_encoded = [f'defense_coach_{col}' for col in defense_coach_classes]

    features = (
        feature_cols +
        offense_cols_encoded +
        defense_cols_encoded +
        offense_coach_cols_encoded +
        defense_coach_cols_encoded
    )
    
    #Return the testing/training sparse and infos
    return {
        'predictors': X_sparse,
        'features': features,
        'offense_classes': offense_classes,
        'defense_classes': defense_classes,
        'offense_coach_classes': offense_coach_classes,
        'defense_coach_classes': defense_coach_classes
    }

def encode_shooters(df):
    print('Encoding shooters, goalies, and passers to columns...')

    passer_cols = [f'passer_{i}_id' for i in range(1,7)]
    primary_passer_prob_cols = [f'passer_{i}_primary_fenwick_assist_probability' for i in range(1,7)]
    secondary_passer_prob_cols = [f'passer_{i}_secondary_fenwick_assist_probability' for i in range(1,7)]

    # Player grouping information
    shooter_arr = df['shooter_id'].values
    goalie_arr = df['goalie_id'].values
    passer_arr = df[passer_cols].values
    primary_passer_prob_arr = df[primary_passer_prob_cols].fillna(0).values
    secondary_passer_prob_arr = df[secondary_passer_prob_cols].fillna(0).values
    
    season_arr = df['season'].astype(str).values
    shooter_team_arr = df['shooter_team_abbr'].astype(str).values
    goalie_team_arr = df['goalie_team_abbr'].astype(str).values

    n_rows = len(df)

    def build_weighted_passer_sparse(prob_arr):
        prob_arr = np.asarray(prob_arr, dtype=np.float64)
        flat_ids = passer_arr.reshape(-1)
        flat_weights = prob_arr.reshape(-1)
        valid = (flat_ids != 0) & (flat_weights > 0)
        if not np.any(valid):
            return sp.csr_matrix((n_rows, 0), dtype=np.float64), np.array([], dtype=object)

        n_slots = passer_arr.shape[1]
        row_idx = np.repeat(np.arange(n_rows, dtype=np.int32), n_slots)[valid]
        season_idx = np.repeat(season_arr, n_slots)[valid]
        team_idx = np.repeat(shooter_team_arr, n_slots)[valid]
        labels = entity_labels(flat_ids[valid], season_idx, team_idx)
        col_idx, classes = pd.factorize(labels, sort=True)
        sparse = sp.csr_matrix((flat_weights[valid], (row_idx, col_idx)), shape=(n_rows, len(classes)), dtype=np.float64)
        sparse.sum_duplicates()
        return sparse, np.asarray(classes, dtype=object)

    shooter_sparse, shooter_classes = indicator_sparse(shooter_arr.reshape(-1, 1), season_arr, shooter_team_arr)
    goalie_sparse, goalie_classes = indicator_sparse(goalie_arr.reshape(-1, 1), season_arr, goalie_team_arr)
    primary_passer_sparse, primary_passer_classes = build_weighted_passer_sparse(primary_passer_prob_arr)
    secondary_passer_sparse, secondary_passer_classes = build_weighted_passer_sparse(secondary_passer_prob_arr)
    
    # Trim input dataframe to sparse and combine with encoding sparse
    feature_cols = ['expected_goals']
    
    features_sparse = columns_to_csr(df, feature_cols)

    X_sparse = sp.hstack([features_sparse, shooter_sparse, goalie_sparse, 
                          primary_passer_sparse, secondary_passer_sparse], format='csr')

    shooter_cols_encoded = [f'shooter_{col}' for col in shooter_classes]
    goalie_cols_encoded = [f'goalie_{col}' for col in goalie_classes]
    primary_passer_cols_encoded = [f'primary_passer_{col}' for col in primary_passer_classes]
    secondary_passer_cols_encoded = [f'secondary_passer_{col}' for col in secondary_passer_classes]
    
    features = (feature_cols + shooter_cols_encoded + goalie_cols_encoded + 
                primary_passer_cols_encoded + secondary_passer_cols_encoded)
    
    # Return the testing/training sparse and infos
    return {
        'predictors': X_sparse,
        'features': features,
        'shooter_classes': shooter_classes,
        'goalie_classes': goalie_classes,
        'primary_passer_classes': primary_passer_classes,
        'secondary_passer_classes': secondary_passer_classes
    }

def prepare_rapm_data(pbp, strengths=wsba.STRENGTHS, shot=False, include_maps=True):
    #Prepare RAPM data by transforming the play-by-play dataframe to a shift-by-shift dataframe
    print('Preparing RAPM data...')

    if shot:
        #Individual Talent Isolation
        if isinstance(strengths, str):
            strengths = [strengths]
            
        pbp = pbp.loc[pbp['strength_state'].isin(strengths)]

        #Add individual player columns
        shooter_col = 'event_player_1_id'
        goalie_col = 'event_goalie_id'

        #Fill Nan player columns
        pbp[[shooter_col,goalie_col]] = pbp[[shooter_col,goalie_col]].fillna(0)

        pbp['shooter_id'] = pbp[shooter_col]
        pbp['goalie_id'] = pbp[goalie_col]
        pbp['shooter_team_abbr'] = pbp['event_team_abbr']
        pbp['goalie_team_abbr'] = np.where(
            pbp['event_team_abbr']==pbp['home_team_abbr'],
            pbp['away_team_abbr'],
            pbp['home_team_abbr']
        )
        if 'is_home' not in pbp.columns:
            pbp['is_home'] = pbp['event_team_abbr'].eq(pbp['home_team_abbr'])
        if 'is_goal' not in pbp.columns:
            pbp['is_goal'] = pbp['event_type'].eq('goal').astype(int)

        #Collect and assign event passers
        home_cols = [f'home_on_{i}_id' for i in range(1,7)]
        away_cols = [f'away_on_{i}_id' for i in range(1,7)]
        home_primary_probs = [f'home_on_{i}_primary_fenwick_assist_probability' for i in range(1,7)]
        away_primary_probs = [f'away_on_{i}_primary_fenwick_assist_probability' for i in range(1,7)]
        home_secondary_probs = [f'home_on_{i}_secondary_fenwick_assist_probability' for i in range(1,7)]
        away_secondary_probs = [f'away_on_{i}_secondary_fenwick_assist_probability' for i in range(1,7)]
        passer_probability_cols = home_primary_probs + away_primary_probs + home_secondary_probs + away_secondary_probs
        missing_probability_cols = [col for col in passer_probability_cols if col not in pbp.columns]
        if missing_probability_cols:
            position_cols = [f'{venue}_on_{i}_pos' for venue in ['home', 'away'] for i in range(1, 7)]
            required_cols = position_cols + ['event_player_2_id', 'event_player_3_id']
            missing_required_cols = [col for col in required_cols if col not in pbp.columns]
            if missing_required_cols:
                raise KeyError(
                    "Shot RAPM requires passer assist probabilities or the columns needed to impute them. "
                    f"Missing columns: {missing_required_cols}"
                )
            from wsba_ml.xg_model import apply_passing_imputation
            pbp = apply_passing_imputation(pbp)

        pbp[home_cols + away_cols + passer_probability_cols] = pbp[home_cols + away_cols + passer_probability_cols].fillna(0)

        for i in range(6):
            pbp[f'passer_{i+1}_id'] = np.where(
                pbp['is_home'],
                pbp[home_cols[i]],
                pbp[away_cols[i]]
            )

            pbp[f'passer_{i+1}_primary_fenwick_assist_probability'] = np.where(
                pbp['is_home'],
                pbp[home_primary_probs[i]],
                pbp[away_primary_probs[i]]
            )

            pbp[f'passer_{i+1}_secondary_fenwick_assist_probability'] = np.where(
                pbp['is_home'],
                pbp[home_secondary_probs[i]],
                pbp[away_secondary_probs[i]]
            )

        # Column lists for downstream encoding
        passer_cols = [f'passer_{i}_id' for i in range(1, 7)]
        primary_passer_probs   = [f'passer_{i}_primary_fenwick_assist_probability' for i in range(1, 7)]
        secondary_passer_probs = [f'passer_{i}_secondary_fenwick_assist_probability' for i in range(1, 7)]

        #Use number of fenwick shots player takes in sample as weight
        pbp['shooter_total_fenwick'] = (
            pbp
            .groupby('shooter_id')['shooter_id']
            .transform('count')
        )

        # Trim dataframe to only the columns needed for shot RAPM
        feature_cols = [col for col in wsba.CONTINUOUS + wsba.BOOLEAN if col in pbp.columns]
        pbp = pbp[
            feature_cols
            + [
                'season',
                'shooter_id',
                'goalie_id',
                'shooter_team_abbr',
                'goalie_team_abbr',
                'shooter_total_fenwick',
                'event_type',
            ]
            + passer_cols
            + primary_passer_probs
            + secondary_passer_probs
            + ['xG', 'is_goal']
        ].query("xG > 0").fillna(0)

        return pbp.rename(columns={'xG': 'expected_goals', 'is_goal': 'goals'})
    
    else:
        #On-Ice Impact Isolation
        #A shift in this context refers to a period of a time no line changes occur for any team
        #Each skater will have its own column in the dataframe (using its player_id)
        player_cols = [f'home_on_{i}_id' for i in range(1, 7)] + [f'away_on_{i}_id' for i in range(1, 7)]
        goalie_cols = ['away_goalie_id', 'home_goalie_id']
        coach_cols = ['away_coach', 'home_coach']

        #Remove shootouts and playoff games
        pbp = pbp.loc[(pbp['period_type']!='SO')&(pbp['season_type']==2)]

        #Fill NaN player columns
        pbp[player_cols+goalie_cols+coach_cols] = pbp[player_cols+goalie_cols+coach_cols].fillna(0)

        #Add game-related context
        pbp = game_info(pbp)

        #Create shift ids
        pbp = label_shifts(pbp)

        # Recalibrate score state
        pbp['score_state'] = np.where(
            pbp['event_team_venue']=='home',
            pbp['home_score']-pbp['away_score'],
            pbp['away_score']-pbp['home_score']
        )

        # Determine if a power-play expired during the shift for each team
        pbp['home_power_play_expiry'] = ((pbp['strength_state'].isin(wsba.STRENGTH_GROUPS['even_strength'])) &
                                (pbp['strength_state_venue'].str[::-1].shift().isin(wsba.STRENGTH_GROUPS['power_play'])) &
                                (pbp['shift_change'] == 1)).astype(int).fillna(0)
        
        pbp['away_power_play_expiry'] = ((pbp['strength_state'].isin(wsba.STRENGTH_GROUPS['even_strength'])) &
                                (pbp['strength_state_venue'].shift().isin(wsba.STRENGTH_GROUPS['power_play'])) &
                                (pbp['shift_change'] == 1)).astype(int).fillna(0)

        # Determine the last team to have taken a penalty
        pbp['home_pen_count'] = (
            ((pbp['event_type'] == 'penalty') & (pbp['event_team_venue'] == 'home'))
            .groupby([pbp['game_id'], pbp['period'], pbp['seconds_elapsed']])
            .transform('sum')
        )

        pbp['away_pen_count'] = (
            ((pbp['event_type'] == 'penalty') & (pbp['event_team_venue'] == 'away'))
            .groupby([pbp['game_id'], pbp['period'], pbp['seconds_elapsed']])
            .transform('sum')
        )

        pbp['home_last_penalty'] = (
            (pbp['event_type'] == 'penalty') &
            (pbp['event_team_venue'] == 'home') &
            (pbp['home_pen_count'] > pbp['away_pen_count'])
        ).astype(int)

        pbp['away_last_penalty'] = (
            (pbp['event_type'] == 'penalty') &
            (pbp['event_team_venue'] == 'away') &
            (pbp['away_pen_count'] > pbp['home_pen_count'])
        ).astype(int)

        #Determine if shift started with a faceoff
        pbp['home_zone_code'] = np.where(pbp['event_team_abbr'] == pbp['home_team_abbr'], pbp['zone_code'], np.nan)
        pbp['home_zone_code'] = np.where((pbp['event_team_abbr'] == pbp['away_team_abbr']) & (pbp['zone_code'] == 'O'), 'D', pbp['home_zone_code'])
        pbp['home_zone_code'] = np.where((pbp['event_team_abbr'] == pbp['away_team_abbr']) & (pbp['zone_code'] == 'D'), 'O', pbp['home_zone_code'])
        pbp['home_zone_code'] = np.where(pbp['zone_code'] == 'N', 'N', pbp['home_zone_code'])
        pbp['shift_start_period'] = pbp.groupby('shift_id')['period'].transform('first')
        pbp['shift_start_seconds'] = pbp.groupby('shift_id')['seconds_elapsed'].transform('first')
        pbp['shift_start_faceoff'] = (
            (pbp['event_type'] == 'faceoff') &
            (pbp['period'] == pbp['shift_start_period']) &
            (pbp['seconds_elapsed'] == pbp['shift_start_seconds'])
        )

        pbp['home_offensive_zone_faceoff'] = (pbp['shift_start_faceoff'] & (pbp['home_zone_code'] == 'O')).astype(int)
        pbp['away_offensive_zone_faceoff'] = (pbp['shift_start_faceoff'] & (pbp['home_zone_code'] == 'D')).astype(int)
        pbp['neutral_zone_faceoff'] = (pbp['shift_start_faceoff'] & (pbp['home_zone_code'] == 'N')).astype(int)

        #Venue stats
        for venue in ['home', 'away']:
            pbp[f'{venue}_corsi'] = ((pbp['event_team_venue'] == venue)*(pbp['event_type'].isin(wsba.FENWICK_EVENTS+['blocked-shot']))).astype(int)
            pbp[f'{venue}_fenwick'] = ((pbp['event_team_venue'] == venue)*(pbp['event_type'].isin(wsba.FENWICK_EVENTS))).astype(int)
            pbp[f'{venue}_shots'] = ((pbp['event_team_venue'] == venue)*(pbp['event_type'].isin(['shot-on-goal', 'goal']))).astype(int)
            pbp[f'{venue}_goals'] = ((pbp['event_team_venue'] == venue)*(pbp['event_type']=='goal')).astype(int)
            pbp[f'{venue}_expected_goals'] = (pbp['event_team_venue'] == venue)*pbp['xG']
            pbp[f'{venue}_penalty_minutes'] = (pbp['event_team_venue'] == venue)*(pbp['penalty_type']!='fighting')*(pbp['penalty_duration'])
        
        #Score stat indicators
        pbp['tied'] = (pbp['home_score'] == pbp['away_score']).astype(int)
        pbp['home_lead_1'] = (pbp['home_score'] - pbp['away_score'] == 1).astype(int)
        pbp['home_lead_2'] = (pbp['home_score'] - pbp['away_score'] == 2).astype(int)
        pbp['home_lead_3_plus'] = (pbp['home_score'] - pbp['away_score'] >= 3).astype(int)
        pbp['away_lead_1'] = (pbp['home_score'] - pbp['away_score'] == -1).astype(int)
        pbp['away_lead_2'] = (pbp['home_score'] - pbp['away_score'] == -2).astype(int)
        pbp['away_lead_3_plus'] = (pbp['home_score'] - pbp['away_score'] <= -3).astype(int)

        #Mark if the game is scoreless or not
        pbp['scoreless'] = ((pbp['home_score'] + pbp['away_score']) == 0).astype(int)

        #Strength state indicators
        for venue in ['home', 'away']:
            for strength in strengths:
                pbp[f'{venue}_strength_{strength}'] = ((pbp['strength_state_venue'].str[::-1] if venue == 'home' else pbp['strength_state_venue'])==strength).astype(int)

        pbp['period_2'] = (pbp['period'] == 2).astype(int)
        pbp['period_3'] = (pbp['period'] == 3).astype(int)
        pbp['period_4_plus'] = (pbp['period'] >= 4).astype(int)

        #Move down to provided strength states for both teams
        strength_mask_cols = [f'home_strength_{s}' for s in strengths] + [f'away_strength_{s}' for s in strengths]
        pbp = pbp.loc[pbp[strength_mask_cols].eq(1).any(axis=1)]

        if include_maps:
            map_events = pbp.loc[
                (
                    pbp['event_type'].isin(wsba.FENWICK_EVENTS + ['blocked-shot'])
                    & pbp['event_team_venue'].isin(['home', 'away'])
                    & pbp['x_adj'].notna()
                    & pbp['y_adj'].notna()
                ),
                ['season', 'game_id', 'shift_id', 'event_team_venue', 'event_type', 'x_adj', 'y_adj', 'xG']
            ].copy()

        #Move from event-based rows to shift-based rows
        shifts = pbp.groupby(['shift_id']).agg(
            {   
                'event_length': 'sum',
                'home_back_to_back': 'max',
                'away_back_to_back': 'max',
                'home_offensive_zone_faceoff': 'max',
                'away_offensive_zone_faceoff': 'max',
                'neutral_zone_faceoff': 'max',
                'home_lead_1': 'max',
                'home_lead_2': 'max',
                'home_lead_3_plus': 'max',
                'away_lead_1': 'max',
                'away_lead_2': 'max',
                'away_lead_3_plus': 'max',
                'tied': 'max',
                'scoreless': 'max',
                'period_2': 'max',
                'period_3': 'max',
                'period_4_plus': 'max',
                'home_power_play_expiry': 'max',
                'away_power_play_expiry': 'max',
                'home_last_penalty': 'max',
                'away_last_penalty': 'max',
                

                **{f'{v}_{s}': 'sum' for v in ['home', 'away'] for s in RAPM_STATS},
                **{
                    col: 'first' for col in (
                    player_cols + [
                        'season',
                        'game_id',
                        'period',
                        'home_team_abbr',
                        'away_team_abbr',
                        'score_state',
                        'strength_state',
                        'home_goalie_id',
                        'away_goalie_id',
                        'home_coach',
                        'away_coach'
                    ]
                )}

            } |
            {f'{v}_strength_{s}': 'max' for v in ['home', 'away'] for s in strengths}
        ).reset_index().rename(columns={'event_length': 'shift_length'})

        shifts = shifts.loc[shifts['shift_length'] > 0].copy()

        #Re-assign shift ids
        shift_id_map = shifts[['shift_id']].copy()
        shift_id_map['mapped_shift_id'] = np.arange(1, len(shift_id_map) + 1)
        shifts['shift_id'] = shift_id_map['mapped_shift_id']
        if include_maps:
            map_events = map_events.merge(shift_id_map, on='shift_id', how='inner')
            map_events['shift_id'] = map_events['mapped_shift_id']
            map_events = map_events.drop(columns=['mapped_shift_id'])

        #Orient dataframe perspective(s)    
        home_offense = orient_perspective(shifts, strengths, home=True)
        away_offense = orient_perspective(shifts, strengths, home=False)

        combined_shifts = pd.concat([home_offense, away_offense], ignore_index=True)

        #Since the target variable is offensive trim dataframe down to strength states provided for the offensive team
        strength_mask_cols = [f'strength_{s}' for s in strengths]
        sbs = combined_shifts.loc[combined_shifts[strength_mask_cols].eq(1).any(axis=1)]

        #Choose a single reference level among the provided non-empty-net strengths.
        drop_strength = reference_strength(strengths)

        #Drop reference level columns
        sbs = sbs.drop(
            columns=(
                ['on_the_fly']
                + ([f'strength_{drop_strength}'] if drop_strength else [])
            ), errors='ignore'
        )

        sbs = sbs.sort_values(by=['season', 'game_id', 'shift_id', 'is_home']).reset_index(drop=True)

        #Return: shift-by-shift dataframe with skater encoding
        return {
            'sbs': sbs,
            'shift_maps': build_shift_map_targets(map_events, sbs) if include_maps else None
        }

def classify_strength_state(strengths):
    s = set(strengths)
    if s <= {"5v5", "4v4", "3v3"}:
        return "even_strength"
    
    offense_nums = [int(st.split("v")[0]) for st in s]
    defense_nums = [int(st.split("v")[1]) for st in s]
    
    all_pp = all(o > d for o, d in zip(offense_nums, defense_nums))
    all_pk = all(o < d for o, d in zip(offense_nums, defense_nums))
    
    if all_pp:
        return "power_play"
    elif all_pk:
        return "penalty_kill"
    else:
        return "all"

def calc_info(sbs, strengths, shot=False):
    def normalize_entity_keys(df):
        df = df.copy()
        if "player_id" in df.columns:
            df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
        if "season" in df.columns:
            df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        if "team_abbr" in df.columns:
            df["team_abbr"] = df["team_abbr"].astype("string")
        return df

    if shot:
        shots = sbs[sbs["shooter_id"] != 0]

        fenwick = (
            shots.groupby(["shooter_id", "season", "shooter_team_abbr"])
            .size()
            .reset_index(name="fenwick")
            .rename(columns={"shooter_id": "player_id", "shooter_team_abbr": "team_abbr"})
        )
        fenwick = normalize_entity_keys(fenwick)

        goals = (
            shots[shots["event_type"] == "goal"]
                .groupby(["shooter_id", "season", "shooter_team_abbr"])
                .size()
                .reset_index(name="goals")
                .rename(columns={"shooter_id": "player_id", "shooter_team_abbr": "team_abbr"})
            )
        goals = normalize_entity_keys(goals)

        xg = (
            shots.groupby(["shooter_id", "season", "shooter_team_abbr"])["expected_goals"]
            .sum()
            .reset_index(name="expected_goals")
            .rename(columns={"shooter_id": "player_id", "shooter_team_abbr": "team_abbr"})
        )
        xg = normalize_entity_keys(xg)

        passer_rows = []
        for i in range(1, 7):
            tmp = sbs[["season", "shooter_team_abbr", f"passer_{i}_id",
                        f"passer_{i}_primary_fenwick_assist_probability",
                        f"passer_{i}_secondary_fenwick_assist_probability", 
                        "event_type",
                        "expected_goals"]].copy()
            tmp = tmp.rename(columns={
                "shooter_team_abbr": "team_abbr",
                f"passer_{i}_id": "player_id",
                f"passer_{i}_primary_fenwick_assist_probability": "primary_prob",
                f"passer_{i}_secondary_fenwick_assist_probability": "secondary_prob",
            })
            passer_rows.append(tmp)

        passer_long = pd.concat(passer_rows, ignore_index=True)
        passer_long = passer_long[(passer_long["player_id"] != 0) & (passer_long["player_id"].notna())]
        passer_long["primary_xa"] = passer_long["primary_prob"] * passer_long["expected_goals"]
        passer_long["secondary_xa"] = passer_long["secondary_prob"] * passer_long["expected_goals"]

        assist_agg = (
            passer_long.groupby(["player_id", "season", "team_abbr"], as_index=False)
            .agg(
                primary_fenwick_assists=("primary_prob", "sum"),
                secondary_fenwick_assists=("secondary_prob", "sum"),
                primary_expected_assists=("primary_xa", "sum"),
                secondary_expected_assists=("secondary_xa", "sum"),
            )
        )
        assist_agg = normalize_entity_keys(assist_agg)

        goal_passers = passer_long[passer_long["event_type"] == 'goal']

        actual_assists = (
            goal_passers.groupby(["player_id", "season", "team_abbr"], as_index=False)
            .agg(
                primary_assists=("primary_prob", "sum"),
                secondary_assists=("secondary_prob", "sum"),
            )
        )
        actual_assists = normalize_entity_keys(actual_assists)

        assist_agg = assist_agg.merge(
            actual_assists,
            on=["player_id", "season", "team_abbr"],
            how="left"
        )

        goalie_events = sbs[sbs["goalie_id"].notna() & (sbs["goalie_id"] != 0)]

        goalie_fenwick = (
            goalie_events
            .groupby(["goalie_id", "season", "goalie_team_abbr"])
            .size()
            .reset_index(name="fenwick_against")
            .rename(columns={"goalie_id": "player_id", "goalie_team_abbr": "team_abbr"})
        )
        goalie_fenwick = normalize_entity_keys(goalie_fenwick)

        goalie_goals = (
            goalie_events[goalie_events["event_type"] == 'goal']
            .groupby(["goalie_id", "season", "goalie_team_abbr"])
            .size()
            .reset_index(name="goals_against")
            .rename(columns={"goalie_id": "player_id", "goalie_team_abbr": "team_abbr"})
        )
        goalie_goals = normalize_entity_keys(goalie_goals)

        goalie_xg = (
            goalie_events
            .groupby(["goalie_id", "season", "goalie_team_abbr"])["expected_goals"]
            .sum()
            .reset_index(name="expected_goals_against")
            .rename(columns={"goalie_id": "player_id", "goalie_team_abbr": "team_abbr"})
        )
        goalie_xg = normalize_entity_keys(goalie_xg)

        goalie_result = (
            goalie_fenwick
            .merge(goalie_goals, on=["player_id","season","team_abbr"], how="left")
            .merge(goalie_xg, on=["player_id","season","team_abbr"], how="left")
            .fillna(0)
        )

        result = (
            fenwick
            .merge(goals, on=["player_id", "season", "team_abbr"], how="outer")
            .merge(xg, on=["player_id", "season", "team_abbr"], how="outer")
            .merge(assist_agg, on=["player_id", "season", "team_abbr"], how="outer")
        ).fillna(0)

        return (
            result[[
                "player_id", "season", "team_abbr",
                "fenwick",
                "goals",
                "expected_goals",
                "primary_fenwick_assists",
                "secondary_fenwick_assists",
                "primary_expected_assists",
                "secondary_expected_assists",
                "primary_assists",
                "secondary_assists",
            ]],
            goalie_result[[
                "player_id", "season", "team_abbr",
                "fenwick_against",
                "goals_against",
                "expected_goals_against",
            ]]
        )
    else:
        strength_state = classify_strength_state(strengths)
        score_states = ['tied', 'lead_1', 'lead_2', 'lead_3_plus', 'behind_1', 'behind_2', 'behind_3_plus']

        offense_cols = [f"offense_on_{i}_id" for i in range(1, 7)]
        defense_cols = [f"defense_on_{i}_id" for i in range(1, 7)]

        sbs = sbs.copy()
        sbs["offense_skaters"] = (sbs[offense_cols] != 0).sum(axis=1).astype(int)
        sbs["defense_skaters"] = (sbs[defense_cols] != 0).sum(axis=1).astype(int)
        sbs["strength_str"] = (
            sbs["offense_skaters"].astype(str) + "v" + sbs["defense_skaters"].astype(str)
        )
        sbs = sbs[sbs["strength_str"].isin(strengths)].copy()

        def build_side(df, side, entity="player"):
            if entity == "player":
                player_cols = offense_cols if side == "offense" else defense_cols
            team_col = "offense_team_abbr" if side == "offense" else "defense_team_abbr"
            coach_col = "offense_coach" if side == "offense" else "defense_coach"

            if entity == "player":
                cols = ["shift_id", "game_id", "season", "shift_length", team_col] + player_cols
            else:
                cols = ["shift_id", "game_id", "season", "shift_length", team_col, coach_col] + score_states

            result = df[cols].copy()

            rename_map = {
                "shift_length": "time_on_ice",
                team_col: "team_abbr",
            }
            if entity == "player":
                rename_map.update({f"{side}_on_{i}_id": f"player_{i}" for i in range(1, 7)})
            else:
                rename_map[coach_col] = "coach_name"

            result = result.rename(columns=rename_map)
            result["side"] = side
            return result

        # Players
        offense_side = build_side(sbs, "offense", "player")
        defense_side = build_side(sbs, "defense", "player")
        result = pd.concat([offense_side, defense_side], ignore_index=True)
        result["strength_state"] = strength_state

        # Coaches
        offense_coach_side = build_side(sbs, "offense", "coach")
        defense_coach_side = build_side(sbs, "defense", "coach")
        coach_result = pd.concat([offense_coach_side, defense_coach_side], ignore_index=True)
        coach_result["strength_state"] = strength_state

        player_cols = [f"player_{i}" for i in range(1, 7)]
        id_vars = ["shift_id", "game_id", "season", "time_on_ice", "team_abbr", "side", "strength_state"]
        coach_id_vars = id_vars + score_states

        if strength_state in ("even_strength", "all"):
            to_melt = result[result["side"] == "offense"]
            to_melt_coach = coach_result[coach_result["side"] == "offense"]
        else:
            to_melt = result
            to_melt_coach = coach_result

        long = to_melt.melt(
            id_vars=id_vars,
            value_vars=player_cols,
            value_name="player_id",
        ).dropna(subset=["player_id"])
        long = long[long["player_id"] != 0]

        long_coach = to_melt_coach[coach_id_vars + ["coach_name"]].copy()
        long_coach = long_coach.dropna(subset=["coach_name"])
        long_coach = long_coach[long_coach["coach_name"] != 0]

        # Games played — unique game_ids per player/coach regardless of strength
        gp = (
            long.groupby("player_id")["game_id"]
            .nunique()
            .reset_index()
            .rename(columns={"game_id": "games_played"})
        )
        gp_coach = (
            long_coach.groupby("coach_name")["game_id"]
            .nunique()
            .reset_index()
            .rename(columns={"game_id": "games_played"})
        )

        def aggregate_toi(df, id_col, strength_state):
            if strength_state in ("even_strength", "all"):
                agg = df.groupby(
                    [id_col, "season", "team_abbr", "strength_state"], as_index=False
                )["time_on_ice"].sum()
            else:
                agg = df.groupby(
                    [id_col, "season", "team_abbr", "side", "strength_state"], as_index=False
                )["time_on_ice"].sum()

                if strength_state == "power_play":
                    agg["strength_state"] = agg["side"].map({"offense": "power_play", "defense": "penalty_kill"})
                elif strength_state == "penalty_kill":
                    agg["strength_state"] = agg["side"].map({"offense": "penalty_kill", "defense": "power_play"})

                agg = agg.drop(columns="side")

            agg["time_on_ice"] = agg["time_on_ice"] / 60
            return agg

        def aggregate_coach_state_toi(df, strength_state):
            value_cols = [f"{state}_time_on_ice" for state in score_states]
            work = df[["coach_name", "season", "team_abbr", "side", "strength_state", "time_on_ice"] + score_states].copy()
            for state in score_states:
                work[f"{state}_time_on_ice"] = work["time_on_ice"] * work[state]

            if strength_state in ("even_strength", "all"):
                agg = work.groupby(
                    ["coach_name", "season", "team_abbr", "strength_state"], as_index=False
                )[value_cols].sum()
            else:
                agg = work.groupby(
                    ["coach_name", "season", "team_abbr", "side", "strength_state"], as_index=False
                )[value_cols].sum()

                if strength_state == "power_play":
                    agg["strength_state"] = agg["side"].map({"offense": "power_play", "defense": "penalty_kill"})
                elif strength_state == "penalty_kill":
                    agg["strength_state"] = agg["side"].map({"offense": "penalty_kill", "defense": "power_play"})

                agg = agg.drop(columns="side")

            agg[value_cols] = agg[value_cols] / 60
            return agg

        agg = aggregate_toi(long, "player_id", strength_state)
        agg = agg.merge(gp, on="player_id", how="left")
        agg['season_type'] = 2

        agg_coach = aggregate_toi(long_coach, "coach_name", strength_state)
        agg_coach = agg_coach.merge(gp_coach, on="coach_name", how="left")
        agg_coach['season_type'] = 2
        state_agg = aggregate_coach_state_toi(long_coach, strength_state)
        agg_coach = agg_coach.merge(
            state_agg,
            on=["coach_name", "season", "team_abbr", "strength_state"],
            how="left"
        )
        state_toi_cols = [f"{state}_time_on_ice" for state in score_states]
        agg_coach[state_toi_cols] = agg_coach[state_toi_cols].fillna(0)

        return (
            agg[["player_id", "season", "team_abbr", "games_played", "time_on_ice", "strength_state", "season_type"]],
            agg_coach[["coach_name", "season", "team_abbr", "games_played", "time_on_ice", "strength_state", "season_type"] + state_toi_cols]
        )


def model_table_format(model_path):
    suffix = Path(str(model_path)).suffix.lower()
    if suffix in {".csv", ".parquet"}:
        return suffix[1:]
    return None


def read_model_table(model_path):
    if model_table_format(model_path) == "parquet":
        return pd.read_parquet(model_path)
    return pd.read_csv(model_path)


def write_model_table(frame, model_path):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    if model_table_format(model_path) == "parquet":
        frame.to_parquet(model_path, index=False)
    else:
        frame.to_csv(model_path, index=False)


def frame_season(frame):
    if "season" not in frame.columns:
        return None

    seasons = sorted({int(season) for season in pd.to_numeric(frame["season"], errors="coerce").dropna().tolist()})
    if not seasons:
        return None
    if len(seasons) == 1:
        return seasons[0]
    return ", ".join(str(season) for season in seasons)


def coef_row(
    *,
    metric,
    strength_state,
    season_value,
    scalar_predictors,
    scalar_values,
    scalar_standard_errors=None,
    scalar_margin_of_errors=None,
    scalar_p_values=None,
    scalar_ci_lower=None,
    scalar_ci_upper=None,
    map_predictors=None,
    map_values=None,
    model_metrics=None,
):
    row = {
        "metric": metric,
        "strength_state": strength_state,
        "season": season_value,
    }
    if model_metrics:
        row.update(model_metrics)
    map_lookup = dict(zip(map_predictors or [], map_values or []))

    for i, predictor in enumerate(scalar_predictors):
        row[predictor] = scalar_values[i]
        if predictor in map_lookup:
            row[f"{predictor}_map"] = map_lookup[predictor]
        if scalar_standard_errors is not None:
            row[f"{predictor}_standard_error"] = scalar_standard_errors[i]
        if scalar_margin_of_errors is not None:
            row[f"{predictor}_margin_of_error"] = scalar_margin_of_errors[i]
        if scalar_p_values is not None:
            row[f"{predictor}_p_value"] = scalar_p_values[i]
        if scalar_ci_lower is not None:
            row[f"{predictor}_ci_lower"] = scalar_ci_lower[i]
        if scalar_ci_upper is not None:
            row[f"{predictor}_ci_upper"] = scalar_ci_upper[i]

    return row


def merge_model_rows(existing_df, new_df):
    if existing_df is None or existing_df.empty:
        return new_df.copy()
    if new_df.empty:
        return existing_df.copy()

    key_cols = ["metric", "strength_state", "season"]
    all_columns = list(dict.fromkeys([*existing_df.columns.tolist(), *new_df.columns.tolist()]))
    existing_aligned = existing_df.reindex(columns=all_columns)
    new_aligned = new_df.reindex(columns=all_columns)

    keep_existing = pd.Series(True, index=existing_aligned.index)
    for _, new_row in new_aligned.iterrows():
        mask = pd.Series(True, index=existing_aligned.index)
        for key in key_cols:
            mask &= existing_aligned[key].astype(str).eq(str(new_row[key]))
        keep_existing &= ~mask

    return pd.concat([existing_aligned.loc[keep_existing], new_aligned], ignore_index=True)


def target_map_flags(targets, maps):
    targets = list(targets)
    if isinstance(maps, (bool, np.bool_)):
        return dict.fromkeys(targets, bool(maps))

    flags = list(maps)
    if len(flags) != len(targets):
        raise ValueError("When maps is a list, it must have the same length as targets.")
    return dict(zip(targets, [bool(flag) for flag in flags]))

def wsba_rapm(
    pbp, 
    strengths, 
    targets=None,
    shot=False, 
    model_path='wsba_reg.parquet',
    feature_filters=None,
    params=None,
    uncertainty=True,
    maps=True,
    solve_workers=1
):
    #Train and calculate the WSBA RAPM model
    if targets is None:
        targets = ['expected_goals']
    if params is None:
        params = {
            'expected_goals': {
                'alpha': 50000
            }
        }

    #Retrieve shift by shift dataframe
    map_flags = target_map_flags(targets, maps)
    include_maps = (not shot) and any(map_flags.get(target, False) and target in MAP_TARGETS for target in targets)
    prepared = prepare_rapm_data(pbp, strengths, shot=shot, include_maps=include_maps)
    sbs = prepared if shot else prepared['sbs']
    shift_maps = None if shot else prepared['shift_maps']
    derived_strength_state = classify_strength_state(strengths)
    
    #Prepare final time_on_ice dataframe
    toi_df, second_df = calc_info(sbs, strengths, shot)

    #Encode and strip dataframe down to sparse matrix
    data = encode_shooters(sbs) if shot else encode_players(sbs)
    predictor_names = data['features']
    sparse = data['predictors']
    model_path_is_table = model_table_format(model_path) is not None
    season_value = frame_season(sbs)

    #Load in existing model data (if it exists)
    if model_path_is_table:
        model_data = {}
    else:
        try:
            #Collect the model data generated while building the model
            with open(model_path, 'r') as f:
                model_data = json.load(f)
        except:
            #Start creation of model
            model_data = {}

    model_type = 'shot' if shot else 'shift'
    resolved_feature_filters = feature_filters or (DEFAULT_FEATURES['shots'] if shot else DEFAULT_FEATURES['shifts'])
    target_views = {
        target: select_target_predictors(sparse, predictor_names, target, feature_filters=resolved_feature_filters)
        for target in targets
    }

    if model_type not in model_data:
        model_data[model_type] = {}

    system_component_cache = {}

    #At provided strengths, run the model for each provided target
    rapm_dfs = []
    coach_dfs = []
    goalie_dfs = []
    model_rows = []
    for target in targets:
        # Select predictors based on current target
        use_shift_map = (not shot) and map_flags.get(target, False) and (target in MAP_TARGETS)
        weight = None if shot else sbs['shift_length'].values
        live_coef_frame = None
        target_params = params[target]
        target_solve_workers = normalize_workers(target_params.get('solve_workers', solve_workers))
        target_view = target_views[target]
        target_sparse = target_view['predictors']
        target_predictor_names = target_view['features']
        target_entity_mask = target_view['entity_mask']
        target_context_mask = target_view['context_mask']
        system_components = None
        if not shot:
            system_key = tuple(target_predictor_names)
            if system_key not in system_component_cache:
                system_component_cache[system_key] = normal_components(target_sparse, sbs['shift_length'].values)
            system_components = system_component_cache[system_key]
        if len(target_predictor_names) != len(predictor_names):
            print(f'Using {len(target_predictor_names)} / {len(predictor_names)} encoded predictors for {target}')

        if shot:
            target_series = sbs[target].values
        else:
            target_series = sbs[f'{target}_for_per_sixty'].values

        regularization_key = 'C' if shot else 'alpha'
        if regularization_key not in target_params:
            raise ValueError(f"RAPM target {target!r} requires params[{target!r}]['{regularization_key}'].")
        optimal = target_params[regularization_key]
        model_data[model_type][f'optimal_{regularization_key}'] = optimal

        print(f'Optimal parameter ({model_type}, {target}): {optimal:.4f}')
        print('### FINAL MODEL ###')

        # Final model fit
        shared_system = None
        shared_solver = None
        map_groups = None
        saved_map_predictors = []
        saved_map_values = []

        if use_shift_map:
            print(f'Fitting spatial maps ({model_type}, {target})...')
            shared_system = penalized_system_from_components(*system_components, optimal)
            if target_solve_workers == 1 or uncertainty:
                shared_solver = splu(shared_system)
            map_groups = fit_shift_map_model(
                sparse=target_sparse,
                weights=weight,
                response=shift_maps[target],
                predictor_names=target_predictor_names,
                alpha=optimal,
                selected_masks={
                    'entity': target_entity_mask,
                    'context': target_context_mask
                },
                system=shared_system,
                solver=shared_solver if target_solve_workers == 1 else None,
                solve_workers=target_solve_workers
            )
            print(f'Finished spatial maps ({model_type}, {target}).')

            intercept_map_predictors, intercept_map_scalar, _ = map_groups['intercept']
            entity_map_predictors, entity_map_scalar, _ = map_groups['entity']
            context_map_predictors, context_map_scalar, _ = map_groups['context']
            scalar_lookup = dict(zip(intercept_map_predictors, intercept_map_scalar))
            scalar_lookup.update(dict(zip(entity_map_predictors, entity_map_scalar)))
            scalar_lookup.update(dict(zip(context_map_predictors, context_map_scalar)))

            live_scalar_predictors = ['(intercept)'] + target_predictor_names
            live_scalar_values = [float(scalar_lookup[name]) for name in live_scalar_predictors]
            coefficient_vector = np.asarray(live_scalar_values, dtype=np.float64)
            model = _SparseRidgeModel(coefficient_vector)
            saved_map_predictors = list(context_map_predictors)
            saved_map_values = list(map_groups['context'][2])
        else:
            print(f'Fitting final {model_type} model ({target})...')
            if shot:
                model = shot_estimator(optimal, target_params)
                model.fit(target_sparse, target_series)
            else:
                needs_repeated_solves = bool(uncertainty)
                model, shared_system, shared_solver = fit_sparse_ridge(
                    target_sparse,
                    target_series,
                    weight,
                    optimal,
                    factorize=needs_repeated_solves,
                    tol=target_params.get('fit_tol', 1e-8),
                    maxiter=target_params.get('fit_max_iter'),
                    system_components=system_components
                )
            print(f'Finished final {model_type} model ({target}).')

            coefficient_vector = coef_vector(model, shot)
            live_scalar_predictors = ['(intercept)'] + target_predictor_names
            live_scalar_values = coefficient_vector.tolist()

        intercept = float(live_scalar_values[0])
        coefs = list(live_scalar_values[1:])

        fit_metrics = model_fit_metrics(
            model,
            target_sparse,
            target_series,
            None if shot else weight,
            shot,
            optimal
        )

        coefficient_lookup = dict(zip(live_scalar_predictors, live_scalar_values))

        if uncertainty:
            standard_errors, margin_of_errors, p_values, ci_lower, ci_upper = residual_coefficient_intervals(
                coefficient_vector,
                model,
                target_sparse,
                target_series,
                None if shot else weight,
                shot,
                optimal,
                system=shared_system,
                solver=shared_solver,
                solve_workers=target_solve_workers
            )
        else:
            standard_errors = np.full_like(coefficient_vector, np.nan, dtype=np.float64)
            margin_of_errors = np.full_like(coefficient_vector, np.nan, dtype=np.float64)
            p_values = np.full_like(coefficient_vector, np.nan, dtype=np.float64)
            ci_lower = np.full_like(coefficient_vector, np.nan, dtype=np.float64)
            ci_upper = np.full_like(coefficient_vector, np.nan, dtype=np.float64)
        live_standard_errors = standard_errors.tolist()
        live_margin_of_errors = margin_of_errors.tolist()
        live_p_values = p_values.tolist()
        live_ci_lower = ci_lower.tolist()
        live_ci_upper = ci_upper.tolist()
        saved_predictors = ['(intercept)'] + [name for name, keep in zip(target_predictor_names, target_context_mask) if keep]
        saved_values = [intercept] + [coef for coef, keep in zip(coefs, target_context_mask) if keep]
        saved_standard_errors = [live_standard_errors[0]] + [se for se, keep in zip(live_standard_errors[1:], target_context_mask) if keep]
        saved_margin_of_errors = [live_margin_of_errors[0]] + [moe for moe, keep in zip(live_margin_of_errors[1:], target_context_mask) if keep]
        saved_p_values = [live_p_values[0]] + [p for p, keep in zip(live_p_values[1:], target_context_mask) if keep]
        saved_ci_lower = [live_ci_lower[0]] + [ci for ci, keep in zip(live_ci_lower[1:], target_context_mask) if keep]
        saved_ci_upper = [live_ci_upper[0]] + [ci for ci, keep in zip(live_ci_upper[1:], target_context_mask) if keep]

        if not model_path_is_table:
            model_data[model_type].update({
                f'{target}_predictors': saved_predictors,
                target: saved_values,
                f'{target}_standard_error': saved_standard_errors,
                f'{target}_margin_of_error': saved_margin_of_errors,
                f'{target}_p_value': saved_p_values,
                f'{target}_ci_lower': saved_ci_lower,
                f'{target}_ci_upper': saved_ci_upper,
                f'{target}_all_predictors': live_scalar_predictors,
                f'{target}_all': live_scalar_values,
                f'{target}_all_standard_error': live_standard_errors,
                f'{target}_all_margin_of_error': live_margin_of_errors,
                f'{target}_all_p_value': live_p_values,
                f'{target}_all_ci_lower': live_ci_lower,
                f'{target}_all_ci_upper': live_ci_upper,
                f'{target}_model_metrics': fit_metrics
            })

        if use_shift_map:
            entity_map_predictors, entity_map_scalar, entity_map_values = map_groups['entity']
            context_map_predictors, context_map_scalar, context_map_values = map_groups['context']

            live_coef_frame = pd.DataFrame({
                'player_id': entity_map_predictors,
                target: entity_map_scalar,
                f'{target}_map': entity_map_values
            })
            live_coef_frame = live_coef_frame.merge(
                pd.DataFrame({
                    'player_id': live_scalar_predictors,
                    f'{target}_standard_error': live_standard_errors,
                    f'{target}_margin_of_error': live_margin_of_errors,
                    f'{target}_p_value': live_p_values,
                    f'{target}_ci_lower': live_ci_lower,
                    f'{target}_ci_upper': live_ci_upper,
                }),
                on='player_id',
                how='left'
            )

            if not model_path_is_table:
                model_data[model_type].update({
                    f'{target}_map_predictors': context_map_predictors,
                    f'{target}_map_scalar': context_map_scalar,
                    f'{target}_map': context_map_values,
                    f'{target}_map_shape': list(MAP_SHAPE),
                    f'{target}_map_x': MAP_X.tolist(),
                    f'{target}_map_y': MAP_Y.tolist()
                })
        else:
            live_coef_frame = pd.DataFrame({
                'player_id': live_scalar_predictors,
                target: live_scalar_values,
                f'{target}_standard_error': live_standard_errors,
                f'{target}_margin_of_error': live_margin_of_errors,
                f'{target}_p_value': live_p_values,
                f'{target}_ci_lower': live_ci_lower,
                f'{target}_ci_upper': live_ci_upper
            })

        # Export
        if not model_path_is_table:
            with open(model_path, 'w') as f:
                json.dump(model_data, f, indent=2)

        model_rows.append(
            coef_row(
                metric=target,
                strength_state=derived_strength_state,
                season_value=season_value,
                scalar_predictors=saved_predictors,
                scalar_values=saved_values,
                scalar_standard_errors=saved_standard_errors,
                scalar_margin_of_errors=saved_margin_of_errors,
                scalar_p_values=saved_p_values,
                scalar_ci_lower=saved_ci_lower,
                scalar_ci_upper=saved_ci_upper,
                map_predictors=saved_map_predictors,
                map_values=saved_map_values,
                model_metrics=fit_metrics,
            )
        )

        coefs = live_coef_frame.copy()

        #Extract group from player dummies
        pattern_off = r'^shooter_(\d+)_(\d+)_([A-Z]{2,3})$' if shot else r'^offense_(\d+)_(\d+)_([A-Z]{2,3})$'
        pattern_def = r'^primary_passer_(\d+)_(\d+)_([A-Z]{2,3})$' if shot else r'^defense_(\d+)_(\d+)_([A-Z]{2,3})$'
        pattern_tri = r'^secondary_passer_(\d+)_(\d+)_([A-Z]{2,3})$' if shot else None
        pattern_goa = r'^goalie_(\d+)_(\d+)_([A-Z]{2,3})$' if shot else None
        offense_col_name = f'rapm_{target}' if shot else f'rapm_{target}_for_per_sixty'
        defense_target_name = f"primary_{target.replace('goals', 'assists')}" if shot else target
        defense_col_name = f'rapm_{defense_target_name}' if shot else f'rapm_{target}_against_per_sixty'
        offense_map_col_name = f'rapm_{target}_for_per_sixty_map'
        defense_map_col_name = f'rapm_{target}_against_per_sixty_map'
        score_states = ['tied', 'lead_1', 'lead_2', 'lead_3_plus', 'behind_1', 'behind_2', 'behind_3_plus']
        score_pattern = '|'.join(score_states)
        pattern_off_coa = r'^offense_coach_([A-ZÀ-ÖØ-Ý.\'\- ]+)_(\d{8})_([A-Z]{2,3})$'
        pattern_def_coa = r'^defense_coach_([A-ZÀ-ÖØ-Ý.\'\- ]+)_(\d{8})_([A-Z]{2,3})$'
        pattern_off_coa_score = rf'^offense_coach_({score_pattern})_([A-ZÀ-ÖØ-İ.\'\- ]+)_(\d{{8}})_([A-Z]{{2,3}})$'
        pattern_def_coa_score = rf'^defense_coach_({score_pattern})_([A-ZÀ-ÖØ-İ.\'\- ]+)_(\d{{8}})_([A-Z]{{2,3}})$'
        
        #Skaters
        offense_RAPM = coefs.loc[
            coefs['player_id'].str.contains(pattern_off, regex=True, na=False)
        ].copy()

        offense_RAPM[['player_id', 'season', 'team_abbr']] = (
            offense_RAPM['player_id'].str.extract(pattern_off)
        )

        offense_RAPM[['player_id', 'season']] = offense_RAPM[['player_id', 'season']].astype('Int64')
        offense_rename = interval_names(target, offense_col_name)
        if use_shift_map and not shot and f'{target}_map' in offense_RAPM.columns:
            offense_rename[f'{target}_map'] = offense_map_col_name
        offense_RAPM = offense_RAPM.rename(columns=offense_rename)

        defense_RAPM = coefs.loc[
            coefs['player_id'].str.contains(pattern_def, regex=True, na=False)
        ].copy()

        defense_RAPM[['player_id', 'season', 'team_abbr']] = (
            defense_RAPM['player_id'].str.extract(pattern_def)
        )

        defense_RAPM[['player_id', 'season']] = defense_RAPM[['player_id', 'season']].astype('Int64')
        defense_rename = interval_names(target, defense_col_name)
        if use_shift_map and not shot and f'{target}_map' in defense_RAPM.columns:
            defense_rename[f'{target}_map'] = defense_map_col_name
        defense_RAPM = defense_RAPM.rename(columns=defense_rename)
        if use_shift_map and not shot and defense_map_col_name in defense_RAPM.columns:
            defense_RAPM[defense_map_col_name] = defense_RAPM[defense_map_col_name].map(
                lambda encoded: transform_rapm_map(encoded, reverse_x=True) if pd.notna(encoded) else encoded
            )

        #Coaches
        offense_coach_RAPM = coefs.loc[
            coefs['player_id'].str.contains(pattern_off_coa, regex=True, na=False)
        ].copy()

        offense_coach_RAPM[['player_id', 'season', 'team_abbr']] = (
            offense_coach_RAPM['player_id'].str.extract(pattern_off_coa)
        )

        offense_coach_RAPM['season'] = offense_coach_RAPM['season'].astype('Int64')
        offense_coach_rename = {
            'player_id': 'coach_name'
        }
        offense_coach_rename.update(interval_names(target, f'rapm_systems_{target}_for_per_sixty'))
        if use_shift_map and not shot and f'{target}_map' in offense_coach_RAPM.columns:
            offense_coach_rename[f'{target}_map'] = f'rapm_systems_{target}_for_per_sixty_map'
        offense_coach_RAPM = offense_coach_RAPM.rename(columns=offense_coach_rename)

        defense_coach_RAPM = coefs.loc[
            coefs['player_id'].str.contains(pattern_def_coa, regex=True, na=False)
        ].copy()

        defense_coach_RAPM[['player_id', 'season', 'team_abbr']] = (
            defense_coach_RAPM['player_id'].str.extract(pattern_def_coa)
        )

        defense_coach_RAPM['season'] = defense_coach_RAPM['season'].astype('Int64')
        defense_coach_rename = {
            'player_id': 'coach_name'
        }
        defense_coach_rename.update(interval_names(target, f'rapm_systems_{target}_against_per_sixty'))
        if use_shift_map and not shot and f'{target}_map' in defense_coach_RAPM.columns:
            defense_coach_rename[f'{target}_map'] = f'rapm_systems_{target}_against_per_sixty_map'
        defense_coach_RAPM = defense_coach_RAPM.rename(columns=defense_coach_rename)
        if use_shift_map and not shot and f'rapm_systems_{target}_against_per_sixty_map' in defense_coach_RAPM.columns:
            defense_coach_RAPM[f'rapm_systems_{target}_against_per_sixty_map'] = defense_coach_RAPM[f'rapm_systems_{target}_against_per_sixty_map'].map(
                lambda encoded: transform_rapm_map(encoded, reverse_x=True) if pd.notna(encoded) else encoded
            )

        offense_score_coach = coefs.loc[
            coefs['player_id'].str.contains(pattern_off_coa_score, regex=True, na=False)
        ].copy()
        if not offense_score_coach.empty:
            offense_score_coach[['score_state', 'coach_name', 'season', 'team_abbr']] = offense_score_coach['player_id'].str.extract(pattern_off_coa_score)
            offense_score_coach['season'] = offense_score_coach['season'].astype('Int64')
            offense_score_dfs = []
            for state in score_states:
                temp = offense_score_coach.loc[offense_score_coach['score_state'] == state].copy()
                if temp.empty:
                    continue
                rename = interval_names(target, f'rapm_{state}_{target}_for_per_sixty')
                if use_shift_map and not shot and f'{target}_map' in temp.columns:
                    rename[f'{target}_map'] = f'rapm_{state}_{target}_for_per_sixty_map'
                offense_score_dfs.append(temp.rename(columns=rename)[['coach_name', 'season', 'team_abbr'] + list(rename.values())])
            if offense_score_dfs:
                offense_coach_RAPM = reduce(lambda left, right: pd.merge(left, right, how='outer', on=['coach_name', 'season', 'team_abbr']), [offense_coach_RAPM] + offense_score_dfs)

        defense_score_coach = coefs.loc[
            coefs['player_id'].str.contains(pattern_def_coa_score, regex=True, na=False)
        ].copy()
        if not defense_score_coach.empty:
            defense_score_coach[['score_state', 'coach_name', 'season', 'team_abbr']] = defense_score_coach['player_id'].str.extract(pattern_def_coa_score)
            defense_score_coach['season'] = defense_score_coach['season'].astype('Int64')
            defense_score_dfs = []
            for state in score_states:
                temp = defense_score_coach.loc[defense_score_coach['score_state'] == state].copy()
                if temp.empty:
                    continue
                rename = interval_names(target, f'rapm_{state}_{target}_against_per_sixty')
                if use_shift_map and not shot and f'{target}_map' in temp.columns:
                    temp[f'{target}_map'] = temp[f'{target}_map'].map(
                        lambda encoded: transform_rapm_map(encoded, reverse_x=True) if pd.notna(encoded) else encoded
                    )
                    rename[f'{target}_map'] = f'rapm_{state}_{target}_against_per_sixty_map'
                defense_score_dfs.append(temp.rename(columns=rename)[['coach_name', 'season', 'team_abbr'] + list(rename.values())])
            if defense_score_dfs:
                defense_coach_RAPM = reduce(lambda left, right: pd.merge(left, right, how='outer', on=['coach_name', 'season', 'team_abbr']), [defense_coach_RAPM] + defense_score_dfs)

        if shot:
            third_RAPM = coefs.loc[
                coefs['player_id'].str.contains(pattern_tri, regex=True, na=False)
            ].copy()

            third_RAPM[['player_id', 'season', 'team_abbr']] = (
                third_RAPM['player_id'].str.extract(pattern_tri)
            )

            third_RAPM[['player_id', 'season']] = third_RAPM[['player_id', 'season']].astype('Int64')
            third_RAPM = third_RAPM.rename(
                columns=interval_names(target, f"rapm_secondary_{target.replace('goals', 'assists')}")
            )

            fourth_RAPM = coefs.loc[
                coefs['player_id'].str.contains(pattern_goa, regex=True, na=False)
            ].copy()

            fourth_RAPM[['player_id', 'season', 'team_abbr']] = (
                fourth_RAPM['player_id'].str.extract(pattern_goa)
            )

            fourth_RAPM[['player_id', 'season']] = fourth_RAPM[['player_id', 'season']].astype('Int64')
            fourth_RAPM = fourth_RAPM.rename(
                columns=interval_names(target, 'rapm_goals_against')
            )

            goalie_df = pd.merge(second_df, fourth_RAPM)

            defense_RAPM = pd.merge(defense_RAPM, third_RAPM, how='inner', on=['player_id', 'season', 'team_abbr'])

        strength_state = derived_strength_state

        #Join and return final RAPM dataframe
        if shot:
            rapm_df = pd.merge(offense_RAPM, defense_RAPM, how='inner', on=['player_id', 'season', 'team_abbr'])
            rapm_df = center_rapm_columns(rapm_df, [offense_col_name, defense_col_name])
            rapm_df['strength_state'] = strength_state
            rapm_df['season_type'] = 2

        else:
            if strength_state in ("even_strength", "all"):
                rapm_df = pd.merge(offense_RAPM, defense_RAPM, how='inner', on=['player_id', 'season', 'team_abbr'])
                rapm_df = center_rapm_columns(rapm_df, [offense_col_name, defense_col_name])
                rapm_df["strength_state"] = strength_state
                
                coach_df = pd.merge(offense_coach_RAPM, defense_coach_RAPM, how='inner', on=['coach_name', 'season', 'team_abbr'])
                coach_df["strength_state"] = strength_state

            else:
                offense_RAPM["strength_state"] = "power_play" if strength_state == "power_play" else "penalty_kill"
                defense_RAPM["strength_state"] = "penalty_kill" if strength_state == "power_play" else "power_play"
                offense_RAPM = center_rapm_columns(offense_RAPM, [offense_col_name])
                defense_RAPM = center_rapm_columns(defense_RAPM, [defense_col_name])
                rapm_df = pd.concat([offense_RAPM, defense_RAPM], ignore_index=True)

                offense_coach_RAPM["strength_state"] = "power_play" if strength_state == "power_play" else "penalty_kill"
                defense_coach_RAPM["strength_state"] = "penalty_kill" if strength_state == "power_play" else "power_play"
                coach_df = pd.concat([offense_coach_RAPM, defense_coach_RAPM], ignore_index=True)

        #Convert log-odds to per-shot impact
        if shot:
            p = model.predict_proba(target_sparse)[:, 1]
            scale = np.mean(p * (1 - p))

            rapm_df = scale_shot_columns(rapm_df, scale)
            goalie_df = scale_shot_columns(goalie_df, scale)

            goalie_dfs.append(goalie_df)
        else:
            coach_df = pd.merge(second_df, coach_df)
            for side in ['for', 'against']:
                systems_col = f'rapm_systems_{target}_{side}_per_sixty'
                total_col = f'rapm_{target}_{side}_per_sixty'
                if systems_col in coach_df.columns:
                    toi = coach_df['time_on_ice'].fillna(0)
                    total = coach_df[systems_col].fillna(0).mul(toi).div(60)
                    systems_se_col = f'{systems_col}_standard_error'
                    systems_lower_col = f'{systems_col}_ci_lower'
                    systems_upper_col = f'{systems_col}_ci_upper'
                    total_var = None
                    total_lower = None
                    total_upper = None
                    if systems_se_col in coach_df.columns:
                        total_var = coach_df[systems_se_col].fillna(0).mul(toi).div(60) ** 2
                    if systems_lower_col in coach_df.columns and systems_upper_col in coach_df.columns:
                        total_lower = coach_df[systems_lower_col].fillna(0).mul(toi).div(60)
                        total_upper = coach_df[systems_upper_col].fillna(0).mul(toi).div(60)
                    for state in score_states:
                        state_col = f'rapm_{state}_{target}_{side}_per_sixty'
                        state_toi_col = f'{state}_time_on_ice'
                        if state_col in coach_df.columns and state_toi_col in coach_df.columns:
                            state_toi = coach_df[state_toi_col].fillna(0)
                            total = total.add(coach_df[state_col].fillna(0).mul(state_toi).div(60))
                            state_se_col = f'{state_col}_standard_error'
                            if total_var is not None and state_se_col in coach_df.columns:
                                total_var = total_var.add(
                                    coach_df[state_se_col].fillna(0).mul(state_toi).div(60) ** 2
                                )
                            state_lower_col = f'{state_col}_ci_lower'
                            state_upper_col = f'{state_col}_ci_upper'
                            if total_lower is not None and state_lower_col in coach_df.columns and state_upper_col in coach_df.columns:
                                total_lower = total_lower.add(coach_df[state_lower_col].fillna(0).mul(state_toi).div(60))
                                total_upper = total_upper.add(coach_df[state_upper_col].fillna(0).mul(state_toi).div(60))
                    coach_df[total_col] = np.where(toi > 0, total.div(toi).mul(60), np.nan)
                    if total_var is not None:
                        total_se_col = f'{total_col}_standard_error'
                        coach_df[total_se_col] = np.where(toi > 0, np.sqrt(total_var).div(toi).mul(60), np.nan)
                        coach_df[f'{total_col}_p_value'] = coefficient_p_values(coach_df[total_col], coach_df[total_se_col])
                    if total_lower is not None and total_upper is not None:
                        total_lower_col = f'{total_col}_ci_lower'
                        total_upper_col = f'{total_col}_ci_upper'
                        coach_df[total_lower_col] = np.where(toi > 0, total_lower.div(toi).mul(60), np.nan)
                        coach_df[total_upper_col] = np.where(toi > 0, total_upper.div(toi).mul(60), np.nan)
                        coach_df[f'{total_col}_margin_of_error'] = np.maximum(
                            np.abs(coach_df[total_col] - coach_df[total_lower_col]),
                            np.abs(coach_df[total_upper_col] - coach_df[total_col])
                        )
                systems_map_col = f'rapm_systems_{target}_{side}_per_sixty_map'
                total_map_col = f'rapm_{target}_{side}_per_sixty_map'
                if systems_map_col in coach_df.columns:
                    coach_df[total_map_col] = coach_df[systems_map_col]
            coach_dfs.append(coach_df)

        rapm_df = pd.merge(toi_df, rapm_df, how='left')
        rapm_dfs.append(rapm_df)
        
    df = reduce(lambda left, right: pd.merge(left, right, how='outer'), rapm_dfs)
    df2 = reduce(lambda left, right: pd.merge(left, right, how='outer'), (goalie_dfs if shot else coach_dfs))

    if model_path_is_table:
        new_model_df = pd.DataFrame(model_rows)
        if Path(model_path).exists():
            try:
                existing_model_df = read_model_table(model_path)
            except Exception:
                existing_model_df = pd.DataFrame()
        else:
            existing_model_df = pd.DataFrame()

        merged_model_df = merge_model_rows(existing_model_df, new_model_df)
        write_model_table(merged_model_df, model_path)

    return df, df2

def rink_grid_format(data, mode="to_grid", shape=(100, 85)):
    if mode == "to_grid":
        df = data.copy()
        grid = np.zeros(shape, dtype=float)
        grid[
            df["x"].to_numpy().astype(int),
            df["y"].to_numpy().astype(int)
        ] = df["value"].to_numpy()
        return grid

    elif mode == "to_df":
        grid = data
        x, y = np.nonzero(grid)
        return pd.DataFrame({
            "x": x,
            "y": y,
            "value": grid[x, y]
        })

def rapm_grid(
    rapm,
    season,
    entity_id,
    group,
    metric='fenwick',
    strength='even_strength'
):
    ev_for, ev_against, st_for, st_against = rapm_four_maps(
        rapm,
        season,
        entity_id,
        group=group,
        metric=metric,
    )

    if strength == 'special_teams':
        return st_for, st_against
    if strength == 'even_strength':
        return ev_for, ev_against

    seasons = season_values(season)
    if seasons is not None:
        rapm = rapm.loc[rapm['season'].isin(seasons)]

    group_id = 'coach_name' if group == 'coach' else 'player_id'
    player_rows = rapm.loc[(rapm[group_id] == entity_id) & (rapm['strength_state'] == strength)]

    return (
        weighted_map(player_rows, f'rapm_{metric}_for_per_sixty_map'),
        weighted_map(player_rows, f'rapm_{metric}_against_per_sixty_map'),
    )


def weighted_map(rows, map_col):
    if rows.empty or map_col not in rows.columns:
        return np.zeros(MAP_SHAPE, dtype=np.float32)

    valid = rows.loc[rows[map_col].notna()].copy()
    if valid.empty:
        return np.zeros(MAP_SHAPE, dtype=np.float32)

    toi = valid['time_on_ice'].to_numpy(dtype=float)
    total_toi = float(toi.sum())
    if total_toi <= 0.0:
        return np.zeros(MAP_SHAPE, dtype=np.float32)

    grids = np.stack([decode_rapm_map(encoded_map) for encoded_map in valid[map_col]])
    weighted_total = (grids / 60.0 * toi[:, None, None]).sum(axis=0)
    return (weighted_total / total_toi * 60.0).astype(np.float32)


def rapm_four_maps(
    rapm,
    season,
    entity_id,
    group=None,
    metric='fenwick',
):
    if not group:
        if isinstance(entity_id, (int, np.integer)):
            group = 'skater'
        else:
            group = 'coach'

    seasons = season_values(season)
    if seasons is not None:
        rapm = rapm.loc[rapm['season'].isin(seasons)]

    group_id = 'coach_name' if group == 'coach' else 'player_id'
    player_rows = rapm.loc[rapm[group_id] == entity_id]

    strength_rows = {
        'even_strength': player_rows.loc[player_rows['strength_state'] == 'even_strength'],
        'power_play': player_rows.loc[player_rows['strength_state'] == 'power_play'],
        'penalty_kill': player_rows.loc[player_rows['strength_state'] == 'penalty_kill'],
    }

    ev_for = weighted_map(strength_rows['even_strength'], f'rapm_{metric}_for_per_sixty_map')
    ev_against = weighted_map(strength_rows['even_strength'], f'rapm_{metric}_against_per_sixty_map')
    st_for = weighted_map(strength_rows['power_play'], f'rapm_{metric}_for_per_sixty_map')
    st_against = weighted_map(strength_rows['penalty_kill'], f'rapm_{metric}_against_per_sixty_map')

    return ev_for, ev_against, st_for, st_against

def plot_rink_grid(left_grid, right_grid, left_label='Defense (Against)', right_label='Offense (For)'):
    combined = np.vstack([left_grid, right_grid])
    
    x = np.linspace(-100, 100, combined.shape[0])
    y = np.linspace(MAP_Y.min(), MAP_Y.max(), combined.shape[1])

    X, Y = np.meshgrid(x, y)

    limit = np.max(np.abs(combined))

    fig, ax = plt.subplots(1, 1, figsize=(10, 12), facecolor='black', edgecolor='black')
    ax.set_facecolor('black')

    wsba_rink('full')

    rwb = LinearSegmentedColormap.from_list(
        'rwb',
        [
            (0.0, (1.0, 0.0, 0.0)),
            (50.0 / 99.0, (1.0, 1.0, 1.0)),
            (1.0, (0.0, 0.0, 1.0)),
        ],
    )

    cont = ax.contourf(
        X, Y, combined.T,
        levels=np.linspace(-limit, limit, 51),
        cmap=rwb,
        alpha=0.7
    )

    cbar = fig.colorbar(
        cont,
        ax=ax,
        orientation='horizontal',
        shrink=0.75,
        aspect=50,
        fraction=0.05,
        pad=0.05,
        ticks=[-limit, 0 ,limit]
    )

    ax.text(-50, -50, left_label, ha='center', va='bottom', fontsize=12)
    ax.text(50, -50, right_label, ha='center', va='bottom', fontsize=12)

    clip_patch = mpatches.FancyBboxPatch(
        (-100, MAP_Y.min()),
        200, MAP_Y.max() - MAP_Y.min(),
        boxstyle="round,pad=0,rounding_size=30",
        transform=ax.transData
    )

    cont.set_clip_path(clip_patch)
    cbar.set_ticklabels(['Deficit', 'Even', 'Surplus'])

    return fig, ax, cbar

def plot_rapm_map(
    rapm,
    season,
    entity_id,
    group=None,
    metric='fenwick',
    strength='even_strength'
):  
    if not group:
        if type(entity_id) == int:
            group = 'skater'
        else:
            group = 'coach'

    plot_name = (
        entity_id if group == 'coach' else wsba.nhl_scrape_player_info(entity_id)['player_name'].iloc[0]
    )

    ev_for, ev_against, st_for, st_against = rapm_four_maps(
        rapm,
        season,
        entity_id,
        group=group,
        metric=metric,
    )

    if strength == 'special_teams':
        grid_for, grid_against = st_for, st_against
    else:
        grid_for, grid_against = ev_for, ev_against

    grid_for_smooth = gaussian_filter(grid_for, sigma=5)
    grid_against_smooth = gaussian_filter(grid_against, sigma=5)

    if strength == 'special_teams':
        fig, ax, cbar = plot_rink_grid(
            grid_against_smooth,
            grid_for_smooth,
            left_label='Penalty Kill',
            right_label='Power Play'
        )
    else:
        fig, ax, cbar = plot_rink_grid(grid_against_smooth, grid_for_smooth)

    cbar.set_label(f'Isolated {metric.replace("_", " ").title()}/60', fontsize=12)
    
    strengths_title = strength.replace("_", " ").title()
    seasons = season_values(season)
    season_label = seasons[0] if seasons and len(seasons) == 1 else f"{seasons[0]}-{seasons[-1]}" if seasons else "All Seasons"
    
    fig.text(0.51, 0.175, f'Strength(s)', ha='center', fontsize=10)
    fig.text(0.51, 0.155, f'{strengths_title}', ha='center', fontsize=10)
    ax.set_title(f'{plot_name}, {season_label}', fontsize=14)

    return fig
