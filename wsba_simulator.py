import wsba_hockey as wsba
import pandas as pd
import numpy as np
import sys
import math
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
import scipy.sparse as sp
import xgboost as xgb
from wsba_ml.nhl_obj import NHLTeam, COACH_SCORE_STRENGTH_PREFIXES
from wsba_ml.rapm import decode_rapm_map, rink_grid_format, MAP_X, MAP_Y, MAP_SHAPE, MAP_SIZE

### WSBA NHL Simulator ###
## Provided is the code for the WSBA NHL Game Simulation Engine ##

## SCALAR CONSTANTS
# These are league average per-sixty rates of respective metrics at specific strengths, if appended
EV_FENWICK = 41.569086
PP_FENWICK = 75.784143
PK_FENWICK = 12.293054
EN_FENWICK = 29.83037337783645
GOALS_PER_FENWICK = 0.064176
EV_CORSI = 59.78943931605245
PP_CORSI = 109.00146851959991
PK_CORSI = 17.681283782423215
EN_CORSI = 37.72137395073106
EV_EXPECTED_GOALS = GOALS_PER_FENWICK * EV_FENWICK
PP_EXPECTED_GOALS = GOALS_PER_FENWICK * PP_FENWICK
PK_EXPECTED_GOALS = GOALS_PER_FENWICK * PK_FENWICK
EN_EXPECTED_GOALS = GOALS_PER_FENWICK * EN_FENWICK

PENALTIES = 3.0702651883385172

RAPM_CONTEXT_PREDICTORS = {
    'period_2',
    'period_3',
    'period_4_plus',
    'offensive_faceoff',
    'defensive_faceoff',
    'lead_1',
    'lead_2',
    'lead_3_plus',
    'behind_1',
    'behind_2',
    'behind_3_plus',
    'shift_length',
    'power_play_expiry',
    'penalty_kill_expiry',
    'is_home',
    'strength_3v3',
    'strength_4v4',
    'strength_4v3',
    'strength_5v3',
}
XG_FEATURE_NAMES = tuple(wsba.CONTINUOUS + wsba.BOOLEAN)
XG_FEATURE_INDEX = {name: idx for idx, name in enumerate(XG_FEATURE_NAMES)}
XG_FEATURE_COUNT = len(XG_FEATURE_NAMES)
XG_MODEL_DIR = Path(__file__).resolve().parent / 'xg_model'
SIM_DATA_DIR = Path(__file__).resolve().parent / 'simulation' / 'sim_data'
XG_MODEL_FILES = {
    'ev': 'wsba_xg_ev.json',
    'pp': 'wsba_xg_pp.json',
    'pk': 'wsba_xg_pk.json',
    'en': 'wsba_xg_en.json',
}

FENWICK_GRID = pd.read_csv(SIM_DATA_DIR / 'league_fenwick_grid.csv')
FENWICK_GRID['value'] = FENWICK_GRID['value'] / 3600
GOALIE_PULL_CURVE = pd.read_csv(SIM_DATA_DIR / 'goalie_pull_curve.csv')

STRENGTHS_MATCH = {
    'even_strength': 'ev',
    'power_play': 'pp',
    'penalty_kill': 'pk',
    'empty_net': 'en',
}
STRENGTH_RATE_KEY_MAP = {
    strength_state: STRENGTHS_MATCH.get(strength_label, 'ev')
    for strength_state, strength_label in wsba.STRENGTH_MAP.items()
}


def _safe_rate(value):
    if not np.isfinite(value):
        return 0.0
    value = float(value)
    return value if value > 0.0 else 0.0

def _load_goalie_pull_curves():
    out = {}
    for score_state, group in GOALIE_PULL_CURVE.groupby('score_state', sort=False):
        probs = (
            pd.to_numeric(group.sort_values('second')['p_goalie_pulled'], errors='coerce')
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        probs = np.clip(probs, 0.0, 1.0)
        probs[probs < 0.10] = 0.0
        out[score_state] = probs

    for score_state in ('down_1', 'down_2', 'down_3', 'down_4_plus'):
        out.setdefault(score_state, np.array([], dtype=float))

    return out

GOALIE_PULL_CURVES = _load_goalie_pull_curves()
GOALIE_PULL_START_SECOND = {
    score_state: (int(np.flatnonzero(curve > 0.0)[0]) + 1 if curve.size and np.any(curve > 0.0) else None)
    for score_state, curve in GOALIE_PULL_CURVES.items()
}
GOALIE_PULL_MIN_START_SECOND = min(
    second for second in GOALIE_PULL_START_SECOND.values()
    if second is not None
)


def _xg_model_key(strength_state='even_strength', empty_net=False, rate_key=None):
    if empty_net:
        return 'en'
    if rate_key in {'ev', 'pp', 'pk'}:
        return rate_key
    strength_label = wsba.STRENGTH_MAP.get(strength_state, 'even_strength')
    return STRENGTHS_MATCH.get(strength_label, 'ev')


def _resolve_xg_model_path(model_key='ev'):
    model_file = XG_MODEL_FILES.get(model_key, XG_MODEL_FILES['ev'])
    model_path = XG_MODEL_DIR / model_file
    if not model_path.exists():
        raise FileNotFoundError(f'WSBA xG model not found: {model_path}')
    return model_path


@lru_cache(maxsize=4)
def _load_xg_features(model_key='ev'):
    model_path = _resolve_xg_model_path(model_key)
    feature_path = model_path.with_name(model_path.name.replace('.json', '_features.json'))
    if feature_path.exists():
        return tuple(json.loads(feature_path.read_text()))
    model = _load_xg_model(model_key)
    return tuple(model.feature_names or XG_FEATURE_NAMES)


@lru_cache(maxsize=4)
def _load_xg_feature_index(model_key='ev'):
    return {name: idx for idx, name in enumerate(_load_xg_features(model_key))}


@lru_cache(maxsize=4)
def _load_xg_model(model_key='ev'):
    booster = xgb.Booster()
    booster.load_model(str(_resolve_xg_model_path(model_key)))
    return booster


def _build_xg_feature_vector(feature_dict, model_key='ev'):
    features = _load_xg_features(model_key)
    feature_index = _load_xg_feature_index(model_key)
    vector = np.zeros((1, len(features)), dtype=np.float32)
    for feature_name, value in feature_dict.items():
        idx = feature_index.get(feature_name)
        if idx is not None:
            vector[0, idx] = float(value)
    return vector, features


def predict_single_shot(feature_dict: dict, empty_net: bool = False, model_path: str = wsba.XG_MODEL, model_key: str = 'ev') -> float:
    if empty_net:
        model_key = 'en'
    model = _load_xg_model(model_key)
    feature_vector, features = _build_xg_feature_vector(feature_dict, model_key)
    try:
        return float(model.inplace_predict(feature_vector, validate_features=False)[0])
    except Exception:
        dmatrix = xgb.DMatrix(data=sp.csr_matrix(feature_vector), feature_names=list(features))
        return float(model.predict(dmatrix)[0])


def load_team_from_roster_csv(team_abbr, roster_dir=None):
    if roster_dir is None:
        roster_dir = Path(__file__).resolve().parent / 'simulation' / 'team_data'
    else:
        roster_dir = Path(roster_dir)
    roster_path = roster_dir / f'{str(team_abbr).lower()}_roster.csv'
    if not roster_path.exists():
        raise FileNotFoundError(f'Roster file not found: {roster_path}')
    return NHLTeam(team_abbr).roster_from_df(pd.read_csv(roster_path))


@lru_cache(maxsize=2)
def _load_rapm_context_coefficients(model_name):
    model_file = {
        'even_strength': 'wsba_rapm_even_strength.json',
        'power_play': 'wsba_rapm_power_play.json',
    }[model_name]
    model_path = Path(__file__).resolve().parent / 'rapm' / model_file
    if not model_path.exists():
        return {}
    model_data = json.loads(model_path.read_text())['shift']
    return {
        predictor: float(coef)
        for predictor, coef in zip(model_data['corsi_predictors'], model_data['corsi'])
        if predictor in RAPM_CONTEXT_PREDICTORS
    }


FAST_CONTEXT_TERMS = {
    'even_strength': _load_rapm_context_coefficients('even_strength'),
    'power_play': _load_rapm_context_coefficients('power_play'),
}

GAME_SUMMARY_COLUMNS = [
    'trial',
    'series_trial',
    'series_winner',
    'game_id',
    'home_team_abbr',
    'away_team_abbr',
    'home_score',
    'away_score',
    'home_fenwick',
    'away_fenwick',
    'home_expected_goals',
    'away_expected_goals',
    'time_on_ice',
    'winner',
    'result',
]

PBP_COLUMNS = [
    'game_id',
    'period',
    'seconds_elapsed',
    'period_seconds_elapsed',
    'event_num',
    'strength_state',
    'event_type',
    'event_team_venue',
    'event_team_abbr',
    'event_player_1_id',
    'event_player_2_id',
    'event_player_3_id',
    'event_player_1_hand',
    'event_goalie_id',
    'penalty_duration',
    'penalty_type',
    'shot_type',
    'event_reason',
    'x_adj',
    'y_adj',
    'xG',
    'home_score',
    'away_score',
    'home_fenwick',
    'away_fenwick',
    'home_expected_goals',
    'away_expected_goals',
]

XG_IDX_EVENT_DISTANCE = XG_FEATURE_INDEX.get('event_distance')
XG_IDX_EVENT_ANGLE = XG_FEATURE_INDEX.get('event_angle')
XG_IDX_IS_HOME = XG_FEATURE_INDEX.get('is_home')
XG_IDX_EMPTY_NET = XG_FEATURE_INDEX.get('empty_net')
XG_IDX_OFFWING = XG_FEATURE_INDEX.get('offwing')
XG_IDX_OTHER_SHOT = XG_FEATURE_INDEX.get('other-shot')
XG_STRENGTH_FEATURE_IDX = {
    strength: XG_FEATURE_INDEX.get(f'strength_{strength}')
    for strength in wsba.STRENGTHS
}
XG_SHOT_TYPE_FEATURE_IDX = {
    shot_type: XG_FEATURE_INDEX[shot_type]
    for shot_type in wsba.SHOT_TYPES
    if shot_type in XG_FEATURE_INDEX
}
SHOT_SAMPLING_ATTRIBUTES = (
    'shot_rate',
    'block_rate',
    'primary_pass_rate',
    'secondary_pass_rate',
    'penalties_taken_rate',
    'penalties_drawn_rate',
)
EVENT_NO = 0
EVENT_AWAY_CORSI = 1
EVENT_HOME_CORSI = 2
EVENT_AWAY_PENALTY = 3
EVENT_HOME_PENALTY = 4
XG_MODEL_CACHE = {}
XG_BASE_PREDICTION_CACHE = {}

def _prepare_grid_sampler(grid_df):
    x = grid_df['x'].to_numpy(dtype=np.float32)
    y = grid_df['y'].to_numpy(dtype=np.float32)
    raw_weights = grid_df['value'].to_numpy(dtype=np.float64)
    total = float(raw_weights.sum())

    if total <= 0.0:
        weights = np.full(len(raw_weights), 1.0 / max(1, len(raw_weights)), dtype=np.float64)
        cdf = np.linspace(1.0 / max(1, len(raw_weights)), 1.0, max(1, len(raw_weights)), dtype=np.float64)
    else:
        weights = raw_weights / total
        cdf = np.cumsum(weights, dtype=np.float64)
        cdf[-1] = 1.0

    return {
        'x': x,
        'y': y,
        'weights': weights,
        'raw_total': total,
        'cdf': cdf,
        'size': len(x),
    }


def _score_diff_for_venue(venue, home_score, away_score):
    return home_score - away_score if venue == 'home' else away_score - home_score


def _score_predictor(score_diff):
    score_diff = int(score_diff)
    if score_diff >= 3:
        return 'lead_3_plus'
    if score_diff == 2:
        return 'lead_2'
    if score_diff == 1:
        return 'lead_1'
    if score_diff <= -3:
        return 'behind_3_plus'
    if score_diff == -2:
        return 'behind_2'
    if score_diff == -1:
        return 'behind_1'
    return None


def _coach_score_state(score_diff):
    score_state = _score_predictor(score_diff)
    return score_state if score_state is not None else 'tied'


def _coach_score_strength_prefix(role):
    if role == 'pp':
        return COACH_SCORE_STRENGTH_PREFIXES['power_play']
    if role == 'pk':
        return COACH_SCORE_STRENGTH_PREFIXES['penalty_kill']
    return COACH_SCORE_STRENGTH_PREFIXES['even_strength']


def _coach_score_state_adjustment_per60(team, score_state, metric, side, role):
    coach = getattr(team, 'coach', None)
    if coach is None:
        return 0.0

    column = f'rapm_{score_state}_{metric}_{side}_per_sixty'
    prefix = _coach_score_strength_prefix(role)
    value = getattr(coach, f'{prefix}_{column}', None)
    if value is None or pd.isna(value):
        value = getattr(coach, column, 0.0)
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _period_predictors(period_number, reg_periods, season_type):
    if period_number == 2:
        return ('period_2',)
    if period_number == 3:
        return ('period_3',)
    if period_number >= 4:
        return ('period_4_plus',)
    return ()


def _strength_predictor(model_name, venue, role, home_skaters, away_skaters):
    offense_skaters = home_skaters if venue == 'home' else away_skaters
    defense_skaters = away_skaters if venue == 'home' else home_skaters

    if model_name == 'even_strength':
        return {
            (3, 3): 'strength_3v3',
            (4, 4): 'strength_4v4',
        }.get((offense_skaters, defense_skaters))

    if model_name == 'power_play' and role == 'pp':
        return {
            (4, 3): 'strength_4v3',
            (5, 3): 'strength_5v3',
        }.get((offense_skaters, defense_skaters))

    return None


def _prepare_player_sampler(players, attribute):
    ids = np.array([int(getattr(player, 'player_id', 0) or 0) for player in players], dtype=np.int64)
    weights = np.array(
        [_player_event_weight(player, attribute) for player in players],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= 0.0:
        if len(players) <= 0:
            weights = np.array([], dtype=np.float64)
            cdf = np.array([], dtype=np.float64)
        else:
            weights = np.full(len(players), 1.0 / len(players), dtype=np.float64)
            cdf = np.linspace(1.0 / len(players), 1.0, len(players), dtype=np.float64)
    else:
        weights /= total
        cdf = np.cumsum(weights, dtype=np.float64)
        cdf[-1] = 1.0

    return {
        'players': tuple(players),
        'ids': ids,
        'weights': weights,
        'cdf': cdf,
        'size': len(players),
    }


def _player_event_weight(player, attribute):
    # Event rates describe talent/tendency; projected TOI controls opportunity.
    rate = max(float(getattr(player, attribute, 0) or 0.0), 0.0)
    toi = max(float(getattr(player, 'time_on_ice', 0) or 0.0), 0.0)
    return rate * toi


def _load_cached_xg_model(model_key):
    cache_key = str(model_key)
    model = XG_MODEL_CACHE.get(cache_key)
    if model is None:
        model = _load_xg_model(cache_key)
        XG_MODEL_CACHE[cache_key] = model
    return model

# Calculate team component measure
def _team_component(team: NHLTeam, metric: str, toi: str, map: bool = False):
    # Build team components as projected-TOI weighted sums of skater components.
    skaters = [p for p in team.active_roster if p.position != 'G']
    is_map_metric = map or metric.endswith('_map')

    def _rapm_grid(value):
        if isinstance(value, np.ndarray):
            return value.astype(np.float32, copy=False)
        if isinstance(value, str) and value:
            return decode_rapm_map(value)
        return np.zeros(MAP_SHAPE, dtype=np.float32)

    if is_map_metric:
        weighted_sum = np.zeros(MAP_SHAPE, dtype=np.float32)

        for player in skaters:
            player_map = _rapm_grid(getattr(player, metric, None))
            weight = float(getattr(player, toi, 0) or 0.0)
            if weight <= 0:
                continue
            weighted_sum += (player_map / 60.0) * weight

        return weighted_sum.astype(np.float32, copy=False)

    weighted_sum = 0.0

    for player in skaters:
        value = getattr(player, metric, 0)
        value = 0.0 if value is None or pd.isna(value) else float(value)
        weight = float(getattr(player, toi, 0) or 0.0)
        if weight <= 0:
            continue
        weighted_sum += (value / 60.0) * weight

    return float(weighted_sum)

# Create sim grid components
def _calculate_grid(league_grid, off_map, def_map):
    # Flip map coordinates for defensive team (to place on top of offense map which is from x -> [0, 99])
    def_map = def_map[::-1, ::-1]

    combined_map = (off_map + def_map) / 2
    combined_grid = rink_grid_format(combined_map, "to_df")

    component_grid = league_grid.merge(
        combined_grid[["x", "y", "value"]],
        on=["x", "y"],
        how="left"
    ).fillna(0)

    # Convert to per-second probability of a shot occuring at that particular coordinate
    component_grid['value'] = (component_grid['value_x'] + component_grid['value_y'] / 3600)
    component_grid['value'] = np.where(component_grid['value'] < 0, 0, component_grid['value'])

    return component_grid[['x', 'y', 'value']]

def _single_shot_xg(x_adj, y_adj, shooting_venue, strength_state, shooter=None, goalie=None, shot_type=None):
    # Use the trained WSBA xG model with default shot coordinates oriented toward the right-side net.
    x_adj = float(x_adj)
    y_adj = float(y_adj)
    goal_x = 89.0
    dx = goal_x - x_adj

    features = {
        'event_distance': float(np.sqrt((dx * dx) + (y_adj * y_adj))),
        'event_angle': float(np.degrees(np.arctan2(abs(y_adj), max(1e-9, abs(dx))))),
        'is_home': float(shooting_venue == 'home'),
        'empty_net': float(goalie is None),
        'offwing': float(
            ((y_adj < 0) and getattr(shooter, 'handedness', None) == 'L')
            or ((y_adj >= 0) and getattr(shooter, 'handedness', None) == 'R')
        ),
    }

    for strength in wsba.STRENGTHS:
        features[f'strength_{strength}'] = float(strength_state == strength)

    if shot_type in wsba.SHOT_TYPES:
        features[shot_type] = 1.0
    else:
        features['other-shot'] = 1.0

    model_key = _xg_model_key(strength_state, empty_net=(goalie is None))
    return float(np.clip(
        predict_single_shot(features, empty_net=(goalie is None), model_key=model_key),
        0.0,
        1.0
    ))

class WSBASimGame:
    __slots__ = (
        'game_id',
        'season_type',
        'strength_state',
        'seconds_elapsed',
        'home_score',
        'away_score',
        'home_fenwick',
        'away_fenwick',
        'home_expected_goals',
        'away_expected_goals',
        'period',
        'period_seconds_elapsed',
        'event_num',
        'last_penalty_second',
        'last_shot_second',
        'last_shot_team_venue',
        'away_penalties',
        'home_penalties',
        'away_boxed_ids',
        'home_boxed_ids',
        'away_goalie_pulled',
        'home_goalie_pulled',
        'away_goalie_pull_cooldown',
        'home_goalie_pull_cooldown',
        'away_skaters_on_ice',
        'home_skaters_on_ice',
        'home_power_play_expiry',
        'away_power_play_expiry',
        'home_penalty_kill_expiry',
        'away_penalty_kill_expiry',
    )

    def __init__(self, game_id: int = 0, season_type: int = 2):
        # Properties of a sim game

        self.game_id = game_id
        self.season_type = season_type
        self.strength_state = '5v5'
        self.seconds_elapsed = 0
        self.home_score = 0
        self.away_score = 0
        self.home_fenwick = 0
        self.away_fenwick = 0
        self.home_expected_goals = 0.0
        self.away_expected_goals = 0.0
        self.period = 0
        self.period_seconds_elapsed = 0
        self.event_num = 0
        self.last_penalty_second = -9999
        self.last_shot_second = -9999
        self.last_shot_team_venue = None
        self.away_penalties = []
        self.home_penalties = []
        self.away_boxed_ids = set()
        self.home_boxed_ids = set()
        self.away_goalie_pulled = False
        self.home_goalie_pulled = False
        self.away_goalie_pull_cooldown = 0
        self.home_goalie_pull_cooldown = 0
        self.away_skaters_on_ice = 5
        self.home_skaters_on_ice = 5
        self.home_power_play_expiry = 0
        self.away_power_play_expiry = 0
        self.home_penalty_kill_expiry = 0
        self.away_penalty_kill_expiry = 0

class WSBASimulation:
    def __init__(
        self,
        home: NHLTeam,
        away: NHLTeam,
        season_type: int = 2,
        reg_periods: int = 3,
        reg_minutes: int = 20
    ):
        ## SIMULATION ENGINE DATA ##
        self.season_type = season_type
        self.reg_periods = reg_periods
        self.reg_seconds = reg_minutes * 60
        self.shootout = (season_type == 2)
        self.ot_seconds = 300 if self.shootout else self.reg_seconds

        ## SIMULATION ENGINE PARAMETERS ##
        self.ev_fenwick = EV_FENWICK / 3600
        self.pp_fenwick = PP_FENWICK / 3600
        self.pk_fenwick = PK_FENWICK / 3600
        self.en_fenwick = EN_FENWICK / 3600
        self.ev_corsi = EV_CORSI / 3600
        self.pp_corsi = PP_CORSI / 3600
        self.pk_corsi = PK_CORSI / 3600
        self.en_corsi = EN_CORSI / 3600
        self.penalties = PENALTIES / 7200

        self.ev_fenwick_map = FENWICK_GRID.loc[FENWICK_GRID['strength_state']=='even_strength']
        self.pp_fenwick_map = FENWICK_GRID.loc[FENWICK_GRID['strength_state']=='power_play']
        self.pk_fenwick_map = FENWICK_GRID.loc[FENWICK_GRID['strength_state']=='penalty_kill']
        self.en_fenwick_map = FENWICK_GRID.loc[FENWICK_GRID['strength_state']=='empty_net']

        ## SIMULATION ENGINE COMPONENTS ##
        self.home, self.away = home, away # Assign simulation engine teams
        self.home_skaters = [p for p in self.home.active_roster if p.position != 'G']
        self.away_skaters = [p for p in self.away.active_roster if p.position != 'G']
        self.home_player_lookup = {p.player_id: p for p in self.home_skaters}
        self.away_player_lookup = {p.player_id: p for p in self.away_skaters}
        self._player_sampling = {
            'home': {
                attribute: _prepare_player_sampler(self.home_skaters, attribute)
                for attribute in SHOT_SAMPLING_ATTRIBUTES
            },
            'away': {
                attribute: _prepare_player_sampler(self.away_skaters, attribute)
                for attribute in SHOT_SAMPLING_ATTRIBUTES
            },
        }
        self.home_goalie = next((p for p in self.home.active_roster if p.position == 'G'), None)
        self.away_goalie = next((p for p in self.away.active_roster if p.position == 'G'), None)

        # Combine the team for and against rates (both ways) to find the per-second probabilities of each team recording a fenwick shot or a penalty
        self.ev_corsi_home = self.ev_corsi + (
            _team_component(home, 'ev_corsi_offense', 'ev_time_on_ice')
            + _team_component(away, 'ev_corsi_defense', 'ev_time_on_ice')
        ) / 7200
        self.ev_corsi_away = self.ev_corsi + (
            _team_component(home, 'ev_corsi_defense', 'ev_time_on_ice')
            + _team_component(away, 'ev_corsi_offense', 'ev_time_on_ice')
        ) / 7200
        self.pp_corsi_home = self.pp_corsi + (
            _team_component(home, 'pp_corsi_offense', 'pp_time_on_ice')
            + _team_component(away, 'pk_corsi_defense', 'pk_time_on_ice')
        ) / 7200
        self.pp_corsi_away = self.pp_corsi + (
            _team_component(home, 'pk_corsi_defense', 'pk_time_on_ice')
            + _team_component(away, 'pp_corsi_offense', 'pp_time_on_ice')
        ) / 7200
        self.ev_corsi_home = _safe_rate(self.ev_corsi_home)
        self.ev_corsi_away = _safe_rate(self.ev_corsi_away)
        self.pp_corsi_home = _safe_rate(self.pp_corsi_home)
        self.pp_corsi_away = _safe_rate(self.pp_corsi_away)

        self.ev_fenwick_map_home = _calculate_grid(self.ev_fenwick_map, 
            _team_component(home, 'ev_fenwick_offense_map', 'ev_time_on_ice', map=True),
            _team_component(away, 'ev_fenwick_defense_map', 'ev_time_on_ice', map=True)
        )
        self.ev_fenwick_map_away = _calculate_grid(self.ev_fenwick_map, 
            _team_component(away, 'ev_fenwick_offense_map', 'ev_time_on_ice', map=True),
            _team_component(home, 'ev_fenwick_defense_map', 'ev_time_on_ice', map=True)
        )
        self.pp_fenwick_map_home = _calculate_grid(self.pp_fenwick_map, 
            _team_component(home, 'pp_fenwick_offense_map', 'pp_time_on_ice', map=True),
            _team_component(away, 'pk_fenwick_defense_map', 'pk_time_on_ice', map=True)
        )
        self.pp_fenwick_map_away = _calculate_grid(self.pp_fenwick_map, 
            _team_component(away, 'pp_fenwick_offense_map', 'pp_time_on_ice', map=True),
            _team_component(home, 'pk_fenwick_defense_map', 'pk_time_on_ice', map=True)
        )

        self.ev_expected_goals_home = _safe_rate(
            (EV_EXPECTED_GOALS / 3600.0) + (
                _team_component(home, 'ev_xg_offense', 'ev_time_on_ice')
                + _team_component(away, 'ev_xg_defense', 'ev_time_on_ice')
            ) / 7200.0
        )
        self.ev_expected_goals_away = _safe_rate(
            (EV_EXPECTED_GOALS / 3600.0) + (
                _team_component(home, 'ev_xg_defense', 'ev_time_on_ice')
                + _team_component(away, 'ev_xg_offense', 'ev_time_on_ice')
            ) / 7200.0
        )
        self.pp_expected_goals_home = _safe_rate(
            (PP_EXPECTED_GOALS / 3600.0) + (
                _team_component(home, 'pp_xg_offense', 'pp_time_on_ice')
                + _team_component(away, 'pk_xg_defense', 'pk_time_on_ice')
            ) / 7200.0
        )
        self.pp_expected_goals_away = _safe_rate(
            (PP_EXPECTED_GOALS / 3600.0) + (
                _team_component(home, 'pk_xg_defense', 'pk_time_on_ice')
                + _team_component(away, 'pp_xg_offense', 'pp_time_on_ice')
            ) / 7200.0
        )
        
        # Use league-average for penalty-kill offense rates
        self.pk_corsi_away = self.pk_fenwick
        self.pk_corsi_home = self.pk_fenwick
        self.pk_fenwick_map_away = self.pk_fenwick_map
        self.pk_fenwick_map_home = self.pk_fenwick_map
        self.pk_expected_goals_away = PK_EXPECTED_GOALS / 3600.0
        self.pk_expected_goals_home = PK_EXPECTED_GOALS / 3600.0
        self.en_corsi_away = self.en_corsi
        self.en_corsi_home = self.en_corsi
        self.en_fenwick_map_away = self.en_fenwick_map
        self.en_fenwick_map_home = self.en_fenwick_map
        self.en_expected_goals_away = EN_EXPECTED_GOALS / 3600.0
        self.en_expected_goals_home = EN_EXPECTED_GOALS / 3600.0

        self._corsi_component = {
            'home': {
                'ev': self.ev_corsi_home,
                'pp': self.pp_corsi_home,
                'pk': self.pk_corsi_home,
                'en': self.en_corsi_home,
            },
            'away': {
                'ev': self.ev_corsi_away,
                'pp': self.pp_corsi_away,
                'pk': self.pk_corsi_away,
                'en': self.en_corsi_away,
            },
        }
        self._fenwick_sampler = {
            'home': {
                'ev': _prepare_grid_sampler(self.ev_fenwick_map_home),
                'pp': _prepare_grid_sampler(self.pp_fenwick_map_home),
                'pk': _prepare_grid_sampler(self.pk_fenwick_map_home),
                'en': _prepare_grid_sampler(self.en_fenwick_map_home),
            },
            'away': {
                'ev': _prepare_grid_sampler(self.ev_fenwick_map_away),
                'pp': _prepare_grid_sampler(self.pp_fenwick_map_away),
                'pk': _prepare_grid_sampler(self.pk_fenwick_map_away),
                'en': _prepare_grid_sampler(self.en_fenwick_map_away),
            },
        }
        self._xg_target_per_fenwick = {
            'home': {
                'ev': float(max(self.ev_expected_goals_home / max(self._fenwick_sampler['home']['ev']['raw_total'], 1e-9), 0.0)),
                'pp': float(max(self.pp_expected_goals_home / max(self._fenwick_sampler['home']['pp']['raw_total'], 1e-9), 0.0)),
                'pk': float(max(self.pk_expected_goals_home / max(self._fenwick_sampler['home']['pk']['raw_total'], 1e-9), 0.0)),
                'en': float(max(self.en_expected_goals_home / max(self._fenwick_sampler['home']['en']['raw_total'], 1e-9), 0.0)),
            },
            'away': {
                'ev': float(max(self.ev_expected_goals_away / max(self._fenwick_sampler['away']['ev']['raw_total'], 1e-9), 0.0)),
                'pp': float(max(self.pp_expected_goals_away / max(self._fenwick_sampler['away']['pp']['raw_total'], 1e-9), 0.0)),
                'pk': float(max(self.pk_expected_goals_away / max(self._fenwick_sampler['away']['pk']['raw_total'], 1e-9), 0.0)),
                'en': float(max(self.en_expected_goals_away / max(self._fenwick_sampler['away']['en']['raw_total'], 1e-9), 0.0)),
            },
        }
        self._xg_lookup_cache = {}
        self._event_rate_cache = {}

        self.penalties_home = self.penalties + (
            _team_component(home, 'taken_pens', 'time_on_ice')
            + _team_component(away, 'drawn_pens', 'time_on_ice')
        ) / 7200
        self.penalties_away = self.penalties + (
            _team_component(home, 'drawn_pens', 'time_on_ice')
            + _team_component(away, 'taken_pens', 'time_on_ice')
        ) / 7200
        self.penalties_home = _safe_rate(self.penalties_home)
        self.penalties_away = _safe_rate(self.penalties_away)

    ## CORE SIMULATION ENGINE ##
    def simulate_game(self, game_id = 0, save_pbp = True):
        # Initilize the game
        sim_game = WSBASimGame(game_id, self.season_type)

        season_type = self.season_type
        reg_periods = self.reg_periods
        home_team_abbr = self.home.team_abbr
        away_team_abbr = self.away.team_abbr
        home_skaters = self.home_skaters
        away_skaters = self.away_skaters
        home_player_lookup = self.home_player_lookup
        away_player_lookup = self.away_player_lookup
        home_goalie = self.home_goalie
        away_goalie = self.away_goalie
        corsi_component = self._corsi_component
        fenwick_sampler = self._fenwick_sampler
        xg_target_per_fenwick = self._xg_target_per_fenwick
        player_sampling = self._player_sampling
        xg_lookup_cache = self._xg_lookup_cache
        event_rate_cache = self._event_rate_cache
        random_random = np.random.random
        random_randint = np.random.randint
        penalties_home_rate = float(self.penalties_home) if np.isfinite(self.penalties_home) and self.penalties_home > 0 else 0.0
        penalties_away_rate = float(self.penalties_away) if np.isfinite(self.penalties_away) and self.penalties_away > 0 else 0.0

        game_events = [] if save_pbp else None
        xg_models = {model_key: _load_cached_xg_model(model_key) for model_key in XG_MODEL_FILES}

        def _sample_prepared(prepared, return_player=True):
            if prepared['size'] <= 0:
                return None
            idx = int(np.searchsorted(prepared['cdf'], random_random(), side='right'))
            if idx >= prepared['size']:
                idx = prepared['size'] - 1
            if return_player:
                return prepared['players'][idx]
            return int(prepared['ids'][idx])

        def _weighted_choice_from_players(players, attribute, return_player=True):
            if not players:
                return None

            weights = []
            total = 0.0
            for player in players:
                weight = _player_event_weight(player, attribute)
                weights.append(weight)
                total += weight

            if total <= 0.0:
                idx = int(random_randint(len(players)))
            else:
                target = float(random_random()) * total
                running = 0.0
                idx = len(players) - 1
                for candidate_idx, weight in enumerate(weights):
                    running += weight
                    if running >= target:
                        idx = candidate_idx
                        break

            if return_player:
                return players[idx]
            return int(getattr(players[idx], 'player_id', 0) or 0)

        def _refresh_team_penalties(team_venue):
            if team_venue == 'away':
                penalties = sim_game.away_penalties
                boxed_ids = sim_game.away_boxed_ids
            else:
                penalties = sim_game.home_penalties
                boxed_ids = sim_game.home_boxed_ids

            idx = 0
            while idx < len(penalties):
                if penalties[idx][1] > 0:
                    idx += 1
                    continue
                boxed_ids.discard(penalties[idx][0])
                penalties.pop(idx)

        def _tick_penalty_box():
            # Advance active penalties by one second.
            if not sim_game.away_penalties and not sim_game.home_penalties:
                return False

            changed = False
            away_active = min(2, len(sim_game.away_penalties))
            for idx in range(away_active):
                sim_game.away_penalties[idx][1] -= 1
            if away_active:
                before = len(sim_game.away_penalties)
                _refresh_team_penalties('away')
                changed = changed or (len(sim_game.away_penalties) != before)

            home_active = min(2, len(sim_game.home_penalties))
            for idx in range(home_active):
                sim_game.home_penalties[idx][1] -= 1
            if home_active:
                before = len(sim_game.home_penalties)
                _refresh_team_penalties('home')
                changed = changed or (len(sim_game.home_penalties) != before)

            return changed

        def _update_strength_state():
            # Strength state is based on the number of active penalties per team
            previous_strength_state = sim_game.strength_state
            previous_home_skaters = sim_game.home_skaters_on_ice
            previous_away_skaters = sim_game.away_skaters_on_ice

            away_penalties = min(2, len(sim_game.away_penalties))
            home_penalties = min(2, len(sim_game.home_penalties))

            is_regular_season_ot = season_type == 2 and sim_game.period > reg_periods
            if is_regular_season_ot:
                away_base = min(5, 3 + home_penalties)
                home_base = min(5, 3 + away_penalties)
            else:
                away_base = max(3, 5 - away_penalties)
                home_base = max(3, 5 - home_penalties)

            away_skaters = min(6, away_base + int(sim_game.away_goalie_pulled))
            home_skaters = min(6, home_base + int(sim_game.home_goalie_pulled))
            sim_game.away_skaters_on_ice = away_skaters
            sim_game.home_skaters_on_ice = home_skaters
            sim_game.strength_state = f'{away_skaters}v{home_skaters}'

            previous_strength_label = wsba.STRENGTH_MAP.get(previous_strength_state)
            current_strength_label = wsba.STRENGTH_MAP.get(sim_game.strength_state)
            if previous_strength_label != 'even_strength' and current_strength_label == 'even_strength':
                prev_home_role = _venue_role('home', previous_strength_state, previous_home_skaters, previous_away_skaters)
                if prev_home_role == 'pp':
                    sim_game.home_power_play_expiry = 10
                    sim_game.away_penalty_kill_expiry = 10
                elif prev_home_role == 'pk':
                    sim_game.home_penalty_kill_expiry = 10
                    sim_game.away_power_play_expiry = 10

        def _eligible_skaters(team_venue):
            # Eligible skaters exclude all players currently serving or waiting in the box
            if team_venue == 'home':
                boxed_ids = sim_game.home_boxed_ids
                team_skaters = home_skaters
            else:
                boxed_ids = sim_game.away_boxed_ids
                team_skaters = away_skaters

            if not boxed_ids:
                return team_skaters
            return [
                p for p in team_skaters
                if p.player_id not in boxed_ids
            ]

        def _active_goalie(team_venue):
            if team_venue == 'home':
                if sim_game.home_goalie_pulled:
                    return None
                return home_goalie
            if sim_game.away_goalie_pulled:
                return None
            return away_goalie

        def _score_deficit(team_venue):
            if team_venue == 'home':
                return sim_game.away_score - sim_game.home_score
            return sim_game.home_score - sim_game.away_score

        def _goalie_pull_probability(second, deficit):
            if deficit <= 0:
                return 0.0

            score_state = f'down_{deficit}' if deficit < 4 else 'down_4_plus'
            start_second = GOALIE_PULL_START_SECOND.get(score_state)
            if start_second is None or second < start_second:
                return 0.0
            curve = GOALIE_PULL_CURVES.get(score_state)
            if curve is None or curve.size == 0:
                return 0.0

            idx = max(0, int(second) - 1)
            if idx < curve.size:
                return float(curve[idx])
            return float(curve[-1])

        def _apply_goalie_pull_state():
            changed = False

            if (
                sim_game.seconds_elapsed < GOALIE_PULL_MIN_START_SECOND
                and sim_game.home_goalie_pull_cooldown <= 0
                and sim_game.away_goalie_pull_cooldown <= 0
                and not sim_game.home_goalie_pulled
                and not sim_game.away_goalie_pulled
            ):
                return False

            if sim_game.home_goalie_pull_cooldown > 0:
                sim_game.home_goalie_pulled = False
                sim_game.home_goalie_pull_cooldown -= 1
            elif not sim_game.home_goalie_pulled:
                home_deficit = _score_deficit('home')
                if home_deficit > 0:
                    pulled = np.random.random() < _goalie_pull_probability(sim_game.seconds_elapsed, home_deficit)
                    if pulled != sim_game.home_goalie_pulled:
                        sim_game.home_goalie_pulled = pulled
                        changed = True

            if sim_game.away_goalie_pull_cooldown > 0:
                sim_game.away_goalie_pulled = False
                sim_game.away_goalie_pull_cooldown -= 1
            elif not sim_game.away_goalie_pulled:
                away_deficit = _score_deficit('away')
                if away_deficit > 0:
                    pulled = np.random.random() < _goalie_pull_probability(sim_game.seconds_elapsed, away_deficit)
                    if pulled != sim_game.away_goalie_pulled:
                        sim_game.away_goalie_pulled = pulled
                        changed = True

            return changed

        def _empty_net_team_venue():
            if sim_game.home_goalie_pulled:
                return 'home'
            if sim_game.away_goalie_pulled:
                return 'away'
            return None

        def _shot_cooldown(team_venue):
            # Apply cooldowns around penalties, period starts, and opponent shot attempts
            if sim_game.period_seconds_elapsed <= 5:
                return True
            if (sim_game.seconds_elapsed - sim_game.last_penalty_second) <= 2:
                return True
            if (
                _empty_net_team_venue() is None
                and sim_game.last_shot_team_venue is not None
                and sim_game.last_shot_team_venue != team_venue
                and (sim_game.seconds_elapsed - sim_game.last_shot_second) <= 5
            ):
                return True
            return False

        def _shot_rate_key(team_venue, strength_state):
            empty_net_team_venue = _empty_net_team_venue()
            if empty_net_team_venue is not None:
                if team_venue == empty_net_team_venue:
                    return 'pp'
                return 'en'

            strength_label = wsba.STRENGTH_MAP.get(strength_state, 'even_strength')
            return STRENGTHS_MATCH[strength_label]

        def _venue_role(team_venue, strength_state, home_skaters_count=None, away_skaters_count=None):
            home_skaters_count = sim_game.home_skaters_on_ice if home_skaters_count is None else home_skaters_count
            away_skaters_count = sim_game.away_skaters_on_ice if away_skaters_count is None else away_skaters_count

            if sim_game.home_goalie_pulled or sim_game.away_goalie_pulled:
                empty_net_team_venue = _empty_net_team_venue()
                if empty_net_team_venue is None:
                    return 'ev'
                return 'pp' if team_venue == empty_net_team_venue else 'en'

            strength_label = wsba.STRENGTH_MAP.get(strength_state, 'even_strength')
            if strength_label == 'power_play':
                return 'pp' if ((team_venue == 'home' and home_skaters_count > away_skaters_count) or (team_venue == 'away' and away_skaters_count > home_skaters_count)) else 'pk'
            if strength_label == 'penalty_kill':
                return 'pk' if ((team_venue == 'home' and home_skaters_count < away_skaters_count) or (team_venue == 'away' and away_skaters_count < home_skaters_count)) else 'pp'
            return 'ev'

        def _tick_context_expiry():
            if sim_game.home_power_play_expiry > 0:
                sim_game.home_power_play_expiry -= 1
            if sim_game.away_power_play_expiry > 0:
                sim_game.away_power_play_expiry -= 1
            if sim_game.home_penalty_kill_expiry > 0:
                sim_game.home_penalty_kill_expiry -= 1
            if sim_game.away_penalty_kill_expiry > 0:
                sim_game.away_penalty_kill_expiry -= 1

        def _context_adjustment_per60(model_name, venue, role):
            coeffs = FAST_CONTEXT_TERMS[model_name]
            adjustment = 0.0

            for predictor in _period_predictors(sim_game.period, reg_periods, season_type):
                adjustment += coeffs.get(predictor, 0.0)

            score_predictor = _score_predictor(_score_diff_for_venue(venue, sim_game.home_score, sim_game.away_score))
            if score_predictor is not None:
                adjustment += coeffs.get(score_predictor, 0.0)

            if venue == 'home':
                adjustment += coeffs.get('is_home', 0.0)

            if venue == 'home':
                if sim_game.home_power_play_expiry > 0:
                    adjustment += coeffs.get('power_play_expiry', 0.0)
                if sim_game.home_penalty_kill_expiry > 0:
                    adjustment += coeffs.get('penalty_kill_expiry', 0.0)
            else:
                if sim_game.away_power_play_expiry > 0:
                    adjustment += coeffs.get('power_play_expiry', 0.0)
                if sim_game.away_penalty_kill_expiry > 0:
                    adjustment += coeffs.get('penalty_kill_expiry', 0.0)

            strength_predictor = _strength_predictor(
                model_name,
                venue,
                role,
                sim_game.home_skaters_on_ice,
                sim_game.away_skaters_on_ice,
            )
            if strength_predictor is not None:
                adjustment += coeffs.get(strength_predictor, 0.0)

            return adjustment

        def _coach_context_adjustment_per60(venue, role):
            offense_score_state = _coach_score_state(
                _score_diff_for_venue(venue, sim_game.home_score, sim_game.away_score)
            )

            if venue == 'home':
                offense_team = home
                defense_team = away
                defense_venue = 'away'
                defense_role = _venue_role('away', sim_game.strength_state)
            else:
                offense_team = away
                defense_team = home
                defense_venue = 'home'
                defense_role = _venue_role('home', sim_game.strength_state)
            defense_score_state = _coach_score_state(
                _score_diff_for_venue(defense_venue, sim_game.home_score, sim_game.away_score)
            )

            return (
                _coach_score_state_adjustment_per60(offense_team, offense_score_state, 'corsi', 'for', role)
                + _coach_score_state_adjustment_per60(defense_team, defense_score_state, 'corsi', 'against', defense_role)
            )

        def _select_event(strength_state):
            # Randomly select a second's event, if any
            score_diff = sim_game.home_score - sim_game.away_score
            if score_diff >= 3:
                score_code = 3
            elif score_diff == 2:
                score_code = 2
            elif score_diff == 1:
                score_code = 1
            elif score_diff <= -3:
                score_code = -3
            elif score_diff == -2:
                score_code = -2
            elif score_diff == -1:
                score_code = -1
            else:
                score_code = 0

            if sim_game.period == 2:
                period_code = 2
            elif sim_game.period == 3:
                period_code = 3
            elif sim_game.period == 4 and not (season_type == 3 and sim_game.period > reg_periods):
                period_code = 4
            else:
                period_code = 0

            state_key = (
                strength_state,
                int(sim_game.home_goalie_pulled),
                int(sim_game.away_goalie_pulled),
                score_code,
                period_code,
                int(sim_game.home_power_play_expiry > 0),
                int(sim_game.home_penalty_kill_expiry > 0),
                int(sim_game.away_power_play_expiry > 0),
                int(sim_game.away_penalty_kill_expiry > 0),
                sim_game.home_skaters_on_ice,
                sim_game.away_skaters_on_ice,
            )
            cached_rates = event_rate_cache.get(state_key)
            if cached_rates is None:
                if sim_game.home_goalie_pulled:
                    away_rate_key = 'en'
                    home_rate_key = 'pp'
                    empty_net_active = True
                elif sim_game.away_goalie_pulled:
                    away_rate_key = 'pp'
                    home_rate_key = 'en'
                    empty_net_active = True
                else:
                    base_rate_key = STRENGTH_RATE_KEY_MAP.get(strength_state, 'ev')
                    away_rate_key = base_rate_key
                    home_rate_key = base_rate_key
                    empty_net_active = False

                away_corsi = corsi_component['away'][away_rate_key]
                home_corsi = corsi_component['home'][home_rate_key]

                away_role = _venue_role('away', strength_state)
                home_role = _venue_role('home', strength_state)
                away_model_name = 'power_play' if away_role in {'pp', 'pk'} else 'even_strength'
                home_model_name = 'power_play' if home_role in {'pp', 'pk'} else 'even_strength'
                away_corsi += _context_adjustment_per60(away_model_name, 'away', away_role) / 3600.0
                home_corsi += _context_adjustment_per60(home_model_name, 'home', home_role) / 3600.0
                away_corsi += _coach_context_adjustment_per60('away', away_role) / 3600.0
                home_corsi += _coach_context_adjustment_per60('home', home_role) / 3600.0

                if away_corsi < 0.0 or not np.isfinite(away_corsi):
                    away_corsi = 0.0
                if home_corsi < 0.0 or not np.isfinite(home_corsi):
                    home_corsi = 0.0

                cached_rates = (away_corsi, home_corsi, empty_net_active)
                event_rate_cache[state_key] = cached_rates

            away_corsi, home_corsi, empty_net_active = cached_rates
            away_penalties = penalties_away_rate
            home_penalties = penalties_home_rate

            if sim_game.period_seconds_elapsed <= 5 or (sim_game.seconds_elapsed - sim_game.last_penalty_second) <= 2:
                away_corsi = 0.0
                home_corsi = 0.0
            elif (
                not empty_net_active
                and sim_game.last_shot_team_venue is not None
                and (sim_game.seconds_elapsed - sim_game.last_shot_second) <= 5
            ):
                if sim_game.last_shot_team_venue == 'away':
                    home_corsi = 0.0
                else:
                    away_corsi = 0.0

            total = away_corsi + home_corsi + away_penalties + home_penalties
            if total <= 0:
                return EVENT_NO
            if total >= 1:
                away_corsi /= total
                home_corsi /= total
                away_penalties /= total
                home_penalties /= total

            roll = float(random_random())
            if roll < away_corsi:
                return EVENT_AWAY_CORSI
            roll -= away_corsi
            if roll < home_corsi:
                return EVENT_HOME_CORSI
            roll -= home_corsi
            if roll < away_penalties:
                return EVENT_AWAY_PENALTY
            roll -= away_penalties
            if roll < home_penalties:
                return EVENT_HOME_PENALTY
            return EVENT_NO

        def _handle_goalie_return_after_empty_net_goal(scoring_team_venue):
            defending_team_venue = 'home' if scoring_team_venue == 'away' else 'away'
            if defending_team_venue == 'home':
                if sim_game.home_goalie_pulled:
                    sim_game.home_goalie_pulled = False
                    sim_game.home_goalie_pull_cooldown = 3
                    _update_strength_state()
            else:
                if sim_game.away_goalie_pulled:
                    sim_game.away_goalie_pulled = False
                    sim_game.away_goalie_pull_cooldown = 3
                    _update_strength_state()

        def _expire_goal_penalty(scoring_team_venue):
            # A power-play goal expires the oldest active penalty on the opposing team
            defending_team_venue = 'home' if scoring_team_venue == 'away' else 'away'
            if defending_team_venue == 'home':
                defending_penalties = sim_game.home_penalties
                defending_boxed_ids = sim_game.home_boxed_ids
            else:
                defending_penalties = sim_game.away_penalties
                defending_boxed_ids = sim_game.away_boxed_ids

            if scoring_team_venue == 'home':
                scoring_penalties = sim_game.home_penalties
            else:
                scoring_penalties = sim_game.away_penalties

            if min(2, len(defending_penalties)) > min(2, len(scoring_penalties)) and defending_penalties:
                defending_boxed_ids.discard(defending_penalties[0][0])
                defending_penalties.pop(0)
                _update_strength_state()

        def _xg_interaction(shooter, primary_passer=None, secondary_passer=None, goalie=None):
            # Individual finishing, passing, and goaltending estimates modify the base xG
            shooter_impact = getattr(shooter, 'finishing_impact', 0.0)
            passer_impact = 0.0
            passer_impact += getattr(primary_passer, 'induction_impact', 0.0) if primary_passer is not None else 0.0
            passer_impact += getattr(secondary_passer, 'secondary_induction_impact', 0.0) if secondary_passer is not None else 0.0
            goalie_impact = getattr(goalie, 'goaltending_impact', 0.0) if goalie is not None else 0.0
            return (shooter_impact + passer_impact + goalie_impact) / 2.0

        def _append_event(
            event_type,
            event_team_venue=None,
            event_team_abbr=None,
            event_player_1_id=np.nan,
            event_player_2_id=np.nan,
            event_player_3_id=np.nan,
            event_goalie_id=np.nan,
            x_adj=np.nan,
            y_adj=np.nan,
            event_xG=np.nan,
            penalty_duration=np.nan,
            penalty_type=np.nan,
            shot_type=np.nan,
            event_reason=np.nan,
            event_player_1_hand=None,
        ):
            sim_game.event_num += 1
            if not save_pbp:
                return
            game_events.append((
                sim_game.game_id,
                sim_game.period,
                sim_game.seconds_elapsed,
                sim_game.period_seconds_elapsed,
                sim_game.event_num,
                sim_game.strength_state,
                event_type,
                event_team_venue,
                event_team_abbr,
                event_player_1_id,
                event_player_2_id,
                event_player_3_id,
                event_player_1_hand,
                event_goalie_id,
                penalty_duration,
                penalty_type,
                shot_type,
                event_reason,
                x_adj,
                y_adj,
                event_xG,
                sim_game.home_score,
                sim_game.away_score,
                sim_game.home_fenwick,
                sim_game.away_fenwick,
                sim_game.home_expected_goals,
                sim_game.away_expected_goals,
            ))

        def _get_xg_lookup(event_team_venue, rate_key, strength, empty_net, shooter_hand, shot_type_key):
            cache_key = (
                event_team_venue,
                rate_key,
                strength,
                bool(empty_net),
                shooter_hand,
                shot_type_key,
            )
            cached = xg_lookup_cache.get(cache_key)
            if cached is not None:
                return cached

            sampler = fenwick_sampler[event_team_venue][rate_key]
            size = sampler['size']
            if size <= 0:
                empty_lookup = np.array([], dtype=np.float32)
                xg_lookup_cache[cache_key] = empty_lookup
                return empty_lookup

            model_key = _xg_model_key(strength, empty_net=empty_net, rate_key=rate_key)
            model_features = _load_xg_features(model_key)
            model_feature_index = _load_xg_feature_index(model_key)
            features = np.zeros((size, len(model_features)), dtype=np.float32)
            x_vals = sampler['x'].astype(np.float32, copy=False)
            y_vals = sampler['y'].astype(np.float32, copy=False)
            dx_vals = 89.0 - x_vals

            idx = model_feature_index.get('event_distance')
            if idx is not None:
                features[:, idx] = np.sqrt((dx_vals * dx_vals) + (y_vals * y_vals))
            idx = model_feature_index.get('event_angle')
            if idx is not None:
                features[:, idx] = np.degrees(np.arctan2(np.abs(y_vals), np.maximum(1e-9, np.abs(dx_vals))))
            idx = model_feature_index.get('is_home')
            if idx is not None:
                features[:, idx] = float(event_team_venue == 'home')
            idx = model_feature_index.get('empty_net')
            if idx is not None:
                features[:, idx] = float(empty_net)
            idx = model_feature_index.get('offwing')
            if idx is not None:
                if shooter_hand == 'L':
                    features[:, idx] = (y_vals < 0.0).astype(np.float32, copy=False)
                elif shooter_hand == 'R':
                    features[:, idx] = (y_vals >= 0.0).astype(np.float32, copy=False)

            strength_idx = model_feature_index.get(f'strength_{strength}')
            if strength_idx is not None:
                features[:, strength_idx] = 1.0

            shot_type_idx = model_feature_index.get(shot_type_key)
            if shot_type_idx is not None:
                features[:, shot_type_idx] = 1.0
            else:
                other_idx = model_feature_index.get('other-shot')
                if other_idx is not None:
                    features[:, other_idx] = 1.0

            model = xg_models[model_key]
            predictions = np.clip(
                model.inplace_predict(features, validate_features=False),
                0.0,
                1.0,
            ).astype(np.float32, copy=False)
            baseline_mean = float(np.dot(predictions.astype(np.float64, copy=False), sampler['weights']))
            target_mean = float(xg_target_per_fenwick[event_team_venue][rate_key])
            if baseline_mean > 1e-9 and target_mean > 0.0:
                predictions = np.clip(
                    predictions * (target_mean / baseline_mean),
                    0.0,
                    1.0,
                ).astype(np.float32, copy=False)
            xg_lookup_cache[cache_key] = predictions
            return predictions

        def xg_finder(x_adj, y_adj, event_team_venue, strength, shooter=None, goalie=None, shot_type=None):
            # Compute the xG of a single shot using cached models and a base-prediction cache.
            x_adj = float(x_adj)
            y_adj = float(y_adj)
            shot_type_key = shot_type if isinstance(shot_type, str) else ''
            shooter_hand = getattr(shooter, 'handedness', None)
            shooter_hand = shooter_hand if isinstance(shooter_hand, str) else ''
            empty_net = goalie is None
            offwing = (
                ((y_adj < 0.0) and shooter_hand == 'L')
                or ((y_adj >= 0.0) and shooter_hand == 'R')
            )
            cache_key = (x_adj, y_adj, event_team_venue, strength, empty_net, shooter_hand, shot_type_key)
            cached_prediction = XG_BASE_PREDICTION_CACHE.get(cache_key)
            if cached_prediction is not None:
                return cached_prediction

            dx = 89.0 - x_adj
            model_key = _xg_model_key(strength, empty_net=empty_net)
            model_features = _load_xg_features(model_key)
            model_feature_index = _load_xg_feature_index(model_key)
            xg_vector = np.zeros((1, len(model_features)), dtype=np.float32)
            idx = model_feature_index.get('event_distance')
            if idx is not None:
                xg_vector[0, idx] = math.sqrt((dx * dx) + (y_adj * y_adj))
            idx = model_feature_index.get('event_angle')
            if idx is not None:
                xg_vector[0, idx] = math.degrees(math.atan2(abs(y_adj), max(1e-9, abs(dx))))
            idx = model_feature_index.get('is_home')
            if idx is not None:
                xg_vector[0, idx] = float(event_team_venue == 'home')
            idx = model_feature_index.get('empty_net')
            if idx is not None:
                xg_vector[0, idx] = float(empty_net)
            idx = model_feature_index.get('offwing')
            if idx is not None:
                xg_vector[0, idx] = float(offwing)

            strength_idx = model_feature_index.get(f'strength_{strength}')
            if strength_idx is not None:
                xg_vector[0, strength_idx] = 1.0

            shot_type_idx = model_feature_index.get(shot_type_key)
            if shot_type_idx is not None:
                xg_vector[0, shot_type_idx] = 1.0
            else:
                other_idx = model_feature_index.get('other-shot')
                if other_idx is not None:
                    xg_vector[0, other_idx] = 1.0

            model = xg_models[model_key]
            try:
                prediction = float(model.inplace_predict(xg_vector, validate_features=False)[0])
            except Exception:
                prediction = _single_shot_xg(
                    x_adj,
                    y_adj,
                    shooting_venue=event_team_venue,
                    strength_state=strength,
                    shooter=shooter,
                    goalie=goalie,
                    shot_type=shot_type,
                )
            prediction = float(np.clip(prediction, 0.0, 1.0))
            XG_BASE_PREDICTION_CACHE[cache_key] = prediction
            return prediction

        def _second_workflow():
            # Single second workflow to determine what will occur in the game

            # Retrieve active strength state
            strength = sim_game.strength_state

            # Roll events die
            event = _select_event(strength)
            if event == EVENT_NO:
                return

            # Penalty branch
            if event == EVENT_AWAY_PENALTY or event == EVENT_HOME_PENALTY:
                event_team_venue = 'away' if event == EVENT_AWAY_PENALTY else 'home'
                event_team_abbr = away_team_abbr if event_team_venue == 'away' else home_team_abbr
                event_type = 'penalty'

                # Randomly select a penalty taker and drawer based on observed rates
                penalized_skaters = _eligible_skaters(event_team_venue)
                drawn_team_venue = 'home' if event_team_venue == 'away' else 'away'
                drawn_skaters = _eligible_skaters(drawn_team_venue)

                if not penalized_skaters or not drawn_skaters:
                    return

                if event_team_venue == 'home' and not sim_game.home_boxed_ids:
                    event_player_1_id = _sample_prepared(player_sampling['home']['penalties_taken_rate'], return_player=False)
                elif event_team_venue == 'away' and not sim_game.away_boxed_ids:
                    event_player_1_id = _sample_prepared(player_sampling['away']['penalties_taken_rate'], return_player=False)
                else:
                    event_player_1_id = _weighted_choice_from_players(penalized_skaters, 'penalties_taken_rate', return_player=False)

                if drawn_team_venue == 'home' and not sim_game.home_boxed_ids:
                    event_player_2_id = _sample_prepared(player_sampling['home']['penalties_drawn_rate'], return_player=False)
                elif drawn_team_venue == 'away' and not sim_game.away_boxed_ids:
                    event_player_2_id = _sample_prepared(player_sampling['away']['penalties_drawn_rate'], return_player=False)
                else:
                    event_player_2_id = _weighted_choice_from_players(drawn_skaters, 'penalties_drawn_rate', return_player=False)

                if event_team_venue == 'home':
                    sim_game.home_penalties.append([event_player_1_id, 120])
                    sim_game.home_boxed_ids.add(event_player_1_id)
                else:
                    sim_game.away_penalties.append([event_player_1_id, 120])
                    sim_game.away_boxed_ids.add(event_player_1_id)
                sim_game.last_penalty_second = sim_game.seconds_elapsed

                _update_strength_state()

                _append_event(
                    event_type,
                    event_team_venue=event_team_venue,
                    event_team_abbr=event_team_abbr,
                    event_player_1_id=event_player_1_id,
                    event_player_2_id=event_player_2_id,
                    penalty_duration=2,
                    penalty_type='minor'
                )

            # Corsi rate
            elif event == EVENT_AWAY_CORSI or event == EVENT_HOME_CORSI:
                event_team_venue = 'away' if event == EVENT_AWAY_CORSI else 'home'
                event_team_abbr = away_team_abbr if event_team_venue == 'away' else home_team_abbr
                event_player_2_id = np.nan
                event_player_3_id = np.nan
                event_goalie_id = np.nan
                event_xG = np.nan
                x_adj = np.nan
                y_adj = np.nan
                event_reason = np.nan
                shot_type = np.nan

                # Randomly select the shooter
                for_skaters = _eligible_skaters(event_team_venue)
                against_team_venue = 'home' if event_team_venue == 'away' else 'away'
                against_skaters = _eligible_skaters(against_team_venue)

                if not for_skaters or not against_skaters:
                    return

                if event_team_venue == 'home' and not sim_game.home_boxed_ids:
                    event_player_1 = _sample_prepared(player_sampling['home']['shot_rate'], return_player=True)
                elif event_team_venue == 'away' and not sim_game.away_boxed_ids:
                    event_player_1 = _sample_prepared(player_sampling['away']['shot_rate'], return_player=True)
                else:
                    event_player_1 = _weighted_choice_from_players(for_skaters, 'shot_rate', return_player=True)
                event_player_1_id = event_player_1.player_id
                sim_game.last_shot_second = sim_game.seconds_elapsed
                sim_game.last_shot_team_venue = event_team_venue

                # Determine if a corsi shot attempt is unblocked (becoming a fenwick shot)
                if np.random.rand() >= event_player_1.fenwick_per_corsi:
                    event_type = 'blocked-shot'

                    # Randomly selecting the blocking player
                    if against_team_venue == 'home' and not sim_game.home_boxed_ids:
                        event_player_2_id = _sample_prepared(player_sampling['home']['block_rate'], return_player=False)
                    elif against_team_venue == 'away' and not sim_game.away_boxed_ids:
                        event_player_2_id = _sample_prepared(player_sampling['away']['block_rate'], return_player=False)
                    else:
                        event_player_2_id = _weighted_choice_from_players(against_skaters, 'block_rate', return_player=False)
                else:
                    # Randomly select (up to) two passers, with a 20 percent chance of there being none
                    passing_skaters = [p for p in for_skaters if p.player_id != event_player_1_id]

                    if passing_skaters and np.random.rand() >= 0.20:
                        event_player_2_id = _weighted_choice_from_players(
                            passing_skaters,
                            'primary_pass_rate',
                            return_player=False,
                        )

                        secondary_skaters = [
                            p for p in passing_skaters
                            if p.player_id != event_player_2_id
                        ]

                        if secondary_skaters and np.random.rand() < event_player_1.secondary_pass_rate:
                            event_player_3_id = _weighted_choice_from_players(
                                secondary_skaters,
                                'secondary_pass_rate',
                                return_player=False,
                            )

                    # Sample a fenwick shot location using the fenwick grids
                    if sim_game.home_goalie_pulled:
                        rate_key = 'pp' if event_team_venue == 'home' else 'en'
                    elif sim_game.away_goalie_pulled:
                        rate_key = 'en' if event_team_venue == 'home' else 'pp'
                    else:
                        rate_key = STRENGTH_RATE_KEY_MAP.get(strength, 'ev')
                    sampler = fenwick_sampler[event_team_venue][rate_key]
                    if sampler['size'] <= 0:
                        return
                    shot_idx = int(np.searchsorted(sampler['cdf'], np.random.random(), side='right'))
                    if shot_idx >= sampler['size']:
                        shot_idx = sampler['size'] - 1

                    # Keep default right-net coordinates for xG; store oriented coordinates in the event log.
                    orientation = 1 if event_team_venue == 'home' else -1
                    x_default = float(sampler['x'][shot_idx])
                    y_default = float(sampler['y'][shot_idx])
                    x_adj = orientation * x_default
                    y_adj = orientation * y_default

                    # Find the xG of the shot based on the coordinates only
                    team_player_lookup = home_player_lookup if event_team_venue == 'home' else away_player_lookup
                    primary_passer = team_player_lookup.get(event_player_2_id)
                    secondary_passer = team_player_lookup.get(event_player_3_id)
                    event_goalie = _active_goalie(against_team_venue)
                    event_goalie_id = getattr(event_goalie, 'player_id', np.nan)
                    shooter_hand = getattr(event_player_1, 'handedness', None)
                    shooter_hand = shooter_hand if shooter_hand in {'L', 'R'} else ''
                    shot_type_key = shot_type if isinstance(shot_type, str) else ''
                    try:
                        xg_lookup = _get_xg_lookup(
                            event_team_venue,
                            rate_key,
                            strength,
                            event_goalie is None,
                            shooter_hand,
                            shot_type_key,
                        )
                        event_xG = float(xg_lookup[shot_idx]) if shot_idx < len(xg_lookup) else 0.0
                    except Exception:
                        event_xG = xg_finder(
                            x_default,
                            y_default,
                            event_team_venue,
                            strength,
                            shooter=event_player_1,
                            goalie=event_goalie,
                            shot_type=shot_type
                        )

                    # Log base xG in PBP/expected-goals totals, but use the interaction-adjusted
                    # probability for the actual goal outcome.
                    base_event_xG = float(np.clip(event_xG, 0.0, 1.0))
                    goal_probability = np.clip(
                        event_xG + _xg_interaction(
                            event_player_1,
                            primary_passer=primary_passer,
                            secondary_passer=secondary_passer,
                            goalie=event_goalie
                        ),
                        0.0,
                        1.0
                    )
                    if event_team_venue == 'home':
                        sim_game.home_fenwick += 1
                        sim_game.home_expected_goals += base_event_xG
                    else:
                        sim_game.away_fenwick += 1
                        sim_game.away_expected_goals += base_event_xG
                    event_xG = base_event_xG

                    # Select the type of fenwick shot
                    if np.random.rand() < goal_probability:
                        event_type = 'goal'
                        if event_team_venue == 'home':
                            sim_game.home_score += 1
                        else:
                            sim_game.away_score += 1
                        if event_goalie is None:
                            _handle_goalie_return_after_empty_net_goal(event_team_venue)
                        _expire_goal_penalty(event_team_venue)
                    # Randomly determine if a fenwick shot attempt reaches net
                    elif np.random.rand() >= event_player_1.shots_per_fenwick:
                        event_type = 'missed-shot'
                    else:
                        event_type = 'shot-on-goal'

                _append_event(
                    event_type,
                    event_team_venue=event_team_venue,
                    event_team_abbr=event_team_abbr,
                    event_player_1_id=event_player_1_id,
                    event_player_2_id=event_player_2_id,
                    event_player_3_id=event_player_3_id,
                    event_goalie_id=event_goalie_id,
                    x_adj=x_adj,
                    y_adj=y_adj,
                    event_xG=event_xG,
                    shot_type=shot_type,
                    event_reason=event_reason,
                    event_player_1_hand=getattr(event_player_1, 'handedness', None)
                )

        def _play_period(period_number, period_seconds=None, stop_on_goal=False):
            # Iterate through the per-second workflow of randomly determining if and what event occurs (and by who)
            sim_game.period = period_number
            sim_game.period_seconds_elapsed = 0
            period_seconds = self.reg_seconds if period_seconds is None else period_seconds
            _update_strength_state()

            for second in range(period_seconds):
                # Mark the current second
                sim_game.seconds_elapsed += 1
                sim_game.period_seconds_elapsed += 1
                score_before = (sim_game.home_score, sim_game.away_score)
                strength_dirty = _tick_penalty_box()
                strength_dirty = _apply_goalie_pull_state() or strength_dirty
                if strength_dirty:
                    _update_strength_state()

                # Execute the workflow
                _second_workflow()
                _tick_context_expiry()

                if stop_on_goal and score_before != (sim_game.home_score, sim_game.away_score):
                    return True

            return False

        for period in range(1, self.reg_periods + 1):
            _play_period(period)

        result = 'regulation'
        if sim_game.home_score == sim_game.away_score:
            if self.season_type == 2:
                ot_finished = _play_period(
                    self.reg_periods + 1,
                    period_seconds=self.ot_seconds,
                    stop_on_goal=True
                )
                if ot_finished:
                    result = 'overtime'
                else:
                    shootout_team_venue = np.random.choice(['home', 'away'])
                    if shootout_team_venue == 'home':
                        sim_game.home_score += 1
                    else:
                        sim_game.away_score += 1

                    _append_event(
                        'shootout',
                        event_team_venue=shootout_team_venue,
                        event_team_abbr=getattr(self, shootout_team_venue).team_abbr,
                        event_reason='coin-flip'
                    )
                    result = 'shootout'
            elif self.season_type == 3:
                ot_period = self.reg_periods + 1
                while sim_game.home_score == sim_game.away_score:
                    ot_finished = _play_period(
                        ot_period,
                        period_seconds=self.ot_seconds,
                        stop_on_goal=True
                    )
                    if ot_finished:
                        break
                    ot_period += 1
                result = 'overtime'
            else:
                raise ValueError(f'Unsupported tied-game season_type: {self.season_type}')

        if sim_game.home_score == sim_game.away_score:
            raise RuntimeError('Simulation ended without a winner.')

        pbp = pd.DataFrame.from_records(game_events, columns=PBP_COLUMNS) if save_pbp else _empty_pbp()
        winner = (
            self.home.team_abbr if sim_game.home_score > sim_game.away_score
            else self.away.team_abbr
        )

        return {
            'game_id': sim_game.game_id,
            'pbp': pbp,
            'home_score': sim_game.home_score,
            'away_score': sim_game.away_score,
            'home_fenwick': sim_game.home_fenwick,
            'away_fenwick': sim_game.away_fenwick,
            'home_expected_goals': sim_game.home_expected_goals,
            'away_expected_goals': sim_game.away_expected_goals,
            'time_on_ice': sim_game.seconds_elapsed,
            'winner': winner,
            'result': result
        }


def _game_summary_record(trial, game, home_team_abbr, away_team_abbr, series_trial=None, series_winner=None):
    return {
        'trial': trial,
        'series_trial': series_trial,
        'series_winner': series_winner,
        'game_id': game['game_id'],
        'home_team_abbr': home_team_abbr,
        'away_team_abbr': away_team_abbr,
        'home_score': game['home_score'],
        'away_score': game['away_score'],
        'home_fenwick': game['home_fenwick'],
        'away_fenwick': game['away_fenwick'],
        'home_expected_goals': game['home_expected_goals'],
        'away_expected_goals': game['away_expected_goals'],
        'time_on_ice': game['time_on_ice'],
        'winner': game['winner'],
        'result': game['result'],
    }


def _simulate_run_trials_game(sim_engine, trial_id, save_pbp=True):
    return sim_engine.simulate_game(game_id=trial_id, save_pbp=save_pbp)


def _playoff_series_matchup(home_team, away_team, game_number):
    if game_number in {1, 2, 5, 7}:
        return home_team, away_team
    return away_team, home_team


def _annotate_series_records(records, series_trial, series_winner):
    for row in records:
        row['series_trial'] = series_trial
        row['series_winner'] = series_winner
    return records


def _sort_summary_records(summaries):
    summaries.sort(
        key=lambda row: (
            row['series_trial'] if row.get('series_trial') is not None else row['trial'],
            row['trial'],
        )
    )
    return summaries


def _series_simulator_key(home_team, away_team):
    return (home_team.team_abbr, away_team.team_abbr)


def _normalize_playoff_series_seed(home_games, away_games):
    home_games = int(home_games)
    away_games = int(away_games)

    if home_games < 0 or away_games < 0:
        raise ValueError('home_games and away_games must be non-negative integers.')
    if home_games >= 4 or away_games >= 4:
        raise ValueError('Seeded playoff series must start before either team has 4 wins.')

    total_games_played = home_games + away_games
    if total_games_played >= 7:
        raise ValueError('Seeded playoff series must start before Game 7 is completed.')

    return home_games, away_games, total_games_played


def _build_series_simulators(base_sim_engine, base_home_team, base_away_team, season_type, reg_periods, reg_minutes):
    simulators = {
        _series_simulator_key(base_home_team, base_away_team): base_sim_engine,
    }
    reverse_key = _series_simulator_key(base_away_team, base_home_team)
    if reverse_key not in simulators:
        simulators[reverse_key] = WSBASimulation(
            home=base_away_team,
            away=base_home_team,
            season_type=season_type,
            reg_periods=reg_periods,
            reg_minutes=reg_minutes,
        )
    return simulators


def _simulate_playoff_series(
    sim_engine,
    series_trial,
    base_home_team,
    base_away_team,
    series_simulators=None,
    save_pbp=True,
    home_games=0,
    away_games=0,
):
    home_abbr = base_home_team.team_abbr
    away_abbr = base_away_team.team_abbr
    home_games, away_games, total_games_played = _normalize_playoff_series_seed(home_games, away_games)
    wins = {home_abbr: home_games, away_abbr: away_games}
    summaries = []
    pbp_records = [] if save_pbp else None

    if series_simulators is None:
        series_simulators = _build_series_simulators(
            sim_engine,
            base_home_team,
            base_away_team,
            getattr(sim_engine, 'season_type', 2),
            getattr(sim_engine, 'reg_periods', 3),
            int(getattr(sim_engine, 'reg_seconds', 1200) / 60),
        )

    for game_number in range(total_games_played + 1, 8):
        if max(wins.values()) >= 4:
            break

        scheduled_home, scheduled_away = _playoff_series_matchup(base_home_team, base_away_team, game_number)
        scheduled_sim = series_simulators[_series_simulator_key(scheduled_home, scheduled_away)]
        game_trial = (int(series_trial) * 10) + game_number
        game = scheduled_sim.simulate_game(game_id=game_trial, save_pbp=save_pbp)

        summaries.append(
            _game_summary_record(
                game_trial,
                game,
                scheduled_home.team_abbr,
                scheduled_away.team_abbr,
                series_trial=series_trial,
            )
        )

        if save_pbp:
            game_events = combine_game_events(game)
            for row in game_events:
                row['trial'] = game_trial
            pbp_records.extend(_annotate_series_records(game_events, series_trial, None))

        winner = game.get('winner')
        if winner in wins:
            wins[winner] += 1

        if wins[home_abbr] >= 4 or wins[away_abbr] >= 4:
            break

    series_winner = home_abbr if wins[home_abbr] > wins[away_abbr] else away_abbr
    for row in summaries:
        row['series_winner'] = series_winner
    if save_pbp:
        for row in pbp_records:
            row['series_winner'] = series_winner

    return {
        'games': summaries,
        'pbp': pbp_records if save_pbp else [],
        'completed_trials': 1,
    }


def _empty_games_frame():
    return pd.DataFrame(columns=GAME_SUMMARY_COLUMNS)


def _empty_pbp():
    return pd.DataFrame()


def combine_game_events(game):
    events = game.get('pbp')
    if events is None:
        return []
    if isinstance(events, list):
        return events
    if events.empty:
        return []
    return events.to_dict('records')


def _team_cache_signature(team):
    active_roster = getattr(team, 'active_roster', []) or []
    player_signature = tuple(
        sorted(
            (
                int(getattr(player, 'player_id', 0) or 0),
                str(getattr(player, 'position', '') or ''),
                float(getattr(player, 'time_on_ice', 0.0) or 0.0),
                float(getattr(player, 'ev_time_on_ice', 0.0) or 0.0),
                float(getattr(player, 'pp_time_on_ice', 0.0) or 0.0),
                float(getattr(player, 'pk_time_on_ice', 0.0) or 0.0),
            )
            for player in active_roster
        )
    )
    return (
        str(getattr(team, 'team_abbr', '') or ''),
        int(getattr(getattr(team, 'starting_goalie', None), 'player_id', 0) or 0),
        player_signature,
    )


_RUN_TRIALS_WORKER_SIM_CACHE = {}


def _get_worker_cached_simulator(home_team, away_team, season_type, reg_periods, reg_minutes):
    cache_key = (
        _team_cache_signature(home_team),
        _team_cache_signature(away_team),
        int(season_type),
        int(reg_periods),
        int(reg_minutes),
    )
    cached = _RUN_TRIALS_WORKER_SIM_CACHE.get(cache_key)
    if cached is None:
        cached = WSBASimulation(
            home=home_team,
            away=away_team,
            season_type=season_type,
            reg_periods=reg_periods,
            reg_minutes=reg_minutes,
        )
        _RUN_TRIALS_WORKER_SIM_CACHE[cache_key] = cached
    return cached


def _resolve_run_trials_teams(sim_engine, home, away, season_type, roster_dir):
    if sim_engine is not None:
        return sim_engine.home, sim_engine.away, getattr(sim_engine, 'season_type', season_type)

    if home is None or away is None:
        raise ValueError('Provide either sim_engine or both home and away teams.')

    home_team = load_team_from_roster_csv(home, roster_dir=roster_dir) if isinstance(home, str) else home
    away_team = load_team_from_roster_csv(away, roster_dir=roster_dir) if isinstance(away, str) else away
    return home_team, away_team, season_type


def _print_run_trials_header(away_team_abbr, home_team_abbr):
    sys.stdout.write(f"away = {away_team_abbr}, home = {home_team_abbr}\n\n")
    sys.stdout.flush()


def _print_trial_progress(completed, trials, away_team_abbr, home_team_abbr):
    sys.stdout.write(f"\r{away_team_abbr} vs {home_team_abbr}: Trial: {completed}/{trials}")
    sys.stdout.flush()


def _build_trial_chunks(trials, start_trial, n_workers, chunks_per_worker=8):
    worker_count = max(1, min(int(n_workers), int(trials)))
    target_chunks = max(worker_count, worker_count * max(1, int(chunks_per_worker)))
    chunk_size = max(1, int(np.ceil(float(trials) / float(target_chunks))))
    chunks = []
    current_trial = start_trial
    remaining = int(trials)

    while remaining > 0:
        count = min(chunk_size, remaining)
        chunks.append((current_trial, count))
        current_trial += count
        remaining -= count

    return chunks


def _run_trials_chunk(task):
    (
        home_team,
        away_team,
        season_type,
        reg_periods,
        reg_minutes,
        start_trial,
        count,
        warmup,
        playoff_series,
        save_pbp,
        home_games,
        away_games,
    ) = task
    sim_engine = _get_worker_cached_simulator(
        home_team,
        away_team,
        season_type,
        reg_periods,
        reg_minutes,
    )
    home_team_abbr = sim_engine.home.team_abbr
    away_team_abbr = sim_engine.away.team_abbr
    series_simulators = None
    if playoff_series:
        swapped_sim_engine = _get_worker_cached_simulator(
            away_team,
            home_team,
            season_type,
            reg_periods,
            reg_minutes,
        )
        series_simulators = {
            _series_simulator_key(home_team, away_team): sim_engine,
            _series_simulator_key(away_team, home_team): swapped_sim_engine,
        }

    if warmup:
        _simulate_run_trials_game(
            sim_engine,
            max(0, start_trial - 1),
            save_pbp=save_pbp,
        )

    summaries = []
    pbp_records = []

    for trial in range(start_trial, start_trial + count):
        if playoff_series:
            series_result = _simulate_playoff_series(
                sim_engine,
                trial,
                home_team,
                away_team,
                series_simulators=series_simulators,
                save_pbp=save_pbp,
                home_games=home_games,
                away_games=away_games,
            )
            summaries.extend(series_result['games'])
            if save_pbp:
                pbp_records.extend(series_result['pbp'])
            continue

        game = _simulate_run_trials_game(
            sim_engine,
            trial,
            save_pbp=save_pbp,
        )
        summaries.append(_game_summary_record(trial, game, home_team_abbr, away_team_abbr))
        if save_pbp:
            pbp_records.extend(combine_game_events(game))

    return {'games': summaries, 'pbp': pbp_records, 'completed_trials': count}


def _chunks_per_worker(n_workers):
    return 8 if int(n_workers) > 1 else 2


class RunTrialsWorkerPool:
    def __init__(self, n_workers):
        self.n_workers = max(1, int(n_workers))
        self.max_workers = self.n_workers
        self._executor = ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=mp.get_context('spawn'),
        )

    def submit(self, *args, **kwargs):
        return self._executor.submit(*args, **kwargs)

    def shutdown(self, wait=True, cancel_futures=False):
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()
        return False


def create_run_trials_worker_pool(n_workers):
    return RunTrialsWorkerPool(n_workers)


def _worker_pool_size(worker_pool):
    if worker_pool is None:
        return None

    for attr in ('max_workers', '_max_workers', 'n_workers'):
        value = getattr(worker_pool, attr, None)
        if value is None:
            continue
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            continue

    return None


def run_trials(
    trials,
    sim_engine=None,
    home=None,
    away=None,
    season_type=2,
    n_workers=1,
    reg_periods=3,
    reg_minutes=20,
    start_trial=1,
    warmup=False,
    roster_dir=None,
    save_pbp=True,
    playoff_series=False,
    home_games=0,
    away_games=0,
    worker_pool=None,
    **_ignored_kwargs,
):
    trials = int(trials)
    start_trial = int(start_trial)

    if trials <= 0:
        return {'games': _empty_games_frame(), 'pbp': _empty_pbp()}

    home_team, away_team, resolved_season_type = _resolve_run_trials_teams(
        sim_engine,
        home,
        away,
        season_type,
        roster_dir,
    )
    home_team_abbr = home_team.team_abbr
    away_team_abbr = away_team.team_abbr
    if playoff_series:
        home_games, away_games, _ = _normalize_playoff_series_seed(home_games, away_games)
    _print_run_trials_header(away_team_abbr, home_team_abbr)
    _print_trial_progress(0, trials, away_team_abbr, home_team_abbr)

    pool_size = _worker_pool_size(worker_pool)
    effective_workers = pool_size if pool_size is not None else int(n_workers)

    if worker_pool is None and effective_workers <= 1:
        sim_engine = WSBASimulation(
            home=home_team,
            away=away_team,
            season_type=resolved_season_type,
            reg_periods=reg_periods,
            reg_minutes=reg_minutes,
        )
        series_simulators = None
        if playoff_series:
            series_simulators = _build_series_simulators(
                sim_engine,
                home_team,
                away_team,
                resolved_season_type,
                reg_periods,
                reg_minutes,
            )

        if warmup:
            _simulate_run_trials_game(
                sim_engine,
                max(0, start_trial - 1),
                save_pbp=save_pbp,
            )

        summaries = []
        pbp_records = []
        for completed, trial in enumerate(range(start_trial, start_trial + trials), start=1):
            if playoff_series:
                series_result = _simulate_playoff_series(
                    sim_engine,
                    trial,
                    home_team,
                    away_team,
                    series_simulators=series_simulators,
                    save_pbp=save_pbp,
                    home_games=home_games,
                    away_games=away_games,
                )
                summaries.extend(series_result['games'])
                if save_pbp:
                    pbp_records.extend(series_result['pbp'])
            else:
                game = _simulate_run_trials_game(
                    sim_engine,
                    trial,
                    save_pbp=save_pbp,
                )
                summaries.append(_game_summary_record(trial, game, home_team_abbr, away_team_abbr))
                if save_pbp:
                    pbp_records.extend(combine_game_events(game))
            _print_trial_progress(completed, trials, away_team_abbr, home_team_abbr)
    else:
        chunks = _build_trial_chunks(
            trials,
            start_trial,
            effective_workers,
            chunks_per_worker=_chunks_per_worker(effective_workers),
        )
        tasks = [
            (
                home_team,
                away_team,
                resolved_season_type,
                reg_periods,
                reg_minutes,
                chunk_start,
                chunk_count,
                warmup and idx == 0,
                playoff_series,
                save_pbp,
                home_games,
                away_games,
            )
            for idx, (chunk_start, chunk_count) in enumerate(chunks)
        ]
        summaries = []
        pbp_records = []
        worker_count = max(1, min(int(effective_workers), len(tasks)))
        owns_worker_pool = worker_pool is None

        if owns_worker_pool:
            worker_pool = create_run_trials_worker_pool(worker_count)

        try:
            completed = 0
            futures = [worker_pool.submit(_run_trials_chunk, task) for task in tasks]
            for future in as_completed(futures):
                chunk_result = future.result()
                summaries.extend(chunk_result['games'])
                if save_pbp:
                    pbp_records.extend(chunk_result['pbp'])
                completed += int(chunk_result.get('completed_trials', 0))
                _print_trial_progress(completed, trials, away_team_abbr, home_team_abbr)
        finally:
            if owns_worker_pool and worker_pool is not None:
                worker_pool.shutdown()

        _sort_summary_records(summaries)

    print()

    _sort_summary_records(summaries)
    games = pd.DataFrame.from_records(summaries, columns=GAME_SUMMARY_COLUMNS)
    pbp = pd.DataFrame.from_records(pbp_records) if save_pbp and pbp_records else _empty_pbp()
    return {'games': games, 'pbp': pbp}
