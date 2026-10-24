import json
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
import torch
import xgboost as xgb
import optuna
import wsba_hockey as wsba
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from wsba_hockey.tools.scraping import fix_players

TARGET = "is_goal"
FEATURES = [
    "event_distance",
    "event_angle",
    "distance_from_last",
    "angle_from_last",
    "seconds_since_last",
    "speed_from_last",
    "speed_of_angle_from_last",
    "event_player_1_shift_time_on_ice",
    "event_on_ice_against_min_shift_time_on_ice",
    "event_on_ice_against_avg_shift_time_on_ice",
    "event_on_ice_against_max_shift_time_on_ice",
    "wrist",
    "deflected",
    "tip-in",
    "slap",
    "backhand",
    "snap",
    "wrap-around",
    "poke",
    "bat",
    "cradle",
    "between-legs",
    "other-shot",
    "prior_same",
    "prior_shot-on-goal",
    "prior_missed-shot",
    "prior_blocked-shot",
    "prior_giveaway",
    "prior_takeaway",
    "prior_hit",
    "prior_faceoff",
    "strength_3v3",
    "strength_3v4",
    "strength_3v5",
    "strength_4v3",
    "strength_4v4",
    "strength_4v5",
    "strength_4v6",
    "strength_5v3",
    "strength_5v4",
    "strength_5v5",
    "strength_5v6",
    "strength_6v4",
    "strength_6v5",
    "empty_net",
    "offwing",
    "short",
    "failed_bank",
    "cross_ice",
    "rebound",
    "rush",
]

MODEL_STRENGTHS = {
    "ev": wsba.STRENGTH_GROUPS["even_strength"],
    "pp": wsba.STRENGTH_GROUPS["power_play"] + ["6v4", "6v5"],
    "pk": wsba.STRENGTH_GROUPS["penalty_kill"],
    "en": wsba.STRENGTHS,
}
DEFAULT_MODEL_NAMES = ("ev", "pp", "pk", "en")

XG_MODEL_DIR = Path(__file__).resolve().parent / "xg_model"
MODEL_DIR = XG_MODEL_DIR
TUNING_DIR = XG_MODEL_DIR / "tuning"
METRICS_PATH = XG_MODEL_DIR / "wsba_xg_metrics.json"

HYPERPARAMETER_SPACE = {
    "eta": {"kind": "uniform", "low": 0.03, "high": 0.15, "default": 0.1},
    "gamma": {"kind": "uniform", "low": 0.0, "high": 0.5, "default": 0.0},
    "max_depth": {"kind": "int", "low": 3, "high": 10, "default": 6},
    "min_child_weight": {"kind": "int", "low": 1, "high": 30, "default": 1},
    "max_delta_step": {"kind": "int", "low": 0, "high": 10, "default": 0},
    "subsample": {"kind": "uniform", "low": 0.65, "high": 1.0, "default": 1.0},
    "colsample_bytree": {"kind": "uniform", "low": 0.65, "high": 1.0, "default": 1.0},
    "colsample_bylevel": {"kind": "uniform", "low": 0.65, "high": 1.0, "default": 1.0},
    "colsample_bynode": {"kind": "uniform", "low": 0.65, "high": 1.0, "default": 1.0},
    "reg_alpha": {"kind": "loguniform", "low": 0.0001, "high": 10.0, "default": 0.0},
    "reg_lambda": {"kind": "loguniform", "low": 0.1, "high": 20.0, "default": 1.0},
    "max_bin": {"kind": "int", "low": 128, "high": 512, "default": 256},
    "grow_policy": {
        "kind": "choice",
        "values": ["depthwise", "lossguide"],
        "default": "depthwise",
    },
    "max_leaves": {"kind": "int", "low": 0, "high": 128, "default": 0},
    "scale_pos_weight": {"kind": "uniform", "low": 1.0, "high": 12.0, "default": 1.0},
}
TUNED_HYPERPARAMETERS = [
    "eta",
    "gamma",
    "max_depth",
    "min_child_weight",
    "max_delta_step",
    "subsample",
    "colsample_bytree",
]
HYPERPARAMETERS = {name: {} for name in MODEL_STRENGTHS}


def _as_polars(frame):
    if isinstance(frame, pl.DataFrame):
        return frame.clone(), False
    if isinstance(frame, pd.DataFrame):
        converted = pl.DataFrame(
            frame.to_dict(orient="records"), infer_schema_length=None
        )
        return (
            converted.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None)),
            True,
        )
    raise TypeError("xG data must be a Polars or Pandas DataFrame")


def _export_parameters(params):
    return {
        key: ",".join(value) if isinstance(value, list) else value
        for key, value in params.items()
    }


def _exclude_outcome_leaking_shots(frame):
    if "event_reason" not in frame.columns:
        return frame
    return frame.filter(
        ~pl.col("event_reason")
        .is_in(["short", "failed-bank-attempt"])
        .fill_null(False)
    )


def apply_time_on_ice(pbp):
    toi = [
        "event_player_1_shift_time_on_ice",
        "event_on_ice_against_min_shift_time_on_ice",
        "event_on_ice_against_avg_shift_time_on_ice",
        "event_on_ice_against_max_shift_time_on_ice",
    ]
    home = [f"home_on_{i}_id" for i in range(1, 7)]
    away = [f"away_on_{i}_id" for i in range(1, 7)]
    sort_columns = [
        column
        for column in ["season", "game_id", "period", "seconds_elapsed", "event_num"]
        if column in pbp.columns
    ]
    pbp = (
        pbp.drop(["_toi_row", *toi], strict=False)
        .with_columns(
            [
                pl.col("event_player_1_id").cast(pl.Int64, strict=False),
                *[pl.col(c).cast(pl.Int64, strict=False) for c in home + away],
                pl.col("ids_on").cast(pl.String, strict=False),
                pl.col("ids_off").cast(pl.String, strict=False),
            ]
        )
        .sort(sort_columns)
        .with_row_index("_toi_row")
    )
    keys = ["season", "game_id", "period", "player_id"]
    change_rows = pbp.filter(pl.col("event_type") == "change").select(
        ["season", "game_id", "period", "seconds_elapsed", "event_num", "ids_on", "ids_off"]
    )
    starts = (
        change_rows.select(
            ["season", "game_id", "period", "seconds_elapsed", "event_num", "ids_on"]
        )
        .with_columns(pl.col("ids_on").str.split(","))
        .explode("ids_on")
        .with_columns(pl.col("ids_on").str.strip_chars().cast(pl.Int64, strict=False).alias("player_id"))
        .filter(pl.col("player_id").is_not_null())
        .rename({"seconds_elapsed": "_start_time", "event_num": "_start_event"})
        .sort(["_start_time"] + keys + ["_start_event"])
    )
    ends = (
        change_rows.select(
            ["season", "game_id", "period", "seconds_elapsed", "event_num", "ids_off"]
        )
        .with_columns(pl.col("ids_off").str.split(","))
        .explode("ids_off")
        .with_columns(pl.col("ids_off").str.strip_chars().cast(pl.Int64, strict=False).alias("player_id"))
        .filter(pl.col("player_id").is_not_null())
        .rename({"seconds_elapsed": "_end_time", "event_num": "_end_event"})
        .sort(["_end_time"] + keys + ["_end_event"])
    )
    ids = ["season", "game_id", "period", "seconds_elapsed", "_toi_row"]
    long = pl.concat(
        [
            pbp.select(ids + home)
            .unpivot(on=home, index=ids, variable_name="position", value_name="player_id")
            .with_columns(pl.lit("home").alias("venue")),
            pbp.select(ids + away)
            .unpivot(on=away, index=ids, variable_name="position", value_name="player_id")
            .with_columns(pl.lit("away").alias("venue")),
        ],
        how="vertical_relaxed",
    ).filter(pl.col("player_id").is_not_null()).sort(["seconds_elapsed"] + keys + ["_toi_row"])
    long = long.join_asof(
        starts,
        left_on="seconds_elapsed",
        right_on="_start_time",
        by=keys,
        strategy="backward",
    ).join_asof(
        ends,
        left_on="seconds_elapsed",
        right_on="_end_time",
        by=keys,
        strategy="backward",
    )
    long = long.with_columns(
        pl.when(
            pl.col("_start_time").is_not_null()
            & (
                pl.col("_end_time").is_null()
                | (pl.col("_end_time") <= pl.col("_start_time"))
                | (pl.col("_end_time") >= pl.col("seconds_elapsed"))
            )
        )
        .then(pl.col("seconds_elapsed") - pl.col("_start_time"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("shift_time_on_ice")
    )
    shooter = pbp.filter(pl.col("event_player_1_id").is_not_null()).select(
        ["season", "game_id", "period", "seconds_elapsed", "_toi_row", pl.col("event_player_1_id").alias("player_id")]
    ).sort(["seconds_elapsed", "season", "game_id", "period", "player_id", "_toi_row"])
    shooter = shooter.join_asof(
        starts,
        left_on="seconds_elapsed",
        right_on="_start_time",
        by=keys,
        strategy="backward",
    ).join_asof(
        ends,
        left_on="seconds_elapsed",
        right_on="_end_time",
        by=keys,
        strategy="backward",
    ).with_columns(
        pl.when(
            pl.col("_start_time").is_not_null()
            & (
                pl.col("_end_time").is_null()
                | (pl.col("_end_time") <= pl.col("_start_time"))
                | (pl.col("_end_time") >= pl.col("seconds_elapsed"))
            )
        )
        .then(pl.col("seconds_elapsed") - pl.col("_start_time"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias(toi[0])
    ).select(["_toi_row", toi[0]])
    team_toi = (
        long.select(["_toi_row", "venue", "shift_time_on_ice"])
        .join(pbp.select(["_toi_row", "event_team_venue"]), on="_toi_row", how="left")
        .filter(pl.col("venue") == pl.col("event_team_venue"))
        .group_by("_toi_row")
        .agg(pl.col("shift_time_on_ice").mean().alias("_event_team_avg_toi"))
    )
    pbp = (
        pbp.join(shooter, on="_toi_row", how="left")
        .join(team_toi, on="_toi_row", how="left")
        .with_columns(
            pl.coalesce([pl.col(toi[0]), pl.col("_event_team_avg_toi")]).alias(toi[0])
        )
    )
    against = (
        long.select(["_toi_row", "venue", "shift_time_on_ice"])
        .join(pbp.select(["_toi_row", "event_team_venue"]), on="_toi_row", how="left")
        .filter(
            ((pl.col("event_team_venue") == "away") & (pl.col("venue") == "home"))
            | ((pl.col("event_team_venue") == "home") & (pl.col("venue") == "away"))
        )
        .group_by("_toi_row")
        .agg(
            [
                pl.col("shift_time_on_ice").min().alias(toi[1]),
                pl.col("shift_time_on_ice").mean().alias(toi[2]),
                pl.col("shift_time_on_ice").max().alias(toi[3]),
            ]
        )
    )
    return pbp.join(against, on="_toi_row", how="left").drop(
        ["_toi_row", "_event_team_avg_toi"]
    )


def prep_xG_data(pbp):
    frame, was_pandas = _as_polars(pbp)
    frame = fix_players(frame)
    order = ["season", "game_id", "period", "seconds_elapsed", "event_num"]
    groups = ["season", "game_id", "period"]
    frame = (
        frame.sort(order)
        .with_columns(
            (
                pl.col("seconds_elapsed")
                - pl.col("seconds_elapsed").shift(1).over(groups)
            )
            .fill_null(0)
            .alias("seconds_since_last")
        )
        .with_columns(
            [
                pl.col("event_team_abbr")
                .shift(1)
                .over(groups)
                .alias("event_team_last"),
                pl.col("event_type").shift(1).over(groups).alias("event_type_last"),
                pl.col("x_adj").shift(1).over(groups).alias("x_adj_last"),
                pl.col("y_adj").shift(1).over(groups).alias("y_adj_last"),
                pl.col("zone_code").shift(1).over(groups).alias("zone_code_last"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("away_team_abbr") == pl.col("event_team_abbr"))
                .then(pl.col("away_score") - pl.col("home_score"))
                .otherwise(pl.col("home_score") - pl.col("away_score"))
                .clip(-4, 4)
                .alias("score_state"),
                pl.when(pl.col("away_team_abbr") == pl.col("event_team_abbr"))
                .then(pl.col("away_skaters") - pl.col("home_skaters"))
                .otherwise(pl.col("home_skaters") - pl.col("away_skaters"))
                .alias("strength_diff"),
                (
                    pl.col("away_skaters").cast(pl.String)
                    + pl.lit("v")
                    + pl.col("home_skaters").cast(pl.String)
                ).alias("strength_state_venue"),
                (
                    (pl.col("x_adj") - pl.col("x_adj_last")).pow(2)
                    + (pl.col("y_adj") - pl.col("y_adj_last")).pow(2)
                )
                .sqrt()
                .alias("distance_from_last"),
                pl.arctan2d(
                    (pl.col("y_adj") - pl.col("y_adj_last")).abs(),
                    (pl.col("x_adj") - pl.col("x_adj_last")).abs(),
                ).alias("angle_from_last"),
            ]
        )
        .with_columns(
            [
                pl.arctan2d(
                    pl.col("y_adj").abs(),
                    pl.when(pl.col("event_team_venue") == "home")
                    .then((89.0 - pl.col("x_adj")).abs())
                    .when(pl.col("event_team_venue") == "away")
                    .then((-89.0 - pl.col("x_adj")).abs())
                    .otherwise(None),
                ).alias("event_angle"),
                pl.when(pl.col("seconds_since_last") == 0)
                .then(0.0)
                .when(
                    pl.col("distance_from_last").is_null()
                    | pl.col("seconds_since_last").is_null()
                )
                .then(None)
                .otherwise(pl.col("distance_from_last") / pl.col("seconds_since_last"))
                .alias("speed_from_last"),
                pl.when(pl.col("seconds_since_last") == 0)
                .then(0.0)
                .when(
                    pl.col("angle_from_last").is_null()
                    | pl.col("seconds_since_last").is_null()
                )
                .then(None)
                .otherwise(pl.col("angle_from_last") / pl.col("seconds_since_last"))
                .alias("speed_of_angle_from_last"),
            ]
        )
    )
    fenwick = pl.col("event_type").is_in(wsba.FENWICK_EVENTS)
    frame = frame.with_columns(
        [
            pl.when(
                fenwick
                & pl.col("zone_code_last").is_in(["N", "D"])
                & (pl.col("zone_code") == "O")
                & (pl.col("seconds_since_last") <= 5)
            )
            .then(1)
            .otherwise(0)
            .alias("rush"),
            pl.when(
                fenwick
                & (pl.col("zone_code_last") == "O")
                & (pl.col("zone_code") == "O")
                & (pl.col("seconds_since_last") <= 5)
            )
            .then(1)
            .otherwise(0)
            .alias("in_zone"),
            pl.when(
                fenwick
                & pl.col("event_type_last").is_in(wsba.FENWICK_EVENTS)
                & (pl.col("seconds_since_last") <= 2)
            )
            .then(1)
            .otherwise(0)
            .alias("rebound"),
            (pl.col("event_type") == "goal")
            .fill_null(False)
            .cast(pl.Int64)
            .alias("is_goal"),
            (pl.col("home_team_abbr") == pl.col("event_team_abbr"))
            .fill_null(False)
            .cast(pl.Int64)
            .alias("is_home"),
        ]
    )
    for shot in wsba.SHOT_TYPES:
        frame = frame.with_columns(
            (pl.col("shot_type") == shot).fill_null(False).cast(pl.Int64).alias(shot)
        )
    for event in wsba.EVENTS[:-1]:
        frame = frame.with_columns(
            (pl.col("event_type_last") == event)
            .fill_null(False)
            .cast(pl.Int64)
            .alias(f"prior_{event}")
        )
    frame = frame.with_columns(
        [
            (~pl.col("shot_type").is_in(wsba.SHOT_TYPES))
            .fill_null(True)
            .cast(pl.Int64)
            .alias("other-shot"),
            (pl.col("event_team_last") == pl.col("event_team_abbr"))
            .fill_null(False)
            .cast(pl.Int64)
            .alias("prior_same"),
        ]
    )
    for strength in wsba.STRENGTHS:
        frame = frame.with_columns(
            (pl.col("strength_state") == strength)
            .fill_null(False)
            .cast(pl.Int64)
            .alias(f"strength_{strength}")
        )
    frame = frame.with_columns(
        [
            (pl.col("event_reason") == "short")
            .fill_null(False)
            .cast(pl.Int64)
            .alias("short"),
            (pl.col("event_reason") == "failed-bank-attempt")
            .fill_null(False)
            .cast(pl.Int64)
            .alias("failed_bank"),
            (pl.col("y_adj_last") * pl.col("y_adj") < 0)
            .fill_null(False)
            .cast(pl.Int64)
            .alias("cross_ice"),
            pl.when(fenwick & pl.col("event_goalie_id").is_null())
            .then(1)
            .otherwise(0)
            .alias("empty_net"),
            pl.when(
                ((pl.col("y_adj") < 0) & (pl.col("event_player_1_hand") == "L"))
                | ((pl.col("y_adj") >= 0) & (pl.col("event_player_1_hand") == "R"))
            )
            .then(1)
            .otherwise(0)
            .alias("offwing"),
        ]
    )
    toi_features = [
        "event_player_1_shift_time_on_ice",
        "event_on_ice_against_min_shift_time_on_ice",
        "event_on_ice_against_avg_shift_time_on_ice",
        "event_on_ice_against_max_shift_time_on_ice",
    ]
    if not all(feature in frame.columns for feature in toi_features):
        frame = apply_time_on_ice(frame)
    return pd.DataFrame(frame.to_dicts()) if was_pandas else frame


def prepare_xG_shots(pbp):
    frame, _ = _as_polars(pbp)
    if "event_index" not in frame.columns:
        frame = frame.with_row_index("event_index")
    frame = apply_time_on_ice(frame)
    frame = frame.filter(pl.col("event_type").is_in(wsba.EVENTS))
    frame = prep_xG_data(frame)
    candidate = frame.filter(
        pl.col("strength_state").is_in(wsba.STRENGTHS)
        & pl.col("event_distance").is_not_null()
        & pl.col("event_angle").is_not_null()
    )
    return _exclude_outcome_leaking_shots(
        candidate.filter(pl.col("event_type").is_in(wsba.FENWICK_EVENTS))
    )


def _model_features(model, model_path):
    sidecar = Path(model_path).with_name(Path(model_path).stem + "_features.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    return model.feature_names or list(FEATURES)


def _model_population(frame, name):
    if name not in MODEL_STRENGTHS:
        raise ValueError(f"Unknown xG model group: {name}")

    model_data = frame.filter(pl.col("strength_state").is_in(MODEL_STRENGTHS[name]))
    if name == "en":
        model_data = model_data.filter(pl.col("empty_net") == 1)
    else:
        model_data = model_data.filter(pl.col("empty_net") == 0)
    return model_data


def wsba_xG(
    pbp,
    features=FEATURES,
    target=TARGET,
    tune_path=None,
    model_path=None,
    tune_runs=20,
    strengths=None,
    empty_net=False,
    gpu=None,
    hypertune=True,
    hyperparameters=None,
    tune_parameters=None,
    parameter_space=None,
    train=True,
    model_dir=MODEL_DIR,
    model_names=None,
):
    frame, _ = _as_polars(pbp)
    if not train:
        if "event_index" not in frame.columns:
            frame = frame.with_row_index("event_index")
        output = frame.select(["event_index"]).clone()
        prediction_columns = []
        names = list(DEFAULT_MODEL_NAMES if model_names is None else model_names)
        for name in names:
            model_data = _model_population(frame, name)
            if model_data.is_empty():
                continue
            model_path_for_group = Path(model_dir) / f"wsba_xg_model_{name}.json"
            model = xgb.Booster()
            model.load_model(str(model_path_for_group))
            model_features = _model_features(model, model_path_for_group)
            missing = [
                feature
                for feature in model_features
                if feature not in model_data.columns
            ]
            if missing:
                model_data = model_data.with_columns(
                    [
                        pl.lit(None).cast(pl.Float32).alias(feature)
                        for feature in missing
                    ]
                )
            matrix = sp.csr_matrix(
                model_data.select(model_features)
                .fill_null(float("nan"))
                .to_numpy()
                .astype(np.float32, copy=False)
            )
            column = f"xG_{name}"
            prediction_columns.append(column)
            predictions = np.clip(
                model.predict(xgb.DMatrix(matrix, feature_names=model_features)),
                0.0,
                1.0,
            )
            output = output.join(
                pl.DataFrame(
                    {"event_index": model_data["event_index"], column: predictions}
                ),
                on="event_index",
                how="left",
            )
        if prediction_columns:
            return output.with_columns(
                pl.coalesce([pl.col(column) for column in prediction_columns]).alias(
                    "wsba_xg"
                )
            )
        return output.with_columns(pl.lit(None).cast(pl.Float32).alias("wsba_xg"))

    feature_names = list(dict.fromkeys(features))
    manual = dict(hyperparameters or {})
    space = parameter_space or HYPERPARAMETER_SPACE
    tuned = list(TUNED_HYPERPARAMETERS if tune_parameters is None else tune_parameters)
    unknown = [name for name in tuned if name not in space]
    if unknown:
        raise ValueError(f"Unknown tuning parameters: {unknown}")
    if strengths is not None:
        frame = frame.filter(pl.col("strength_state").is_in(strengths))
    if empty_net is not None:
        frame = frame.filter(pl.col("empty_net") == int(empty_net))
    frame = _exclude_outcome_leaking_shots(frame)
    frame = frame.filter(pl.col(target).is_not_null())

    missing = [
        column for column in [target, *feature_names] if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing model columns: {missing}")
    data = frame.select([target, *feature_names]).with_columns(
        [pl.col(c).cast(pl.Float32, strict=False) for c in [target, *feature_names]]
    )
    feature_names = []
    dropped_features = []
    for feature in data.columns:
        if feature == target:
            continue

        values = data.get_column(feature).fill_nan(None).drop_nulls()
        if values.n_unique() <= 1:
            dropped_features.append(feature)
        else:
            feature_names.append(feature)

    if dropped_features:
        print(f"Dropped single-level features: {dropped_features}")
    data = data.select([target, *feature_names])
    sparse = sp.csr_matrix(data.fill_null(float("nan")).to_numpy())
    labels = sparse[:, 0].toarray().ravel()
    predictors = sparse[:, 1:]
    build_info = getattr(xgb, "build_info", lambda: {})()
    gpu = (
        torch.cuda.is_available()
        and str(build_info.get("USE_CUDA", "")).upper() in {"ON", "TRUE", "1"}
        if gpu is None
        else bool(gpu)
    )
    device = {"tree_method": "hist", "device": "cuda" if gpu else "cpu"}
    base = {
        **device,
        **manual,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }
    fixed_seed = int(manual.get("seed", 42))
    dtrain = xgb.DMatrix(predictors, label=labels, feature_names=feature_names)
    groups = frame.get_column("game_id").to_numpy()
    folds = list(
        StratifiedGroupKFold(
            n_splits=min(10, len(np.unique(groups))), shuffle=True, random_state=42
        ).split(predictors, labels, groups=groups)
    )
    print("Tuning parameters:", tuned)
    print("Manual parameter overrides:", manual)
    print("Search space:", space)
    if hypertune:
        def suggest_parameter(trial, name, spec):
            if spec["kind"] == "uniform":
                return trial.suggest_float(name, spec["low"], spec["high"])
            if spec["kind"] == "int":
                return trial.suggest_int(name, spec["low"], spec["high"])
            if spec["kind"] == "choice":
                return trial.suggest_categorical(name, spec["values"])
            if spec["kind"] == "loguniform":
                return trial.suggest_float(
                    name, spec["low"], spec["high"], log=True
                )
            raise ValueError(
                f"Unknown search-space kind for {name}: {spec['kind']}"
            )

        def objective(trial):
            print(f"## LOOP: {trial.number + 1} ##")
            values = {}
            for name, spec in space.items():
                if name in manual:
                    values[name] = manual[name]
                elif name in tuned:
                    values[name] = suggest_parameter(trial, name, spec)
                else:
                    values[name] = spec["default"]
            params = {
                **base,
                **values,
                "seed": fixed_seed,
                "eval_metric": "logloss",
            }
            cv = pl.from_pandas(
                xgb.cv(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=1000,
                    folds=folds,
                    early_stopping_rounds=25,
                    metrics=["logloss"],
                    maximize=False,
                    seed=fixed_seed,
                )
            )
            ll = cv["test-logloss-mean"].arg_min()
            trial_log_loss = float(cv["test-logloss-mean"][ll])
            trial.set_user_attr("LL_rounds", int(ll + 1))
            trial.set_user_attr("params", values)
            print(f"# Log Loss: {trial_log_loss:.6f} #")
            return trial_log_loss

        print("### CROSS-VALIDATED OPTUNA TUNING ###")
        if tune_runs <= 0:
            raise ValueError("tune_runs must be positive when hypertune=True")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=fixed_seed),
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=int(tune_runs), show_progress_bar=False)
        tuning = []
        for trial in study.trials:
            values = {
                name: trial.user_attrs["params"][name]
                for name in space
            }
            tuning.append(
                {
                    "trial": trial.number + 1,
                    "n_rows": frame.height,
                    "n_games": frame.get_column("game_id").n_unique(),
                    **_export_parameters(values),
                    "seed": fixed_seed,
                    "LL": float(trial.value),
                    "LL_rounds": trial.user_attrs["LL_rounds"],
                }
            )
        tuning = pl.DataFrame(tuning).sort("LL", descending=False)
        if tune_path is not None:
            Path(tune_path).parent.mkdir(parents=True, exist_ok=True)
            tuning.write_csv(tune_path)
        best = tuning.row(0, named=True)
        params_grid = {**base, **{name: best[name] for name in space}}
        for name, spec in space.items():
            if spec["kind"] == "int":
                params_grid[name] = int(params_grid[name])
        cv_final = tuning
    else:
        if tune_path is None:
            raise ValueError("tune_path is required when hypertune=False")
        best = pl.read_csv(tune_path).sort("LL").row(0, named=True)
        cv_final = pl.DataFrame([best])
        params_grid = {
            **base,
            **{name: best.get(name, spec["default"]) for name, spec in space.items()},
        }
        params_grid.update(manual)
        params_grid["eval_metric"] = "logloss"
        for name, spec in space.items():
            if spec["kind"] == "int":
                params_grid[name] = int(params_grid[name])
    seed = fixed_seed if hypertune else int(cv_final.get_column("seed").item(0))
    rounds = int(cv_final.get_column("LL_rounds").item(0))
    final_params = {**params_grid, "seed": seed, "eval_metric": "logloss"}
    print("Training model...")
    print("Selected params:")
    print(final_params)
    model = xgb.train(final_params, dtrain, num_boost_round=rounds, verbose_eval=2)
    model_path = Path(model_path or MODEL_DIR / "wsba_xg_model_all.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    model_path.with_name(model_path.stem + "_features.json").write_text(
        json.dumps(feature_names, indent=2)
    )
    return model


def evaluate_xg_model(
    shots,
    model_dir=MODEL_DIR,
    plot_dir=XG_MODEL_DIR,
    model_names=None,
    metrics_path=None,
    shap_random_state=42,
):
    import matplotlib.pyplot as plt
    import shap
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
        roc_curve,
    )

    frame, _ = _as_polars(shots)
    frame = _exclude_outcome_leaking_shots(frame)
    if "event_index" not in frame.columns:
        frame = frame.with_row_index("event_index")
    names = list(DEFAULT_MODEL_NAMES if model_names is None else model_names)
    predictions = wsba_xG(frame, train=False, model_dir=model_dir, model_names=names)
    scored = frame.join(predictions, on="event_index", how="left")
    aggregate = scored.filter(pl.col("wsba_xg").is_not_null())
    if aggregate.is_empty():
        raise ValueError("No held-out shots were scored by the available xG models")
    y = aggregate.get_column(TARGET).to_numpy().astype(np.int8, copy=False)
    probabilities = (
        aggregate.get_column("wsba_xg").to_numpy().astype(np.float64, copy=False)
    )
    fpr, tpr, _ = roc_curve(y, probabilities)
    auc_value = float(roc_auc_score(y, probabilities))
    output_metrics = {
        "n_rows": int(aggregate.height),
        "n_goals": int(y.sum()),
        "n_games": int(aggregate.get_column("game_id").n_unique()),
        "roc_auc": auc_value,
        "pr_auc": float(average_precision_score(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "total_xg": float(probabilities.sum()),
        "calibration": float(y.sum() / probabilities.sum()),
        "min_xg": float(probabilities.min()),
        "max_xg": float(probabilities.max()),
    }

    plot_dir = Path(plot_dir)
    dependence_dir = plot_dir / "dependence"
    plot_dir.mkdir(parents=True, exist_ok=True)
    dependence_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}", color="#2f6f73")
    axis.plot([0, 1], [0, 1], "--", color="#777777")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("WSBA xG ROC-AUC")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "wsba_xg_roc_auc.png", dpi=160)
    plt.close(figure)

    calibration_bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.clip(
        np.digitize(probabilities, calibration_bins) - 1, 0, len(calibration_bins) - 2
    )
    calibration = []
    for bin_id, (lower, upper) in enumerate(
        zip(calibration_bins[:-1], calibration_bins[1:])
    ):
        mask = bin_ids == bin_id
        if np.any(mask):
            calibration.append(
                (
                    float(probabilities[mask].mean()),
                    float(y[mask].mean()),
                    int(mask.sum()),
                )
            )
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.plot(
        [row[0] for row in calibration],
        [row[1] for row in calibration],
        "s-",
        color="#2f6f73",
        label="held-out test",
    )
    axis.plot([0, 1], [0, 1], "--", color="#777777", label="Perfect calibration")
    axis.set_xlabel("Mean Predicted Probability")
    axis.set_ylabel("Observed Goal Rate")
    axis.set_title("WSBA xG Calibration")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "wsba_xg_calibration.png", dpi=160)
    plt.close(figure)
    output_metrics["calibration_bins"] = [
        {"mean_predicted": row[0], "observed_goal_rate": row[1], "n_shots": row[2]}
        for row in calibration
    ]

    # Explain each held-out shot with the model that produced its score.
    row_ids = aggregate.get_column("event_index").to_list()
    row_position = {row_id: index for index, row_id in enumerate(row_ids)}
    model_rows = []
    all_features = []
    for name in names:
        group_rows = _model_population(aggregate, name)
        if group_rows.is_empty():
            continue
        model_path_for_group = Path(model_dir) / f"wsba_xg_model_{name}.json"
        model = xgb.Booster()
        model.load_model(str(model_path_for_group))
        model_features = _model_features(model, model_path_for_group)
        missing = [
            feature for feature in model_features if feature not in group_rows.columns
        ]
        if missing:
            group_rows = group_rows.with_columns(
                [pl.lit(None).cast(pl.Float32).alias(feature) for feature in missing]
            )
        matrix = (
            group_rows.select(model_features)
            .fill_null(float("nan"))
            .to_numpy()
            .astype(np.float32, copy=False)
        )
        explanation = shap.TreeExplainer(model)(
            pd.DataFrame(matrix, columns=model_features), check_additivity=False
        )
        values = np.asarray(explanation.values, dtype=np.float32)
        if values.shape != matrix.shape:
            raise ValueError(
                f"Unexpected SHAP matrix shape {values.shape}; expected {matrix.shape}"
            )
        base_values = np.asarray(explanation.base_values, dtype=np.float32)
        if base_values.ndim == 0:
            base_values = np.full(len(group_rows), float(base_values), dtype=np.float32)
        else:
            base_values = base_values.reshape(-1)
        if base_values.size != len(group_rows):
            raise ValueError(f"Unexpected SHAP base-value shape {base_values.shape}")
        group_positions = np.asarray(
            [row_position[row_id] for row_id in group_rows["event_index"].to_list()],
            dtype=np.int64,
        )
        model_rows.append((group_positions, model_features, values, base_values))
        all_features.extend(
            feature for feature in model_features if feature not in all_features
        )

    if not all_features or sum(len(rows[0]) for rows in model_rows) != aggregate.height:
        raise ValueError("SHAP explanations must cover every scored held-out shot")
    all_features = list(all_features)
    feature_index = {feature: index for index, feature in enumerate(all_features)}
    shap_values = np.zeros((aggregate.height, len(all_features)), dtype=np.float32)
    shap_base_values = np.zeros(aggregate.height, dtype=np.float32)
    feature_frame = aggregate
    missing = [
        feature for feature in all_features if feature not in feature_frame.columns
    ]
    if missing:
        feature_frame = feature_frame.with_columns(
            [pl.lit(None).cast(pl.Float32).alias(feature) for feature in missing]
        )
    feature_values = (
        feature_frame.select(all_features)
        .fill_null(float("nan"))
        .to_numpy()
        .astype(np.float32, copy=False)
    )
    for positions, model_features, values, base_values in model_rows:
        feature_positions = [feature_index[feature] for feature in model_features]
        shap_values[np.ix_(positions, feature_positions)] = values
        shap_base_values[positions] = base_values

    goal_positions = np.flatnonzero(y == 1)
    if len(goal_positions) == 0:
        raise ValueError(
            "Cannot create the requested goal waterfall: held-out data has no goals"
        )
    sample_position = int(
        np.random.default_rng(shap_random_state).choice(goal_positions)
    )
    waterfall = shap.Explanation(
        values=shap_values[sample_position],
        base_values=shap_base_values[sample_position],
        data=feature_values[sample_position],
        feature_names=all_features,
    )
    shap.plots.waterfall(waterfall, max_display=min(14, len(all_features)), show=False)
    figure = plt.gcf()
    figure.set_size_inches(10, 7)
    figure.suptitle("WSBA xG Sample Goal SHAP Waterfall", y=1.02)
    figure.tight_layout()
    figure.savefig(
        plot_dir / "wsba_xg_sample_waterfall.png", dpi=160, bbox_inches="tight"
    )
    plt.close(figure)

    importance = shap.Explanation(
        values=shap_values,
        data=feature_values,
        feature_names=all_features,
    )
    shap.plots.beeswarm(
        importance,
        max_display=min(20, len(all_features)),
        show=False,
    )
    figure = plt.gcf()
    figure.set_size_inches(10, 7)
    plt.title("WSBA xG Feature Importance")
    figure.tight_layout()
    figure.savefig(plot_dir / "wsba_xg_feature_importance.png", dpi=160)
    plt.close(figure)

    feature_values_frame = pd.DataFrame(feature_values, columns=all_features)
    for feature in all_features:
        values = feature_values_frame[feature].to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        if np.unique(values[np.isfinite(values)]).size < 2:
            continue
        try:
            shap.dependence_plot(
                feature,
                shap_values,
                feature_values,
                feature_names=all_features,
                interaction_index="auto",
                show=False,
            )
        except (ValueError, IndexError):
            plt.close("all")
            continue
        figure = plt.gcf()
        figure.set_size_inches(9, 7)
        plt.title(f"WSBA xG SHAP Dependence - {feature}")
        figure.tight_layout()
        figure.savefig(dependence_dir / f"wsba_xg_{feature}_dependence.png", dpi=160)
        plt.close(figure)

    metrics_file = (
        Path(metrics_path)
        if metrics_path is not None
        else plot_dir / "wsba_xg_metrics.json"
    )
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {"metrics": output_metrics}
    metrics_file.write_text(json.dumps(metrics_payload, indent=2))
    return metrics_payload


def apply_xG(pbp, model_dir=MODEL_DIR):
    frame, was_pandas = _as_polars(pbp)
    if "event_index" not in frame.columns:
        frame = frame.with_row_index("event_index")
    predictions = wsba_xG(
        prepare_xG_shots(frame), train=False, model_dir=model_dir
    ).select(["event_index", "wsba_xg"])
    result = frame.join(predictions, on="event_index", how="left").with_columns(
        pl.col("wsba_xg").fill_null(0.0).alias("xG")
    )
    if was_pandas:
        output = pd.DataFrame(result.to_dicts())
        output.index = pd.DataFrame(pbp).index
        return output
    return result
