"""WSBA expected-goals model.

Python version of the corrected xG notebook. Training and prediction use the
same Polars preparation and CSR XGBoost matrices.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
import torch
import xgboost as xgb
import wsba_hockey as wsba
from sklearn.model_selection import GroupKFold
from wsba_hockey.tools.scraping import fix_players


TARGET = "is_goal"
FEATURES = [
    "event_distance", "event_angle", "distance_from_last", "angle_from_last",
    "seconds_since_last", "speed_from_last", "speed_of_angle_from_last",
    "event_player_1_shift_time_on_ice",
    "event_on_ice_against_min_shift_time_on_ice",
    "event_on_ice_against_avg_shift_time_on_ice",
    "event_on_ice_against_max_shift_time_on_ice",
    "wrist", "deflected", "tip-in", "slap", "backhand", "snap",
    "wrap-around", "poke", "bat", "cradle", "between-legs", "other-shot",
    "prior_same", "prior_shot-on-goal", "prior_missed-shot", "prior_blocked-shot",
    "prior_giveaway", "prior_takeaway", "prior_hit", "prior_faceoff",
    "strength_3v3", "strength_3v4", "strength_3v5", "strength_4v3", "strength_4v4",
    "strength_4v5", "strength_4v6", "strength_5v3", "strength_5v4", "strength_5v5",
    "strength_5v6", "strength_6v4", "strength_6v5", "empty_net", "offwing",
    "short", "failed_bank", "cross_ice", "rebound", "rush",
]
MODEL_STRENGTHS = {
    "ev": wsba.STRENGTH_GROUPS["even_strength"],
    "pp": wsba.STRENGTH_GROUPS["power_play"] + ["6v4", "6v5"],
    "pk": wsba.STRENGTH_GROUPS["penalty_kill"],
    "en": wsba.STRENGTHS,
}

XG_MODEL_DIR = Path(__file__).resolve().parent / "xg_model"
MODEL_DIR = XG_MODEL_DIR / "models"
TUNING_DIR = XG_MODEL_DIR / "tuning"
CV_DIR = XG_MODEL_DIR / "cv"
METRICS_PATH = XG_MODEL_DIR / "wsba_xg_overall_metrics.json"

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
    "grow_policy": {"kind": "choice", "values": ["depthwise", "lossguide"], "default": "depthwise"},
    "max_leaves": {"kind": "int", "low": 0, "high": 128, "default": 0},
    "scale_pos_weight": {"kind": "uniform", "low": 1.0, "high": 12.0, "default": 1.0},
}
TUNED_HYPERPARAMETERS = [
    "eta", "gamma", "max_depth", "min_child_weight", "max_delta_step",
    "subsample", "colsample_bytree",
]
HYPERPARAMETERS = {name: {} for name in MODEL_STRENGTHS}


def _as_polars(frame):
    if isinstance(frame, pl.DataFrame):
        return frame.clone(), False
    if isinstance(frame, pd.DataFrame):
        converted = pl.DataFrame(frame.to_dict(orient="records"), infer_schema_length=None)
        return converted.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None)), True
    raise TypeError("xG data must be a Polars or Pandas DataFrame")


def _sample_parameter(name, parameter_space):
    spec = parameter_space[name]
    if spec["kind"] == "uniform":
        return float(np.random.uniform(spec["low"], spec["high"]))
    if spec["kind"] == "int":
        return int(np.random.randint(spec["low"], spec["high"] + 1))
    if spec["kind"] == "choice":
        return np.random.choice(spec["values"]).item()
    if spec["kind"] == "loguniform":
        return float(np.exp(np.random.uniform(np.log(spec["low"]), np.log(spec["high"]))))
    raise ValueError(f"Unknown search-space kind for {name}: {spec['kind']}")


def _export_parameters(params):
    return {key: ",".join(value) if isinstance(value, list) else value for key, value in params.items()}


def apply_time_on_ice(pbp):
    toi = [
        "event_player_1_shift_time_on_ice",
        "event_on_ice_against_min_shift_time_on_ice",
        "event_on_ice_against_avg_shift_time_on_ice",
        "event_on_ice_against_max_shift_time_on_ice",
    ]
    home = [f"home_on_{i}_id" for i in range(1, 7)]
    away = [f"away_on_{i}_id" for i in range(1, 7)]
    pbp = pbp.drop(["_toi_row", *toi], strict=False).with_columns([
        pl.col("event_player_1_id").cast(pl.Int64, strict=False),
        *[pl.col(c).cast(pl.Int64, strict=False) for c in home + away],
    ]).with_row_index("_toi_row")
    ids = ["season", "game_id", "period", "seconds_elapsed", "_toi_row"]
    long = pl.concat([
        pbp.select(ids + home).unpivot(on=home, index=ids, variable_name="position", value_name="player_id").with_columns(pl.lit("home").alias("venue")),
        pbp.select(ids + away).unpivot(on=away, index=ids, variable_name="position", value_name="player_id").with_columns(pl.lit("away").alias("venue")),
    ], how="vertical_relaxed").filter(pl.col("player_id").is_not_null()).sort(["season", "game_id", "period", "player_id", "_toi_row"])
    player_group = ["season", "game_id", "period", "player_id"]
    long = long.with_columns(pl.col("_toi_row").shift(1).over(player_group).alias("_prev_row")).with_columns(
        (pl.col("_prev_row").is_null() | (pl.col("_toi_row") != pl.col("_prev_row") + 1)).cast(pl.Int64).alias("_new_shift")
    ).with_columns(pl.col("_new_shift").cum_sum().over(player_group).alias("_shift_id"))
    shift_group = player_group + ["_shift_id"]
    long = long.with_columns(pl.col("seconds_elapsed").first().over(shift_group).alias("_shift_start")).with_columns(
        (pl.col("seconds_elapsed") - pl.col("_shift_start")).alias("shift_time_on_ice")
    )
    shooter = long.select(["_toi_row", "player_id", "shift_time_on_ice"]).join(
        pbp.select(["_toi_row", "event_player_1_id"]),
        left_on=["_toi_row", "player_id"], right_on=["_toi_row", "event_player_1_id"], how="inner",
    ).select(["_toi_row", pl.col("shift_time_on_ice").alias(toi[0])])
    pbp = pbp.join(shooter, on="_toi_row", how="left")
    against = long.select(["_toi_row", "venue", "shift_time_on_ice"]).join(
        pbp.select(["_toi_row", "event_team_venue"]), on="_toi_row", how="left"
    ).filter(
        ((pl.col("event_team_venue") == "away") & (pl.col("venue") == "home"))
        | ((pl.col("event_team_venue") == "home") & (pl.col("venue") == "away"))
    ).group_by("_toi_row").agg([
        pl.col("shift_time_on_ice").min().alias(toi[1]),
        pl.col("shift_time_on_ice").mean().alias(toi[2]),
        pl.col("shift_time_on_ice").max().alias(toi[3]),
    ])
    return pbp.join(against, on="_toi_row", how="left").drop("_toi_row")


def prep_xG_data(pbp):
    frame, was_pandas = _as_polars(pbp)
    frame = fix_players(frame)
    order = ["season", "game_id", "period", "seconds_elapsed", "event_num"]
    groups = ["season", "game_id", "period"]
    frame = frame.sort(order).with_columns(
        (pl.col("seconds_elapsed") - pl.col("seconds_elapsed").shift(1).over(groups)).fill_null(0).alias("seconds_since_last")
    ).with_columns([
        pl.col("event_team_abbr").shift(1).over(groups).alias("event_team_last"),
        pl.col("event_type").shift(1).over(groups).alias("event_type_last"),
        pl.col("x_adj").shift(1).over(groups).alias("x_adj_last"),
        pl.col("y_adj").shift(1).over(groups).alias("y_adj_last"),
        pl.col("zone_code").shift(1).over(groups).alias("zone_code_last"),
    ]).with_columns([
        pl.when(pl.col("away_team_abbr") == pl.col("event_team_abbr")).then(pl.col("away_score") - pl.col("home_score")).otherwise(pl.col("home_score") - pl.col("away_score")).clip(-4, 4).alias("score_state"),
        pl.when(pl.col("away_team_abbr") == pl.col("event_team_abbr")).then(pl.col("away_skaters") - pl.col("home_skaters")).otherwise(pl.col("home_skaters") - pl.col("away_skaters")).alias("strength_diff"),
        (pl.col("away_skaters").cast(pl.String) + pl.lit("v") + pl.col("home_skaters").cast(pl.String)).alias("strength_state_venue"),
        ((pl.col("x_adj") - pl.col("x_adj_last")).pow(2) + (pl.col("y_adj") - pl.col("y_adj_last")).pow(2)).sqrt().alias("distance_from_last"),
        pl.arctan2d((pl.col("y_adj") - pl.col("y_adj_last")).abs(), (89 - (pl.col("x_adj") - pl.col("x_adj_last"))).abs()).alias("angle_from_last"),
    ]).with_columns([
        pl.when(pl.col("seconds_since_last") == 0).then(0.0).when(pl.col("distance_from_last").is_null() | pl.col("seconds_since_last").is_null()).then(None).otherwise(pl.col("distance_from_last") / pl.col("seconds_since_last")).alias("speed_from_last"),
        pl.when(pl.col("seconds_since_last") == 0).then(0.0).when(pl.col("angle_from_last").is_null() | pl.col("seconds_since_last").is_null()).then(None).otherwise(pl.col("angle_from_last") / pl.col("seconds_since_last")).alias("speed_of_angle_from_last"),
    ])
    fenwick = pl.col("event_type").is_in(wsba.FENWICK_EVENTS)
    frame = frame.with_columns([
        pl.when(fenwick & pl.col("zone_code_last").is_in(["N", "D"]) & (pl.col("zone_code") == "O") & (pl.col("seconds_since_last") <= 5)).then(1).otherwise(0).alias("rush"),
        pl.when(fenwick & (pl.col("zone_code_last") == "O") & (pl.col("zone_code") == "O") & (pl.col("seconds_since_last") <= 5)).then(1).otherwise(0).alias("in_zone"),
        pl.when(fenwick & pl.col("event_type_last").is_in(wsba.FENWICK_EVENTS) & (pl.col("seconds_since_last") <= 2)).then(1).otherwise(0).alias("rebound"),
        (pl.col("event_type") == "goal").fill_null(False).cast(pl.Int64).alias("is_goal"),
        (pl.col("home_team_abbr") == pl.col("event_team_abbr")).fill_null(False).cast(pl.Int64).alias("is_home"),
    ])
    for shot in wsba.SHOT_TYPES:
        frame = frame.with_columns((pl.col("shot_type") == shot).fill_null(False).cast(pl.Int64).alias(shot))
    for event in wsba.EVENTS[:-1]:
        frame = frame.with_columns((pl.col("event_type_last") == event).fill_null(False).cast(pl.Int64).alias(f"prior_{event}"))
    frame = frame.with_columns([
        (~pl.col("shot_type").is_in(wsba.SHOT_TYPES)).fill_null(True).cast(pl.Int64).alias("other-shot"),
        (pl.col("event_team_last") == pl.col("event_team_abbr")).fill_null(False).cast(pl.Int64).alias("prior_same"),
    ])
    for strength in wsba.STRENGTHS:
        frame = frame.with_columns((pl.col("strength_state") == strength).fill_null(False).cast(pl.Int64).alias(f"strength_{strength}"))
    frame = frame.with_columns([
        (pl.col("event_reason") == "short").fill_null(False).cast(pl.Int64).alias("short"),
        (pl.col("event_reason") == "failed-bank-attempt").fill_null(False).cast(pl.Int64).alias("failed_bank"),
        (pl.col("y_adj_last") * pl.col("y_adj") < 0).fill_null(False).cast(pl.Int64).alias("cross_ice"),
        pl.when(fenwick & pl.col("event_goalie_id").is_null()).then(1).otherwise(0).alias("empty_net"),
        pl.when(((pl.col("y_adj") < 0) & (pl.col("event_player_1_hand") == "L")) | ((pl.col("y_adj") >= 0) & (pl.col("event_player_1_hand") == "R"))).then(1).otherwise(0).alias("offwing"),
    ])
    frame = apply_time_on_ice(frame)
    return pd.DataFrame(frame.to_dicts()) if was_pandas else frame


def prepare_xG_shots(pbp):
    frame, _ = _as_polars(pbp)
    if "event_index" not in frame.columns:
        frame = frame.with_row_index("event_index")
    candidate = frame.filter(
        pl.col("event_type").is_in(wsba.EVENTS)
        & pl.col("strength_state").is_in(wsba.STRENGTHS)
        & pl.col("event_distance").is_not_null()
        & pl.col("event_angle").is_not_null()
    )
    return prep_xG_data(candidate).filter(pl.col("event_type").is_in(wsba.FENWICK_EVENTS))


def wsba_xG(
    pbp, features=FEATURES, target=TARGET, tune_path=None, cv_path=None,
    model_path=None, tune_runs=20, cv_runs=20, strengths=None, empty_net=False,
    gpu=None, hypertune=True, hyperparameters=None, tune_parameters=None,
    parameter_space=None,
):
    frame, _ = _as_polars(pbp)
    feature_names = list(dict.fromkeys(features))
    manual = dict(hyperparameters or {})
    space = parameter_space or HYPERPARAMETER_SPACE
    tuned = list(TUNED_HYPERPARAMETERS if tune_parameters is None else tune_parameters)
    unknown = [name for name in tuned if name not in space]
    if unknown:
        raise ValueError(f"Unknown tuning parameters: {unknown}")
    if strengths is not None:
        frame = frame.filter(pl.col("strength_state").is_in(strengths))
    frame = frame.filter((pl.col("empty_net") == int(empty_net)) & pl.col(target).is_not_null())
    print(f"Training population: {frame.height:,} rows across {frame.get_column('game_id').n_unique():,} games")
    missing = [column for column in [target, *feature_names] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing model columns: {missing}")
    data = frame.select([target, *feature_names]).with_columns([pl.col(c).cast(pl.Float32, strict=False) for c in [target, *feature_names]])
    feature_names = [c for c in feature_names if (data.get_column(c).fill_null(0) != 0).any()]
    data = data.select([target, *feature_names])
    sparse = sp.csr_matrix(data.fill_null(float("nan")).to_numpy())
    labels = sparse[:, 0].toarray().ravel()
    predictors = sparse[:, 1:]
    build_info = getattr(xgb, "build_info", lambda: {})()
    gpu = torch.cuda.is_available() and str(build_info.get("USE_CUDA", "")).upper() in {"ON", "TRUE", "1"} if gpu is None else bool(gpu)
    device = {"tree_method": "hist", "device": "cuda" if gpu else "cpu"}
    base = {"objective": "binary:logistic", "eval_metric": ["logloss", "auc"], **device, **manual}
    fixed_seed = manual.get("seed")
    dtrain = xgb.DMatrix(predictors, label=labels, feature_names=feature_names)
    groups = frame.get_column("game_id").to_numpy()
    folds = list(GroupKFold(n_splits=min(10, len(np.unique(groups)))).split(predictors, labels, groups=groups))
    print("Tuning parameters:", tuned)
    print("Manual parameter overrides:", manual)
    print("Search space:", space)
    if hypertune:
        tuning = []
        print("### HYPERTUNING ###")
        for i in range(tune_runs):
            print(f"## LOOP: {i + 1} ##")
            values = {name: manual[name] if name in manual else _sample_parameter(name, space) if name in tuned else spec["default"] for name, spec in space.items()}
            seed = int(fixed_seed if fixed_seed is not None else np.random.randint(0, 10000))
            params = {**base, **values, "seed": seed}
            cv = pl.from_pandas(xgb.cv(params=params, dtrain=dtrain, num_boost_round=1000, folds=folds, early_stopping_rounds=25, metrics=["logloss", "auc"], seed=seed))
            ll = cv["test-logloss-mean"].arg_min()
            auc = cv["test-auc-mean"].arg_max()
            tuning.append({"trial": i + 1, "n_rows": frame.height, "n_games": frame.get_column("game_id").n_unique(), **_export_parameters(params), "LL": float(cv["test-logloss-mean"][ll]), "LL_rounds": int(ll + 1), "AUC": float(cv["test-auc-mean"][auc]), "AUC_rounds": int(auc + 1)})
        tuning = pl.DataFrame(tuning).sort("AUC", descending=True)
        if tune_path is not None:
            Path(tune_path).parent.mkdir(parents=True, exist_ok=True)
            tuning.write_csv(tune_path)
        best = tuning.row(0, named=True)
        params_grid = {**base, **{name: best[name] for name in space}}
        for name, spec in space.items():
            if spec["kind"] == "int":
                params_grid[name] = int(params_grid[name])
        cv_rows = []
        print("### CROSS-VALIDATION ###")
        for i in range(cv_runs):
            print(f"## LOOP: {i + 1} ##")
            seed = int(fixed_seed if fixed_seed is not None else np.random.randint(0, 10000))
            params = {**params_grid, "seed": seed}
            cv = pl.from_pandas(xgb.cv(params=params, dtrain=dtrain, num_boost_round=1000, folds=folds, early_stopping_rounds=25, metrics=["logloss", "auc"], seed=seed))
            auc = cv["test-auc-mean"].arg_max()
            ll = cv["test-logloss-mean"].arg_min()
            cv_rows.append({"run": i + 1, "n_rows": frame.height, "n_games": frame.get_column("game_id").n_unique(), **_export_parameters(params), "AUC_rounds": int(auc + 1), "AUC": float(cv["test-auc-mean"][auc]), "LL_rounds": int(ll + 1), "LL": float(cv["test-logloss-mean"][ll])})
        cv_final = pl.DataFrame(cv_rows).sort("AUC", descending=True)
        if cv_path is not None:
            Path(cv_path).parent.mkdir(parents=True, exist_ok=True)
            cv_final.write_csv(cv_path)
    else:
        if tune_path is None or cv_path is None:
            raise ValueError("tune_path and cv_path are required when hypertune=False")
        best = pl.read_csv(tune_path).row(0, named=True)
        cv_final = pl.read_csv(cv_path)
        params_grid = {**base, **{name: best.get(name, spec["default"]) for name, spec in space.items()}}
        params_grid.update(manual)
        for name, spec in space.items():
            if spec["kind"] == "int":
                params_grid[name] = int(params_grid[name])
    seed = int(cv_final.get_column("seed").item(0))
    rounds = int(cv_final.get_column("AUC_rounds").item(0)) + (0 if "run" in cv_final.columns else 1)
    final_params = {**params_grid, "seed": seed}
    print("Training model...")
    print("Selected params:")
    print(final_params)
    model = xgb.train(final_params, dtrain, num_boost_round=rounds, verbose_eval=2)
    model_path = Path(model_path or MODEL_DIR / "wsba_xg_model.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    model_path.with_name(model_path.stem + "_features.json").write_text(json.dumps(feature_names, indent=2))
    return model


def _model_features(model, model_path):
    sidecar = Path(model_path).with_name(Path(model_path).stem + "_features.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    return model.feature_names or list(FEATURES)


def score_xG_models(shots, model_dir=MODEL_DIR, model_names=None):
    frame, _ = _as_polars(shots)
    output = frame.select(["event_index"]).clone()
    columns = []
    names = list(MODEL_STRENGTHS if model_names is None else model_names)
    for name in names:
        strengths = MODEL_STRENGTHS[name]
        model_data = frame.filter(pl.col("strength_state").is_in(strengths) & (pl.col("empty_net") == int(name == "en")))
        if model_data.is_empty():
            continue
        model_path = Path(model_dir) / f"wsba_xg_model_{name}.json"
        model = xgb.Booster()
        model.load_model(str(model_path))
        model_features = _model_features(model, model_path)
        missing = [feature for feature in model_features if feature not in model_data.columns]
        if missing:
            model_data = model_data.with_columns([pl.lit(None).cast(pl.Float32).alias(feature) for feature in missing])
        matrix = sp.csr_matrix(model_data.select(model_features).fill_null(float("nan")).to_numpy().astype(np.float32, copy=False))
        column = f"xG_{name}"
        columns.append(column)
        predictions = np.clip(model.predict(xgb.DMatrix(matrix, feature_names=model_features)), 0.0, 1.0)
        output = output.join(pl.DataFrame({"event_index": model_data["event_index"], column: predictions}), on="event_index", how="left")
    return output.with_columns(pl.coalesce([pl.col(column) for column in columns]).alias("wsba_xg")) if columns else output.with_columns(pl.lit(None).cast(pl.Float32).alias("wsba_xg"))


def evaluate_xG_models(shots, model_dir=MODEL_DIR, plot_dir=XG_MODEL_DIR, model_names=None):
    """Run the notebook's results tasks on one shared prepared PBP frame."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        auc,
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
        roc_curve,
    )

    frame, _ = _as_polars(shots)
    names = list(MODEL_STRENGTHS if model_names is None else model_names)
    scored = frame.join(score_xG_models(frame, model_dir=model_dir, model_names=names), on="event_index", how="left")
    output_metrics = {}

    def metrics_for(data, prediction_column):
        data = data.filter(pl.col(prediction_column).is_not_null())
        y = data.get_column(TARGET).to_numpy().astype(np.int8, copy=False)
        predictions = data.get_column(prediction_column).to_numpy()
        fpr, tpr, _ = roc_curve(y, predictions)
        return {
            "n_rows": int(data.height),
            "n_goals": int(y.sum()),
            "n_games": int(data.get_column("game_id").n_unique()),
            "roc_auc": float(roc_auc_score(y, predictions)),
            "pr_auc": float(average_precision_score(y, predictions)),
            "log_loss": float(log_loss(y, predictions, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y, predictions)),
            "total_xg": float(predictions.sum()),
            "min_xg": float(predictions.min()),
            "max_xg": float(predictions.max()),
            "roc_auc_curve": float(auc(fpr, tpr)),
        }

    aggregate = scored.filter(pl.col("wsba_xg").is_not_null())
    output_metrics = {"wsba_xg": metrics_for(aggregate, "wsba_xg")}
    output_metrics["wsba_xg"].pop("roc_auc_curve", None)

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    y = aggregate.get_column(TARGET).to_numpy().astype(np.int8, copy=False)
    predictions = aggregate.get_column("wsba_xg").to_numpy()
    fpr, tpr, _ = roc_curve(y, predictions)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("WSBA xG ROC-AUC")
    plt.legend()
    plt.savefig(plot_dir / "wsba_xg_roc_auc.png", bbox_inches="tight")
    plt.close()

    observed, predicted = calibration_curve(y, predictions, n_bins=20)
    plt.figure()
    plt.plot(predicted, observed, "s-")
    plt.plot([0, 1], [0, 1], "--")
    plt.title("WSBA xG Calibration")
    plt.savefig(plot_dir / "wsba_xg_calibration.png", bbox_inches="tight")
    plt.close()

    for name in names:
        model_path = Path(model_dir) / f"wsba_xg_model_{name}.json"
        if not model_path.exists():
            continue
        model = xgb.Booster()
        model.load_model(str(model_path))
        model_features = _model_features(model, model_path)
        importance = [
            {"feature": model_features[int(feature[1:])] if feature.startswith("f") and feature[1:].isdigit() else feature, "gain": float(gain)}
            for feature, gain in model.get_score(importance_type="gain").items()
        ]
        if importance:
            importance = sorted(importance, key=lambda item: item["gain"], reverse=True)[:20]
            plt.figure(figsize=(10, 7))
            plt.barh([item["feature"] for item in importance][::-1], [item["gain"] for item in importance][::-1])
            plt.xlabel("Gain")
            plt.title(f"{name.upper()} Feature Importance")
            plt.savefig(plot_dir / f"wsba_xg_{name}_feature_importance.png", bbox_inches="tight")
            plt.close()

    Path(plot_dir / "wsba_xg_overall_metrics.json").write_text(json.dumps(output_metrics, indent=2))
    return output_metrics


model_results = evaluate_xG_models


def apply_xG(pbp, model_dir=MODEL_DIR):
    frame, was_pandas = _as_polars(pbp)
    if "event_index" not in frame.columns:
        frame = frame.with_row_index("event_index")
    predictions = score_xG_models(prepare_xG_shots(frame), model_dir=model_dir).select(["event_index", "wsba_xg"])
    result = frame.join(predictions, on="event_index", how="left").with_columns(pl.col("wsba_xg").fill_null(0.0).alias("xG"))
    if was_pandas:
        output = pd.DataFrame(result.to_dicts())
        output.index = pd.DataFrame(pbp).index
        return output
    return result

