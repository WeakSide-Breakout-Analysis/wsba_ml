from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import gaussian_filter

import wsba_hockey as wsba
from wsba_hockey.tools.plotting import wsba_rink
from wsba_ml.rapm import MAP_Y, weighted_map, rapm_four_maps
from wsba_ml.wsba_hockey_bridge import nhl_agg_stats_pandas


def _find_workspace_root() -> Path:
    configured = os.environ.get("WSBA_WORKSPACE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "data").exists() and (parent / "wsba_hockey").exists():
            return parent
    return here.parent.parent


def _workspace_or_package_path(workspace_path: Path, package_attr: str) -> Path:
    if workspace_path.exists():
        return workspace_path

    value = getattr(wsba, package_attr, None)
    return Path(value) if value else workspace_path


def _workspace_or_country_info_path(workspace_path: Path) -> Path:
    if workspace_path.exists():
        return workspace_path

    value = getattr(wsba, "INFO_PATH", None)
    return Path(value).with_name("nhl_countryinfo.csv") if value else workspace_path


WORKSPACE_ROOT = _find_workspace_root()
DATA_ROOT = Path(os.environ.get("WSBA_DATA_ROOT") or (WORKSPACE_ROOT / "data")).expanduser().resolve()
WSBA_HOCKEY_PACKAGE_ROOT = WORKSPACE_ROOT / "wsba_hockey" / "src" / "wsba_hockey"

HS_DATA_PATH = DATA_ROOT / "datasets" / "nhl_hs_dataset.csv"
WSBA_LOGO_PATH = _workspace_or_package_path(
    WSBA_HOCKEY_PACKAGE_ROOT / "tools" / "utils" / "wsba.png",
    "IMG_PATH",
)
COUNTRY_INFO_PATH = _workspace_or_country_info_path(
    WSBA_HOCKEY_PACKAGE_ROOT / "tools" / "teaminfo" / "nhl_countryinfo.csv",
)
TEAM_LOGO_RAW_BASE = "https://raw.githubusercontent.com/WeakSide-Breakout-Analysis/wsba_nhl_logo/main"
NHL_ASSET_BASE = "https://assets.nhle.com"
DEFAULT_OUTPUT_PATH = WORKSPACE_ROOT / "player_card.png"

CARD_SIZE = (1240, 746)
MIN_TOI_PG_FOR_PERCENTILES = 7.0
MIN_SPECIAL_TEAMS_TOI_PG = 1.0
SECTION_LINE_ROW_HEIGHT = 24
SECTION_VALUE_WIDTH = 26
SECTION_GAP = 4
SECTION_LABEL_RATIO = 0.44
ACCENT_BAR_HEIGHT = 14
REMOTE_IMAGE_TIMEOUT = 1.0
RAPM_DATA_DIR = DATA_ROOT / "shots_rapm"
COACH_GAR_DATA_PATH = DATA_ROOT / "gar" / "nhl_gar_dataset_coach.csv"
RAPM_PLOTTING_METRICS = {
    "expected_goals": "Expected Goals",
    "goals": "Goals",
    "fenwick": "Fenwick Shots",
}
RAPM_HEATMAP_LIMIT_PERCENTILE = 98
RAPM_HEATMAP_NEUTRAL_RATIO = 0.04

BLUE = (31, 22, 255)
BLUE_MID = (107, 97, 245)
RED = (255, 48, 48)
PINK = (243, 184, 194)
GOLD = (255, 193, 37)
WHITE = (247, 247, 247)
BLACK = (0, 0, 0)
GRID = (19, 19, 24)
DIVIDER = (235, 235, 235)
GRAY = (72, 72, 80)
PERCENTILE_RED = (255, 0, 0)
PERCENTILE_WHITE = (255, 255, 255)
PERCENTILE_BLUE = (0, 0, 255)
PERCENTILE_MIDPOINT = 50.0
PERCENTILE_MAX = 100.0
_REMOTE_IMAGE_BLOCKED_HOSTS: set[str] = set()

METRIC_MAP = {
    "Offense": "offense",
    "Defense": "defense",
    "Penalties": "penalties",
    "EV Generation": "even_strength_generation",
    "EV Suppression": "even_strength_suppression",
    "PP Generation": "power_play_generation",
    "PK Suppression": "penalty_kill_suppression",
    "Finishing": "finishing",
    "Induction": "induction",
    "Taken Penalties": "taken_penalties",
    "Drawn Penalties": "drawn_penalties",
    "Time On Ice": "time_on_ice_per_games_played",
    "Competition": "quality_of_competition",
    "Teammates": "quality_of_teammates",
    "O-Zone Faceoffs": "offensive_zone_faceoffs_percentage",
    "N-Zone Faceoffs": "neutral_zone_faceoffs_percentage",
    "D-Zone Faceoffs": "defensive_zone_faceoffs_percentage",
    "Shot Rate": "fenwick_per_sixty",
    "Shot Quality": "expected_goals_per_fenwick",
    "Pass Rate": "fenwick_assists_per_sixty",
    "Pass Quality": "expected_assists_per_fenwick_assists",
    "Hits Given": "hits_for_per_sixty",
    "Hits Taken": "hits_against_per_sixty",
    "Blocked Shots": "blocked_shots_per_sixty",
    "Giveaways": "giveaways",
    "Takeaways": "takeaways",
    "Consistency": "game_score_consistency",
    "Carries": "carries_per_sixty",
    "Dump-Ins": "dump_ins_per_sixty",
    "Retrievals": "retrievals_per_sixty",
    "Forechecking": "forechecking_pressures_per_sixty",
    "Pass Entries": "pass_entries_per_sixty",
    "Carry Entries": "carry_entries_per_sixty",
    "Pass Exits": "pass_exits_per_sixty",
    "Carry Exits": "carry_exits_per_sixty",
    "Entry Targets": "entries_against_per_sixty",
    "Carries Against": "carry_entries_against_per_sixty",
    "Disrupted Entries": "disrupted_entries_per_sixty",
    "GAR %": "goals_above_replacement",
}

MICROIMPACT_LABELS = [
    "Carries",
    "Dump-Ins",
    "Retrievals",
    "Forechecking",
    "Pass Entries",
    "Carry Entries",
    "Pass Exits",
    "Carry Exits",
    "Entry Targets",
    "Carries Against",
    "Disrupted Entries",
]

COACH_IMPACT_LABELS = [
    "EV Generation",
    "EV Suppression",
    "Taken Penalties",
    "Drawn Penalties",
    "Goals Above Replacement",
]

COACH_SCORE_STATE_ROWS = [
    ("Behind 3+ EV", "behind_3_plus"),
    ("Behind 2 EV", "behind_2"),
    ("Behind 1 EV", "behind_1"),
    ("Tied EV", "tied"),
    ("Lead 1 EV", "lead_1"),
    ("Lead 2 EV", "lead_2"),
    ("Lead 3+ EV", "lead_3_plus"),
]

BASE_AGG_COLUMNS = [
    "player_name",
    "player_id",
    "season",
    "team_abbr",
    "position",
    "handedness",
    "height_in",
    "weight_lbs",
    "age",
    "birth_country",
    "headshot",
    "games_played",
    "time_on_ice",
    "even_strength_generation",
    "even_strength_suppression",
    "power_play_generation",
    "penalty_kill_suppression",
    "generation",
    "suppression",
    "advancement",
    "prevention",
    "finishing",
    "induction",
    "taken_penalties",
    "drawn_penalties",
    "even_strength",
    "offense",
    "defense",
    "transition",
    "production",
    "penalties",
    "expected_goals_above_replacement",
    "goals_above_replacement",
]


@dataclass(frozen=True)
class MetricLine:
    label: str
    value: float | int | None


@dataclass(frozen=True)
class PixelRect:
    x: int
    y: int
    width: int
    height: int

    @property
    def left(self) -> int:
        return self.x

    @property
    def top(self) -> int:
        return self.y

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> int:
        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        return self.y + self.height // 2

    def inset(self, dx: int, dy: int | None = None) -> "PixelRect":
        dy = dx if dy is None else dy
        return PixelRect(self.x + dx, self.y + dy, self.width - 2 * dx, self.height - 2 * dy)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return np.nan
    return float(numerator) / float(denominator)


def _seasons_key(seasons: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted({int(season) for season in seasons if pd.notna(season)}))


def _normalize_rapm_plotting_metric(metric: str) -> str:
    normalized = str(metric or "").strip().lower().replace(" ", "_")
    if normalized not in RAPM_PLOTTING_METRICS:
        allowed = ", ".join(RAPM_PLOTTING_METRICS)
        raise ValueError(f"plotting_metric must be one of: {allowed}.")
    return normalized


def _hex_to_rgb(value: str | None, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(value, str):
        return fallback
    cleaned = value.strip().lstrip("#")
    if len(cleaned) != 6:
        return fallback
    try:
        return tuple(int(cleaned[idx:idx + 2], 16) for idx in (0, 2, 4))
    except ValueError:
        return fallback


@lru_cache(maxsize=1)
def _teaminfo() -> pd.DataFrame:
    return wsba.repo_load_teaminfo()


@lru_cache(maxsize=1)
def _countryinfo() -> pd.DataFrame:
    return pd.read_csv(COUNTRY_INFO_PATH)


@lru_cache(maxsize=64)
def _team_colors(team_abbr: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    try:
        teaminfo = _teaminfo()
        team_row = (
            teaminfo.loc[teaminfo["team_abbr"].astype(str).str.upper() == str(team_abbr).upper()]
            .drop_duplicates(subset=["team_abbr"], keep="last")
        )
        if not team_row.empty:
            row = team_row.iloc[-1]
            primary = _hex_to_rgb(row.get("primary_color"), GOLD)
            secondary = _hex_to_rgb(row.get("secondary_color"), WHITE)
            return primary, secondary
    except Exception:
        pass
    return GOLD, WHITE


@lru_cache(maxsize=128)
def _country_flag_url(country_code: str) -> str | None:
    code = str(country_code or "").strip().upper()
    if not code:
        return None

    countryinfo = _countryinfo()
    match = countryinfo.loc[
        countryinfo["country_abbr"].fillna("").str.upper().eq(code)
        | countryinfo["country_abbr_2"].fillna("").str.upper().eq(code)
        | countryinfo["team_id"].fillna("").astype(str).str.upper().eq(code)
    ]
    if match.empty:
        return None

    row = match.iloc[0]
    path = row.get("country_flag_large") or row.get("thumbnail_url")
    if not isinstance(path, str) or not path.strip():
        return None
    return f"{NHL_ASSET_BASE}{path}" if path.startswith("/") else path


def _weighted_average(values: pd.Series, weights: pd.Series | np.ndarray) -> float:
    value_series = pd.to_numeric(values, errors="coerce")
    weight_series = pd.to_numeric(pd.Series(weights, index=values.index), errors="coerce")
    mask = value_series.notna() & weight_series.notna() & (weight_series > 0)
    if not mask.any():
        return np.nan
    return float(np.average(value_series[mask], weights=weight_series[mask]))


def _safe_series_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float).div(denominator.astype(float).replace(0, np.nan))
    return result


def _group_weighted_average(
    frame: pd.DataFrame,
    value_column: str,
    weights: pd.Series,
    player_ids: pd.Series,
) -> pd.Series:
    if value_column not in frame.columns:
        return pd.Series(dtype=float)

    values = pd.to_numeric(frame[value_column], errors="coerce")
    weight_values = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weight_values.notna() & (weight_values > 0)
    if not mask.any():
        return pd.Series(dtype=float)

    numerators = (values[mask] * weight_values[mask]).groupby(player_ids[mask]).sum()
    denominators = weight_values[mask].groupby(player_ids[mask]).sum()
    return _safe_series_divide(numerators, denominators)


def _normalize_position(value: str) -> str:
    token = str(value).upper()
    return "D" if token == "D" else "F"


def _season_label(seasons: list[int]) -> str:
    ordered = sorted({int(season) for season in seasons if pd.notna(season)})
    if not ordered:
        return ""
    if len(ordered) == 1:
        return str(ordered[0])
    return f"{str(ordered[0])[:4]}-{str(ordered[-1])[4:]}"


def _default_card_layout() -> dict[str, PixelRect]:
    return {
        "title": PixelRect(24, 18, 480, 58),
        "header_bar": PixelRect(20, 88, 484, ACCENT_BAR_HEIGHT),
        "headshot_panel": PixelRect(24, 122, 198, 154),
        "team_title": PixelRect(254, 108, 86, 32),
        "team_panel": PixelRect(248, 136, 104, 104),
        "bio_title": PixelRect(356, 108, 149, 32),
        "bio_panel": PixelRect(375, 141, 121, 131),
        "season_label": PixelRect(254, 233, 90, 34),
        "info_divider": PixelRect(20, 274, 484, ACCENT_BAR_HEIGHT),
        "gar_title": PixelRect(24, 298, 226, 34),
        "gar_panel": PixelRect(24, 334, 226, 294),
        "impacts_title": PixelRect(277, 298, 219, 34),
        "impacts_panel": PixelRect(277, 334, 219, 294),
        "locational_title_ev": PixelRect(526, 15, 444, 34),
        "locational_panel_ev": PixelRect(526, 52, 444, 236),
        "locational_cbar": PixelRect(526, 296, 444, 32),
        "locational_title_st": PixelRect(526, 355, 444, 34),
        "locational_panel_st": PixelRect(526, 392, 444, 236),
        "deployment_title": PixelRect(991, 15, 219, 34),
        "deployment_panel": PixelRect(991, 52, 219, 190),
        "tendencies_title": PixelRect(991, 232, 219, 34),
        "tendencies_panel": PixelRect(991, 268, 219, 360),
        "footer_blue": PixelRect(20, 638, 1190, ACCENT_BAR_HEIGHT),
        "footer_red": PixelRect(20, 672, 1190, ACCENT_BAR_HEIGHT),
        "footer_text": PixelRect(20, 689, 1085, 42),
    }


def _resolve_card_layout(card_layout: dict[str, PixelRect] | None) -> dict[str, PixelRect]:
    base = _default_card_layout()
    if not card_layout:
        return base
    merged = dict(base)
    merged.update(card_layout)
    return merged


def _default_coach_card_layout() -> dict[str, PixelRect]:
    return {
        "title": PixelRect(20, 18, 554, 58),
        "header_bar": PixelRect(20, 84, 1200, ACCENT_BAR_HEIGHT),
        "identity_title": PixelRect(20, 108, 554, 30),
        "identity_panel": PixelRect(20, 108, 554, 112),
        "identity_logo": PixelRect(30, 122, 150, 88),
        "season_label": PixelRect(30, 206, 150, 20),
        "info_divider": PixelRect(20, 236, 1200, ACCENT_BAR_HEIGHT),
        "systems_title": PixelRect(210, 100, 286, 30),
        "systems_panel": PixelRect(210, 132, 286, 100),
        "offense_score_title": PixelRect(24, 252, 269, 52),
        "offense_score_panel": PixelRect(24, 314, 269, 314),
        "defense_score_title": PixelRect(305, 252, 269, 52),
        "defense_score_panel": PixelRect(305, 314, 269, 314),
        "locational_title_systems": PixelRect(574, 15, 646, 34),
        "locational_panel_systems": PixelRect(574, 52, 646, 250),
        "locational_cbar": PixelRect(574, 310, 646, 32),
        "locational_title_overall": PixelRect(574, 346, 646, 34),
        "locational_panel_overall": PixelRect(574, 384, 646, 244),
        "footer_blue": PixelRect(20, 638, 1190, ACCENT_BAR_HEIGHT),
        "footer_red": PixelRect(20, 672, 1190, ACCENT_BAR_HEIGHT),
        "footer_text": PixelRect(20, 689, 1085, 42),
    }


def _resolve_coach_card_layout(card_layout: dict[str, PixelRect] | None) -> dict[str, PixelRect]:
    base = _default_coach_card_layout()
    if not card_layout:
        return base
    merged = dict(base)
    merged.update(card_layout)
    return merged


def _render_figure_to_image(fig, *, tight: bool = False) -> Image.Image:
    buffer = BytesIO()
    bbox = "tight" if tight else None
    fig.savefig(buffer, format="png", dpi=200, facecolor="black", bbox_inches=bbox, pad_inches=0.0)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGBA")


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_from_png_bytes(payload: bytes) -> Image.Image:
    return Image.open(BytesIO(payload)).convert("RGBA")


@lru_cache(maxsize=8)
def _rapm_shifts(seasons_key: tuple[int, ...]) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(RAPM_DATA_DIR / f"nhl_shots_rapm_{season}.parquet") for season in seasons_key],
        ignore_index=True,
    )


@lru_cache(maxsize=256)
def _rapm_plot_data_cached(
    player_id: int,
    seasons_key: tuple[int, ...],
    player_pp_toi: float,
    player_pk_toi: float,
    plotting_metric: str,
):
    shifts = _rapm_shifts(seasons_key)

    ev_for, ev_against, st_for, st_against = rapm_four_maps(
        shifts,
        seasons_key,
        int(player_id),
        group="skater",
        metric=plotting_metric,
    )

    if player_pp_toi < MIN_SPECIAL_TEAMS_TOI_PG:
        st_for = np.zeros_like(st_for)
    if player_pk_toi < MIN_SPECIAL_TEAMS_TOI_PG:
        st_against = np.zeros_like(st_against)

    ev_for = gaussian_filter(ev_for, sigma=5)
    ev_against = gaussian_filter(ev_against, sigma=5)
    st_for = gaussian_filter(st_for, sigma=5)
    st_against = gaussian_filter(st_against, sigma=5)

    all_grids = np.stack([ev_for, ev_against, st_for, st_against])
    nonzero_values = np.abs(all_grids[np.abs(all_grids) > 0])
    if nonzero_values.size:
        limit = float(np.percentile(nonzero_values, RAPM_HEATMAP_LIMIT_PERCENTILE))
        limit = max(limit, float(nonzero_values.max()) * 0.35)
    else:
        limit = 1.0
    return ev_against, ev_for, st_against, st_for, limit


def _rapm_plot_data(player_id: int, seasons: list[int], player: pd.Series):
    player_pp_toi = player.get("power_play_time_on_ice_per_games_played", 0)
    player_pk_toi = player.get("penalty_kill_time_on_ice_per_games_played", 0)
    player_pp_toi = 0.0 if player_pp_toi is None or pd.isna(player_pp_toi) else float(player_pp_toi)
    player_pk_toi = 0.0 if player_pk_toi is None or pd.isna(player_pk_toi) else float(player_pk_toi)
    return _rapm_plot_data_cached(
        int(player_id),
        _seasons_key(seasons),
        round(player_pp_toi, 6),
        round(player_pk_toi, 6),
        "fenwick",
    )


def _rapm_rink_image(
    left_grid: np.ndarray,
    right_grid: np.ndarray,
    left_label: str,
    right_label: str,
    limit: float,
    size: tuple[int, int],
) -> Image.Image:
    x = np.linspace(-100, 100, np.vstack([left_grid, right_grid]).shape[0])
    y = np.linspace(MAP_Y.min(), MAP_Y.max(), np.vstack([left_grid, right_grid]).shape[1])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8.6, 3.25), facecolor="black")
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set_facecolor("black")

    cmap = _red_white_blue_cmap()
    plt.sca(ax)
    wsba_rink("full", ax=ax, figsize=(8.6, 3.25))
    combined = np.vstack([left_grid, right_grid])
    contour_levels = np.linspace(-limit, limit, 17)
    neutral_threshold = limit * RAPM_HEATMAP_NEUTRAL_RATIO
    contour_values = np.ma.masked_where(np.abs(combined.T) <= neutral_threshold, combined.T)
    ax.contourf(
        X,
        Y,
        contour_values,
        levels=contour_levels,
        cmap=cmap,
        alpha=0.78,
        extend="both",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    image = _render_figure_to_image(fig, tight=True)
    label_strip = 18
    rink_size = (size[0], max(1, size[1] - label_strip))
    contained = ImageOps.contain(image, rink_size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, (0, 0, 0, 255))
    offset_x = (size[0] - contained.width) // 2
    offset_y = 0
    mask = Image.new("L", contained.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    radius = max(24, min(contained.height // 2, contained.width // 7))
    mask_draw.rounded_rectangle((0, 0, contained.width - 1, contained.height - 1), radius=radius, fill=255)
    canvas.paste(contained, (offset_x, offset_y), mask)
    draw = ImageDraw.Draw(canvas)
    label_font = _font(14)
    label_y = size[1] - 10
    draw.text((size[0] * 0.25, label_y), left_label, font=label_font, fill=WHITE, anchor="mm")
    draw.text((size[0] * 0.75, label_y), right_label, font=label_font, fill=WHITE, anchor="mm")
    return canvas


def _rapm_cbar_image(limit: float, size: tuple[int, int], plotting_metric: str) -> Image.Image:
    canvas = Image.new("RGBA", size, (0, 0, 0, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(12)
    label_font = _font(12)
    title_y = 7
    gradient_y0 = 14
    gradient_height = 6
    gradient = np.array(
        [_percentile_color(value) for value in np.linspace(0.0, PERCENTILE_MAX, size[0])],
        dtype=np.uint8,
    )
    rgba = np.concatenate([gradient, np.full((size[0], 1), 255, dtype=np.uint8)], axis=1)
    gradient_image = Image.fromarray(np.tile(rgba[np.newaxis, :, :], (gradient_height, 1, 1)), mode="RGBA")
    canvas.paste(gradient_image, (0, gradient_y0), gradient_image)
    metric_label = RAPM_PLOTTING_METRICS[plotting_metric]
    draw.text((size[0] / 2, title_y), f"Impact on {metric_label} per 60", font=title_font, fill=WHITE, anchor="mm")
    label_y = size[1] - 6
    draw.text((0, label_y), "deficit", font=label_font, fill=WHITE, anchor="lm")
    draw.text((size[0] / 2, label_y), "even", font=label_font, fill=WHITE, anchor="mm")
    draw.text((size[0], label_y), "surplus", font=label_font, fill=WHITE, anchor="rm")
    return canvas


@lru_cache(maxsize=256)
def _rapm_panel_images_cached(
    player_id: int,
    seasons_key: tuple[int, ...],
    player_pp_toi: float,
    player_pk_toi: float,
    plotting_metric: str,
    ev_size: tuple[int, int],
    st_size: tuple[int, int],
    cbar_size: tuple[int, int],
) -> tuple[bytes, bytes, bytes]:
    ev_against, ev_for, st_against, st_for, rapm_limit = _rapm_plot_data_cached(
        player_id,
        seasons_key,
        player_pp_toi,
        player_pk_toi,
        plotting_metric,
    )
    ev_image = _rapm_rink_image(
        ev_against,
        ev_for,
        "EV Suppression",
        "EV Generation",
        rapm_limit,
        ev_size,
    )
    cbar_image = _rapm_cbar_image(rapm_limit, cbar_size, plotting_metric)
    st_image = _rapm_rink_image(
        st_against,
        st_for,
        "PK Suppression",
        "PP Generation",
        rapm_limit,
        st_size,
    )
    return (
        _image_to_png_bytes(ev_image),
        _image_to_png_bytes(cbar_image),
        _image_to_png_bytes(st_image),
    )



@lru_cache(maxsize=8)
def _load_hs_dataset_cached(seasons_key: tuple[int, ...]) -> pd.DataFrame:
    df = pd.read_csv(HS_DATA_PATH)
    df = df.loc[df['season'].isin(seasons_key)]

    if "season_type" in df.columns:
        df = df.loc[df["season_type"].fillna(2).astype(int) == 2].copy()
    else:
        df = df.copy()

    df["_source_order"] = np.arange(len(df))
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df = df.loc[df["player_id"].notna()].copy()
    df["position_group"] = df["position"].map(_normalize_position)
    df = df.sort_values(["player_id", "season", "_source_order"]).reset_index(drop=True)
    return df


def _load_hs_dataset(seasons: list[int]) -> pd.DataFrame:
    return _load_hs_dataset_cached(_seasons_key(seasons))


def _aggregate_weighted_features(df: pd.DataFrame) -> pd.DataFrame:
    player_ids = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    toi = pd.to_numeric(df["time_on_ice"], errors="coerce").fillna(0.0)
    gp = pd.to_numeric(df["games_played"], errors="coerce").fillna(0.0)
    a3z_toi = pd.to_numeric(df["a3z_time_on_ice"], errors="coerce").fillna(0.0)
    a3z_toi = a3z_toi.where(a3z_toi > 0, toi)
    consistency_weights = gp.where(gp > 0, toi)

    totals = pd.DataFrame(
        {
            "total_toi": toi.groupby(player_ids).sum(),
            "total_gp": gp.groupby(player_ids).sum(),
        }
    )

    rate_columns = [
        "fenwick_per_sixty",
        "fenwick_assists_per_sixty",
        "hits_for_per_sixty",
        "hits_against_per_sixty",
        "blocked_shots_per_sixty",
        "carries_per_sixty",
        "dump_ins_per_sixty",
        "retrievals_per_sixty",
        "forechecking_pressures_per_sixty",
        "pass_entries_per_sixty",
        "carry_entries_per_sixty",
        "pass_exits_per_sixty",
        "carry_exits_per_sixty",
        "entries_against_per_sixty",
        "carry_entries_against_per_sixty",
        "disrupted_entries_per_sixty",
    ]

    rate_frame = pd.DataFrame(index=df.index)
    for column in rate_columns + ["giveaways_per_sixty", "takeaways_per_sixty"]:
        if column in df.columns:
            rate_frame[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        else:
            rate_frame[column] = 0.0

    weighted_rate_sums = rate_frame.mul(toi, axis=0).groupby(player_ids).sum()
    rate_results = weighted_rate_sums[rate_columns].div(totals["total_toi"].replace(0, np.nan), axis=0)
    count_results = weighted_rate_sums[["giveaways_per_sixty", "takeaways_per_sixty"]].div(60.0)
    count_results = count_results.rename(
        columns={
            "giveaways_per_sixty": "giveaways",
            "takeaways_per_sixty": "takeaways",
        }
    )
    fenwick_weights = rate_frame["fenwick_per_sixty"] * toi / 60.0
    fenwick_assist_weights = rate_frame["fenwick_assists_per_sixty"] * toi / 60.0

    ev_toi = pd.to_numeric(
        df.get("even_strength_time_on_ice_per_games_played", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).fillna(0.0) * gp
    pp_toi = pd.to_numeric(
        df.get("power_play_time_on_ice_per_games_played", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).fillna(0.0) * gp
    pk_toi = pd.to_numeric(
        df.get("penalty_kill_time_on_ice_per_games_played", pd.Series(0.0, index=df.index)),
        errors="coerce",
    ).fillna(0.0) * gp

    result = pd.DataFrame(index=totals.index)
    result["time_on_ice_per_games_played"] = _safe_series_divide(totals["total_toi"], totals["total_gp"])
    result["even_strength_time_on_ice_per_games_played"] = _safe_series_divide(
        ev_toi.groupby(player_ids).sum(), totals["total_gp"]
    )
    result["power_play_time_on_ice_per_games_played"] = _safe_series_divide(
        pp_toi.groupby(player_ids).sum(), totals["total_gp"]
    )
    result["penalty_kill_time_on_ice_per_games_played"] = _safe_series_divide(
        pk_toi.groupby(player_ids).sum(), totals["total_gp"]
    )
    result["quality_of_competition"] = _group_weighted_average(df, "quality_of_competition", toi, player_ids)
    result["quality_of_teammates"] = _group_weighted_average(df, "quality_of_teammates", toi, player_ids)
    result["offensive_zone_faceoffs_percentage"] = _group_weighted_average(
        df, "offensive_zone_faceoffs_percentage", a3z_toi, player_ids
    )
    result["neutral_zone_faceoffs_percentage"] = _group_weighted_average(
        df, "neutral_zone_faceoffs_percentage", a3z_toi, player_ids
    )
    result["defensive_zone_faceoffs_percentage"] = _group_weighted_average(
        df, "defensive_zone_faceoffs_percentage", a3z_toi, player_ids
    )
    result["expected_goals_per_fenwick"] = _group_weighted_average(
        df, "expected_goals_per_fenwick", fenwick_weights, player_ids
    )
    result["expected_assists_per_fenwick_assists"] = _group_weighted_average(
        df,
        "expected_assists_per_fenwick_assists",
        fenwick_assist_weights,
        player_ids,
    )
    result["game_score_consistency"] = _group_weighted_average(
        df, "game_score_consistency", consistency_weights, player_ids
    )

    result = result.join(rate_results)
    result = result.join(count_results)
    result = result.reset_index().rename(columns={"player_id": "player_id"})
    result["player_id"] = result["player_id"].astype(int)
    return result


@lru_cache(maxsize=8)
def _card_dataset_cached(seasons_key: tuple[int, ...]) -> pd.DataFrame:
    df = _load_hs_dataset_cached(seasons_key)
    base = nhl_agg_stats_pandas(df[BASE_AGG_COLUMNS], ["player_id"], comparison=False)
    weighted = _aggregate_weighted_features(df)

    summary = (
        base.merge(weighted, on="player_id", how="left", suffixes=("", "_weighted"))
        .sort_values("player_id")
        .reset_index(drop=True)
    )

    for column in weighted.columns:
        if column == "player_id":
            continue
        weighted_column = f"{column}_weighted"
        if weighted_column in summary.columns:
            if column in summary.columns:
                summary[column] = summary[weighted_column].combine_first(summary[column])
            else:
                summary[column] = summary[weighted_column]
            summary = summary.drop(columns=[weighted_column])

    sorted_df = df.sort_values(["player_id", "season", "_source_order"]).reset_index(drop=True)
    last_rows = (
        sorted_df.drop_duplicates(subset=["player_id"], keep="last")
        .loc[
            :,
            [
                "player_id",
                "team_abbr",
                "headshot",
                "age",
                "birth_country",
                "handedness",
                "player_name",
                "position_group",
                "weight_lbs",
                "height_in",
            ],
        ]
        .rename(
            columns={
                "team_abbr": "last_team_abbr",
                "headshot": "last_headshot",
                "age": "last_age",
                "birth_country": "last_birth_country",
                "handedness": "last_handedness",
                "player_name": "player_name_last",
                "position_group": "position_group_last",
                "weight_lbs": "weight_lbs_last",
                "height_in": "height_in_last",
            }
        )
    )
    season_bounds = (
        sorted_df.groupby("player_id")["season"]
        .agg(season_min="min", season_max="max")
        .reset_index()
    )
    season_bounds["season_label"] = np.where(
        season_bounds["season_min"] == season_bounds["season_max"],
        season_bounds["season_min"].astype(str),
        season_bounds["season_min"].astype(str).str[:4] + "-" + season_bounds["season_max"].astype(str).str[4:],
    )

    summary = summary.merge(last_rows, on="player_id", how="left")
    summary = summary.merge(season_bounds[["player_id", "season_label"]], on="player_id", how="left")

    summary["player_name"] = summary["player_name_last"].where(summary["player_name_last"].ne(""), summary["player_name"])
    summary["team_abbr"] = summary["last_team_abbr"].fillna(summary["team_abbr"])
    summary["headshot"] = summary["last_headshot"].fillna(summary["headshot"])
    summary["age"] = summary["last_age"].fillna(summary["age"])
    summary["birth_country"] = summary["last_birth_country"].fillna(summary["birth_country"])
    summary["handedness"] = summary["last_handedness"].fillna(summary["handedness"])
    summary["height_in"] = summary["height_in_last"].fillna(summary["height_in"])
    summary["weight_lbs"] = summary["weight_lbs_last"].fillna(summary["weight_lbs"])
    existing_position_group = (
        summary["position_group"] if "position_group" in summary.columns else pd.Series(np.nan, index=summary.index)
    )
    summary["position_group"] = summary["position_group_last"].fillna(existing_position_group)
    summary["position_group"] = summary["position_group"].fillna(summary["position"].map(_normalize_position))

    summary["eligible_for_percentiles"] = summary["time_on_ice_per_games_played"] >= MIN_TOI_PG_FOR_PERCENTILES
    summary.loc[
        summary["power_play_time_on_ice_per_games_played"].fillna(0) < MIN_SPECIAL_TEAMS_TOI_PG,
        "power_play_generation",
    ] = np.nan
    summary.loc[
        summary["penalty_kill_time_on_ice_per_games_played"].fillna(0) < MIN_SPECIAL_TEAMS_TOI_PG,
        "penalty_kill_suppression",
    ] = np.nan

    for column in dict.fromkeys(METRIC_MAP.values()):
        if column not in summary.columns:
            summary[column] = np.nan

        percentile_column = f"{column}_percentile"
        summary[percentile_column] = np.nan

        for position_group, position_frame in summary.groupby("position_group"):
            eligible_mask = position_frame["eligible_for_percentiles"] & position_frame[column].notna()
            eligible_index = position_frame.loc[eligible_mask].index
            if eligible_index.empty:
                continue

            values = position_frame.loc[eligible_index, column]
            if column == "giveaways":
                values = -values

            ranked = (
                values
                .rank(pct=True, method="max")
                .mul(100)
                .round()
                .clip(0, 100)
            )
            summary.loc[eligible_index, percentile_column] = ranked

    keep = [
        "player_id",
        "player_name",
        "season_label",
        "team_abbr",
        "position_group",
        "handedness",
        "height_in",
        "weight_lbs",
        "age",
        "birth_country",
        "headshot",
        "eligible_for_percentiles",
        "even_strength_time_on_ice_per_games_played",
        "power_play_time_on_ice_per_games_played",
        "penalty_kill_time_on_ice_per_games_played",
    ]
    keep += [column for column in dict.fromkeys(METRIC_MAP.values())]
    keep += [f"{column}_percentile" for column in dict.fromkeys(METRIC_MAP.values())]

    return summary[keep].copy()


def _card_dataset(seasons: list[int]) -> pd.DataFrame:
    return _card_dataset_cached(_seasons_key(seasons))


@lru_cache(maxsize=8)
def _load_coach_gar_dataset_cached(seasons_key: tuple[int, ...]) -> pd.DataFrame:
    if not COACH_GAR_DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(COACH_GAR_DATA_PATH)
    if "season" in df.columns:
        df = df.loc[pd.to_numeric(df["season"], errors="coerce").isin(seasons_key)].copy()
    if "coach_name" in df.columns:
        df["coach_name"] = df["coach_name"].astype(str).str.strip()
        df = df.loc[df["coach_name"].ne("") & df["coach_name"].str.lower().ne("nan")]
    return df.reset_index(drop=True)


@lru_cache(maxsize=8)
def _coach_card_dataset_cached(seasons_key: tuple[int, ...]) -> pd.DataFrame:
    df = _load_coach_gar_dataset_cached(seasons_key)
    if df.empty:
        return pd.DataFrame()

    metric_columns = [
        "time_on_ice",
        "even_strength_generation",
        "even_strength_suppression",
        "taken_penalties",
        "drawn_penalties",
        "goals_above_replacement",
        *[f"{state}_even_strength_generation" for _label, state in COACH_SCORE_STATE_ROWS],
        *[f"{state}_even_strength_suppression" for _label, state in COACH_SCORE_STATE_ROWS],
    ]
    metric_columns = [column for column in metric_columns if column in df.columns]
    numeric = df[["coach_name", *metric_columns]].copy()
    for column in metric_columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    summary = numeric.groupby("coach_name", dropna=False, as_index=False)[metric_columns].sum(min_count=1)

    ordered = df.sort_values(["coach_name", "season", "team_abbr"], ascending=[True, True, True]).reset_index(drop=True)
    last_rows = ordered.drop_duplicates(subset=["coach_name"], keep="last")[["coach_name", "team_abbr"]].rename(
        columns={"team_abbr": "last_team_abbr"}
    )
    season_bounds = (
        ordered.groupby("coach_name")["season"]
        .agg(season_min="min", season_max="max")
        .reset_index()
    )
    season_bounds["season_label"] = np.where(
        season_bounds["season_min"] == season_bounds["season_max"],
        season_bounds["season_min"].astype(str),
        season_bounds["season_min"].astype(str).str[:4] + "-" + season_bounds["season_max"].astype(str).str[4:],
    )

    summary = summary.merge(last_rows, on="coach_name", how="left")
    summary = summary.merge(season_bounds[["coach_name", "season_label"]], on="coach_name", how="left")
    summary["team_abbr"] = summary["last_team_abbr"].fillna("")
    summary["coach_id"] = summary["coach_name"]

    percentile_columns = [
        "even_strength_generation",
        "even_strength_suppression",
        "taken_penalties",
        "drawn_penalties",
        "goals_above_replacement",
        *[f"{state}_even_strength_generation" for _label, state in COACH_SCORE_STATE_ROWS],
        *[f"{state}_even_strength_suppression" for _label, state in COACH_SCORE_STATE_ROWS],
    ]
    for column in percentile_columns:
        if column not in summary.columns:
            summary[column] = np.nan
        percentile_column = f"{column}_percentile"
        summary[percentile_column] = np.nan
        eligible = summary[column].notna()
        if not eligible.any():
            continue
        summary.loc[eligible, percentile_column] = (
            summary.loc[eligible, column]
            .rank(pct=True, method="max")
            .mul(100)
            .round()
            .clip(0, 100)
        )

    keep = ["coach_name", "coach_id", "season_label", "team_abbr", "time_on_ice"]
    keep += percentile_columns
    keep += [f"{column}_percentile" for column in percentile_columns]
    keep = [column for column in keep if column in summary.columns]
    return summary[keep].copy()


def _coach_card_dataset(seasons: list[int]) -> pd.DataFrame:
    return _coach_card_dataset_cached(_seasons_key(seasons))


@lru_cache(maxsize=8)
def _coach_rapm_dataset_cached(seasons_key: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for season in seasons_key:
        path = RAPM_DATA_DIR / f"nhl_shots_rapm_coach_{season}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "season_type" in frame.columns:
            frame = frame.loc[pd.to_numeric(frame["season_type"], errors="coerce").fillna(2).eq(2)].copy()
        if "strength_state" in frame.columns:
            frame = frame.loc[frame["strength_state"].astype(str).eq("even_strength")].copy()
        if "coach_name" in frame.columns:
            frame["coach_name"] = frame["coach_name"].astype(str).str.strip()
            frame = frame.loc[frame["coach_name"].ne("") & frame["coach_name"].str.lower().ne("nan")]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@lru_cache(maxsize=256)
def _coach_rapm_panel_images_cached(
    coach_name: str,
    seasons_key: tuple[int, ...],
    plotting_metric: str,
    systems_size: tuple[int, int],
    overall_size: tuple[int, int],
    cbar_size: tuple[int, int],
) -> tuple[bytes, bytes, bytes]:
    rapm = _coach_rapm_dataset_cached(seasons_key)
    coach_rows = rapm.loc[rapm["coach_name"].astype(str).str.upper().eq(str(coach_name).upper())].copy()
    if coach_rows.empty:
        raise ValueError(f"Coach RAPM data unavailable for {coach_name}.")

    score_state_for_columns = [f"rapm_{state}_{plotting_metric}_for_per_sixty_map" for _label, state in COACH_SCORE_STATE_ROWS]
    score_state_against_columns = [f"rapm_{state}_{plotting_metric}_against_per_sixty_map" for _label, state in COACH_SCORE_STATE_ROWS]

    systems_for = gaussian_filter(
        weighted_map(coach_rows, f"rapm_systems_{plotting_metric}_for_per_sixty_map"),
        sigma=5,
    )
    systems_against = gaussian_filter(
        weighted_map(coach_rows, f"rapm_systems_{plotting_metric}_against_per_sixty_map"),
        sigma=5,
    )
    overall_for = gaussian_filter(
        sum(
            (weighted_map(coach_rows, column) for column in [f"rapm_systems_{plotting_metric}_for_per_sixty_map", *score_state_for_columns]),
            np.zeros_like(systems_for),
        ),
        sigma=5,
    )
    overall_against = gaussian_filter(
        sum(
            (weighted_map(coach_rows, column) for column in [f"rapm_systems_{plotting_metric}_against_per_sixty_map", *score_state_against_columns]),
            np.zeros_like(systems_against),
        ),
        sigma=5,
    )

    all_grids = np.stack([systems_for, systems_against, overall_for, overall_against])
    nonzero_values = np.abs(all_grids[np.abs(all_grids) > 0])
    if nonzero_values.size:
        limit = float(np.percentile(nonzero_values, RAPM_HEATMAP_LIMIT_PERCENTILE))
        limit = max(limit, float(nonzero_values.max()) * 0.35)
    else:
        limit = 1.0

    systems_image = _rapm_rink_image(
        systems_against,
        systems_for,
        "EV Suppression",
        "EV Generation",
        limit,
        systems_size,
    )
    overall_image = _rapm_rink_image(
        overall_against,
        overall_for,
        "EV Suppression",
        "EV Generation",
        limit,
        overall_size,
    )
    cbar_image = _rapm_cbar_image(limit, cbar_size, plotting_metric)
    return (
        _image_to_png_bytes(systems_image),
        _image_to_png_bytes(cbar_image),
        _image_to_png_bytes(overall_image),
    )


@lru_cache(maxsize=1)
def _font_path() -> Path | None:
    candidates = [
        Path.home() / "Downloads" / "Bebas_Neue" / "BebasNeue-Regular.ttf",
        Path.home() / "Downloads" / "BebasNeue-Regular.ttf",
        WORKSPACE_ROOT,
        Path.home() / "Downloads",
        Path.home() / "AppData" / "Local" / "Microsoft" / "Windows" / "Fonts",
        Path("C:/Windows/Fonts"),
    ]
    for base in candidates:
        if base.is_file():
            return base
        if not base.exists():
            continue
        for pattern in ("*Bebas*.ttf", "*Bebas*.otf"):
            match = next(base.rglob(pattern), None)
            if match:
                return match
    return None


@lru_cache(maxsize=64)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    bebas = _font_path()
    if bebas:
        return ImageFont.truetype(str(bebas), size)

    for fallback in ("arial.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(fallback, size)
        except OSError:
            continue

    return ImageFont.load_default()


def _fit_font(draw: ImageDraw.ImageDraw, text: str, width: int, start_size: int, min_size: int = 14):
    size = start_size
    while size >= min_size:
        trial = _font(size)
        bbox = draw.textbbox((0, 0), text, font=trial)
        if bbox[2] - bbox[0] <= width:
            return trial
        size -= 2
    return _font(min_size)


def _interpolate_color(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    return tuple(
        int(round(start[idx] + (end[idx] - start[idx]) * max(0.0, min(1.0, float(t)))))
        for idx in range(3)
    )


def _percentile_color(percentile: float | int | None) -> tuple[int, int, int]:
    if percentile is None or pd.isna(percentile):
        return BLACK

    value = max(0.0, min(float(percentile), PERCENTILE_MAX))
    if value <= PERCENTILE_MIDPOINT:
        return _interpolate_color(PERCENTILE_RED, PERCENTILE_WHITE, value / PERCENTILE_MIDPOINT)

    return _interpolate_color(
        PERCENTILE_WHITE,
        PERCENTILE_BLUE,
        (value - PERCENTILE_MIDPOINT) / (PERCENTILE_MAX - PERCENTILE_MIDPOINT),
    )


def _red_white_blue_cmap():
    return plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "wsba_percentile_rwb",
        [
            (0.0, tuple(channel / 255.0 for channel in PERCENTILE_RED)),
            (PERCENTILE_MIDPOINT / PERCENTILE_MAX, tuple(channel / 255.0 for channel in PERCENTILE_WHITE)),
            (1.0, tuple(channel / 255.0 for channel in PERCENTILE_BLUE)),
        ],
    )


def _draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=WHITE):
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def _draw_right(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=WHITE):
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    draw.text((xy[0] - width, xy[1]), text, font=font, fill=fill)


def _draw_centered_bold(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font,
    fill=BLACK,
):
    for dx, dy in ((0, 0), (1, 0), (0, 1)):
        _draw_centered(draw, (xy[0] + dx, xy[1] + dy), text, font, fill=fill)


@lru_cache(maxsize=256)
def _load_remote_image_cached(url: str) -> Image.Image | None:
    host = urlparse(url).netloc.lower()
    if host and host in _REMOTE_IMAGE_BLOCKED_HOSTS:
        return None
    try:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=REMOTE_IMAGE_TIMEOUT) as response:
            return Image.open(BytesIO(response.read())).convert("RGBA")
    except Exception:
        if host:
            _REMOTE_IMAGE_BLOCKED_HOSTS.add(host)
        return None


def _load_remote_image(url: str | float | None) -> Image.Image | None:
    if not isinstance(url, str) or not url:
        return None

    image = _load_remote_image_cached(url)
    return None if image is None else image.copy()


def _load_country_flag(country_code: str | float | None) -> Image.Image | None:
    if not isinstance(country_code, str) or not country_code.strip():
        return None
    return _load_remote_image(_country_flag_url(country_code))


def _load_team_logo(team_abbr: str) -> Image.Image | None:
    abbr = str(team_abbr or "").strip().upper()
    if not abbr:
        return None

    for filename in (f"{abbr}_dark.png", f"{abbr}_light.png"):
        image = _load_remote_image(f"{TEAM_LOGO_RAW_BASE}/{filename}")
        if image is not None:
            return image
    return None


def _placeholder_headshot(name: str, team_abbr: str) -> Image.Image:
    image = Image.new("RGBA", (260, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((10, 10, 250, 290), radius=20, fill=(18, 18, 24, 255), outline=DIVIDER, width=3)
    _draw_centered(draw, (130, 110), team_abbr, _font(60), fill=GOLD)
    _draw_centered(draw, (130, 215), name.split(" ")[-1][:12], _font(28), fill=WHITE)
    return image


def _load_headshot(name: str, team_abbr: str, headshot_url: str | None) -> Image.Image:
    image = _load_remote_image(headshot_url)
    if image is None:
        image = _placeholder_headshot(name, team_abbr)
    return image


def _draw_background(draw: ImageDraw.ImageDraw):
    draw.rectangle((0, 0, CARD_SIZE[0], CARD_SIZE[1]), fill=BLACK)


def _draw_section_title(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    title: str,
    font_size: int = 18,
):
    font = _font(font_size)
    _draw_centered(draw, ((x0 + x1) // 2, y), title.upper(), font)
    draw.line((x0, y + 16, x1, y + 16), fill=DIVIDER, width=2)


def _draw_fitted_section_title(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    title: str,
    font_size: int = 18,
    min_size: int = 12,
):
    font = _fit_font(draw, title.upper(), x1 - x0 - 8, font_size, min_size)
    _draw_centered(draw, ((x0 + x1) // 2, y), title.upper(), font)
    draw.line((x0, y + 16, x1, y + 16), fill=DIVIDER, width=2)


def _draw_multiline_section_title(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    title: str,
    font_size: int = 18,
    spacing: int = 0,
):
    font = _font(font_size)
    lines = [line.strip().upper() for line in str(title).split("\n") if line.strip()]
    if not lines:
        lines = [""]
    heights: list[int] = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        heights.append(bbox[3] - bbox[1])
    total_height = sum(heights) + spacing * max(0, len(lines) - 1)
    current_y = y - (total_height / 2)
    center_x = (x0 + x1) // 2
    for line, height in zip(lines, heights):
        _draw_centered(draw, (center_x, int(round(current_y + (height / 2)))), line, font)
        current_y += height + spacing
    line_y = int(round(y + (total_height / 2) + 8))
    draw.line((x0, line_y, x1, line_y), fill=DIVIDER, width=2)


def _draw_metric_line(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    line: MetricLine,
    *,
    row_height: int = SECTION_LINE_ROW_HEIGHT,
    value_width: int = SECTION_VALUE_WIDTH,
    gap: int = SECTION_GAP,
    label_ratio: float = SECTION_LABEL_RATIO,
):
    content_width = x1 - x0
    label_width = int(content_width * label_ratio)
    square_side = row_height - 6
    value_width = square_side
    bar_width = content_width - label_width - value_width - gap

    label_font = _fit_font(draw, line.label.upper(), label_width - 8, 14, 10)
    value_font = _font(16)
    label_x1 = x0 + label_width
    label_center_x = x0 + (label_width // 2)
    value_x0 = label_x1
    value_x1 = value_x0 + square_side
    bar_x0 = value_x1 + gap
    bar_y0 = y + 3
    bar_y1 = bar_y0 + square_side

    label_center_y = (bar_y0 + bar_y1) // 2
    draw.text((label_center_x, label_center_y), line.label.upper(), font=label_font, fill=WHITE, anchor="mm")
    draw.rectangle((bar_x0, bar_y0, x1, bar_y1), fill=BLACK)

    if line.value is None or pd.isna(line.value):
        draw.rectangle((value_x0, bar_y0, value_x1, bar_y1), fill=BLACK)
        return

    percentile = int(round(float(line.value)))
    color = _percentile_color(percentile)
    draw.rectangle((value_x0, bar_y0, value_x1, bar_y1), fill=color)
    _draw_centered_bold(
        draw,
        ((value_x0 + value_x1) // 2, (bar_y0 + bar_y1) // 2 - 1),
        str(percentile),
        value_font,
        fill=BLACK,
    )

    fill_width = int(bar_width * max(0, min(percentile, PERCENTILE_MAX)) / PERCENTILE_MAX)
    if fill_width > 0:
        draw.rectangle((bar_x0, bar_y0, bar_x0 + fill_width, bar_y1), fill=color)


def _draw_dual_metric_row(
    draw: ImageDraw.ImageDraw,
    x0: int,
    x1: int,
    y: int,
    label: str,
    offense_value: float | int | None,
    defense_value: float | int | None,
    *,
    row_height: int = 20,
    label_width: int = 126,
    column_gap: int = 12,
):
    content_width = x1 - x0
    metric_width = max(80, (content_width - label_width - column_gap) // 2)
    offense_x0 = x0 + label_width
    defense_x0 = offense_x0 + metric_width + column_gap
    label_font = _fit_font(draw, label.upper(), label_width - 8, 12, 9)
    draw.text((x0 + 2, y + (row_height // 2)), label.upper(), font=label_font, fill=WHITE, anchor="lm")

    def draw_metric_box(metric_x0: int, value: float | int | None):
        square_side = row_height - 6
        square_x1 = metric_x0 + square_side
        bar_x0 = square_x1 + 4
        bar_x1 = metric_x0 + metric_width
        bar_y0 = y + 3
        bar_y1 = bar_y0 + square_side
        draw.rectangle((metric_x0, bar_y0, square_x1, bar_y1), fill=BLACK)
        draw.rectangle((bar_x0, bar_y0, bar_x1, bar_y1), fill=BLACK)
        if value is None or pd.isna(value):
            return
        percentile = int(round(float(value)))
        color = _percentile_color(percentile)
        draw.rectangle((metric_x0, bar_y0, square_x1, bar_y1), fill=color)
        _draw_centered_bold(
            draw,
            ((metric_x0 + square_x1) // 2, (bar_y0 + bar_y1) // 2 - 1),
            str(percentile),
            _font(13),
            fill=BLACK,
        )
        fill_width = int((bar_x1 - bar_x0) * max(0, min(percentile, PERCENTILE_MAX)) / PERCENTILE_MAX)
        if fill_width > 0:
            draw.rectangle((bar_x0, bar_y0, bar_x0 + fill_width, bar_y1), fill=color)

    draw_metric_box(offense_x0, offense_value)
    draw_metric_box(defense_x0, defense_value)


def _draw_donut(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    percentile: int,
    *,
    outer: int = 92,
    inner: int = 46,
    font_size: int = 72,
):
    x, y = center
    ring_color = _percentile_color(percentile)
    ring_bounds = (x - outer + 7, y - outer + 7, x + outer - 7, y + outer - 7)
    start_angle = -90
    sweep_angle = max(0.0, min(float(percentile), PERCENTILE_MAX)) * 360.0 / PERCENTILE_MAX
    draw.ellipse((x - outer, y - outer, x + outer, y + outer), fill=(15, 15, 19))
    draw.arc(
        ring_bounds,
        start=0,
        end=359.9,
        fill=(28, 28, 34),
        width=22,
    )
    draw.arc(
        ring_bounds,
        start=start_angle,
        end=start_angle + sweep_angle,
        fill=ring_color,
        width=22,
    )
    draw.ellipse((x - inner, y - inner, x + inner, y + inner), fill=BLACK)
    draw.text((x, y), str(percentile), font=_font(font_size), fill=WHITE, anchor="mm")


def _draw_team_badge(draw: ImageDraw.ImageDraw, center: tuple[int, int], team_abbr: str):
    x, y = center
    draw.ellipse((x - 38, y - 38, x + 38, y + 38), fill=WHITE, outline=GOLD, width=6)
    _draw_centered(draw, (x, y + 2), team_abbr.upper(), _font(34), fill=BLACK)


def _paste_team_logo(image: Image.Image, rect: PixelRect, team_abbr: str):
    logo = _load_team_logo(team_abbr)
    if logo is None:
        draw = ImageDraw.Draw(image)
        _draw_team_badge(draw, (rect.center_x, rect.center_y), team_abbr)
        return

    logo = ImageOps.contain(logo, (rect.width, rect.height))
    x = rect.left + (rect.width - logo.width) // 2
    y = rect.top + (rect.height - logo.height) // 2
    image.paste(logo, (x, y), logo if logo.mode == "RGBA" else None)


def wsba_skater_card(
    player_id: int,
    seasons: list[int],
    output_path: str | Path | None = None,
    card_layout: dict[str, PixelRect] | None = None,
    microimpacts: bool = False,
    plotting_metric: str = "fenwick",
) -> Image.Image:
    plotting_metric = _normalize_rapm_plotting_metric(plotting_metric)
    dataset = _card_dataset(seasons)
    row = dataset.loc[dataset["player_id"].astype(int) == int(player_id)]
    if row.empty:
        raise ValueError(f"Player id {player_id} was not found in {HS_DATA_PATH}.")

    player = row.iloc[0]
    output = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_PATH
    layout = _resolve_card_layout(card_layout)
    team_primary, team_secondary = _team_colors(str(player["team_abbr"]))

    image = Image.new("RGB", CARD_SIZE, BLACK)
    draw = ImageDraw.Draw(image)
    _draw_background(draw)

    name = str(player["player_name"]).upper()
    title_rect = layout["title"]
    name_font = _fit_font(draw, name, title_rect.width, 54, 28)
    _draw_centered(draw, (title_rect.center_x, title_rect.center_y), name, name_font)

    top_bar = layout["header_bar"]
    draw.rectangle((top_bar.left, top_bar.top, top_bar.right, top_bar.bottom), fill=team_primary)

    team_title_rect = layout["team_title"]
    bio_title_rect = layout["bio_title"]
    _draw_section_title(draw, team_title_rect.left, team_title_rect.right, team_title_rect.center_y, "Team", font_size=15)
    _draw_section_title(draw, bio_title_rect.left, bio_title_rect.right, bio_title_rect.center_y, "Biographical Information", font_size=14)

    team_panel = layout["team_panel"]
    _paste_team_logo(image, team_panel, str(player["team_abbr"]))

    headshot = _load_headshot(name, str(player["team_abbr"]), player["headshot"])
    headshot_rect = layout["headshot_panel"]
    headshot = ImageOps.contain(headshot, (headshot_rect.width, headshot_rect.height))
    headshot_x = headshot_rect.left + (headshot_rect.width - headshot.width) // 2
    headshot_y = headshot_rect.top + (headshot_rect.height - headshot.height) // 2
    image.paste(headshot, (headshot_x, headshot_y), headshot if headshot.mode == "RGBA" else None)

    bio_font = _font(13)
    bio_value_font = _font(13)
    bio_panel = layout["bio_panel"]
    labels = [
        ("AGE", player["age"]),
        ("HEIGHT (IN)", player["height_in"]),
        ("WEIGHT (LBS)", player["weight_lbs"]),
        ("POSITION", player["position_group"]),
        ("HANDEDNESS", player["handedness"]),
        ("NATIONALITY", player["birth_country"]),
    ]
    base_y = bio_panel.top + 5
    for idx, (label, value) in enumerate(labels):
        y = base_y + idx * 20
        draw.text((bio_panel.left + 3, y), label, font=bio_font, fill=WHITE)
        if label == "NATIONALITY":
            flag = _load_country_flag(value)
            if flag is not None:
                flag = ImageOps.contain(flag, (28, 20), method=Image.Resampling.LANCZOS)
                flag_x = bio_panel.right - 4 - flag.width
                flag_y = y + 1
                image.paste(flag, (flag_x, flag_y), flag)
                continue

        rendered_value = (
            ""
            if pd.isna(value)
            else int(value)
            if isinstance(value, float) and float(value).is_integer()
            else value
        )
        _draw_right(draw, (bio_panel.right - 4, y), f"{rendered_value}", bio_value_font)

    season_font = _font(20)
    season_rect = layout["season_label"]
    _draw_centered(draw, (season_rect.center_x, season_rect.center_y), str(player["season_label"]), season_font)

    info_divider = layout["info_divider"]
    draw.rectangle((info_divider.left, info_divider.top, info_divider.right, info_divider.bottom), fill=team_secondary)

    gar_title_rect = layout["gar_title"]
    impacts_title_rect = layout["impacts_title"]
    locational_title_ev_rect = layout["locational_title_ev"]
    locational_title_st_rect = layout["locational_title_st"]
    deployment_title_rect = layout["deployment_title"]
    tendencies_title_rect = layout["tendencies_title"]
    _draw_section_title(draw, gar_title_rect.left, gar_title_rect.right, gar_title_rect.center_y, "GAR %", font_size=22)
    _draw_section_title(draw, impacts_title_rect.left, impacts_title_rect.right, impacts_title_rect.center_y, "Impacts")
    _draw_section_title(
        draw,
        locational_title_ev_rect.left,
        locational_title_ev_rect.right,
        locational_title_ev_rect.center_y,
        "Even Strength Locational Impacts",
        font_size=13,
    )
    _draw_section_title(
        draw,
        locational_title_st_rect.left,
        locational_title_st_rect.right,
        locational_title_st_rect.center_y,
        "Special Teams Locational Impacts",
        font_size=13,
    )
    _draw_section_title(draw, deployment_title_rect.left, deployment_title_rect.right, deployment_title_rect.center_y, "Deployment")
    _draw_section_title(
        draw,
        tendencies_title_rect.left,
        tendencies_title_rect.right,
        tendencies_title_rect.center_y,
        "Microimpacts" if microimpacts else "Tendencies",
    )

    gar_panel = layout["gar_panel"]
    impacts_panel = layout["impacts_panel"]
    locational_ev_panel = layout["locational_panel_ev"]
    locational_st_panel = layout["locational_panel_st"]
    locational_cbar_rect = layout["locational_cbar"]
    deployment_panel = layout["deployment_panel"]
    tendencies_panel = layout["tendencies_panel"]

    try:
        player_pp_toi = player.get("power_play_time_on_ice_per_games_played", 0)
        player_pk_toi = player.get("penalty_kill_time_on_ice_per_games_played", 0)
        player_pp_toi = 0.0 if player_pp_toi is None or pd.isna(player_pp_toi) else float(player_pp_toi)
        player_pk_toi = 0.0 if player_pk_toi is None or pd.isna(player_pk_toi) else float(player_pk_toi)
        ev_bytes, cbar_bytes, st_bytes = _rapm_panel_images_cached(
            int(player_id),
            _seasons_key(seasons),
            round(player_pp_toi, 6),
            round(player_pk_toi, 6),
            plotting_metric,
            (locational_ev_panel.width, locational_ev_panel.height),
            (locational_st_panel.width, locational_st_panel.height),
            (locational_cbar_rect.width, locational_cbar_rect.height),
        )
        ev_image = _image_from_png_bytes(ev_bytes)
        cbar_image = _image_from_png_bytes(cbar_bytes)
        st_image = _image_from_png_bytes(st_bytes)
        image.paste(ev_image, (locational_ev_panel.left, locational_ev_panel.top), ev_image)
        image.paste(cbar_image, (locational_cbar_rect.left, locational_cbar_rect.top), cbar_image)
        image.paste(st_image, (locational_st_panel.left, locational_st_panel.top), st_image)
    except Exception:
        pass

    gar_pct = int(round(float(player["goals_above_replacement_percentile"]))) if pd.notna(player["goals_above_replacement_percentile"]) else 0
    _draw_donut(draw, (gar_panel.center_x, gar_panel.top + 110), gar_pct)

    for idx, label in enumerate(["Offense", "Defense", "Penalties"]):
        _draw_metric_line(
            draw,
            gar_panel.left + 8,
            gar_panel.right,
            gar_panel.bottom - 84 + idx * 28,
            MetricLine(label, player[f"{METRIC_MAP[label]}_percentile"]),
            row_height=SECTION_LINE_ROW_HEIGHT,
            label_ratio=SECTION_LABEL_RATIO,
        )

    impacts = [
        MetricLine("EV Generation", player[f"{METRIC_MAP['EV Generation']}_percentile"]),
        MetricLine("EV Suppression", player[f"{METRIC_MAP['EV Suppression']}_percentile"]),
        MetricLine("PP Generation", player[f"{METRIC_MAP['PP Generation']}_percentile"]),
        MetricLine("PK Suppression", player[f"{METRIC_MAP['PK Suppression']}_percentile"]),
        MetricLine("Finishing", player[f"{METRIC_MAP['Finishing']}_percentile"]),
        MetricLine("Induction", player[f"{METRIC_MAP['Induction']}_percentile"]),
        MetricLine("Taken Penalties", player[f"{METRIC_MAP['Taken Penalties']}_percentile"]),
        MetricLine("Drawn Penalties", player[f"{METRIC_MAP['Drawn Penalties']}_percentile"]),
    ]
    impact_offsets = [14, 42, 90, 118, 166, 194, 240, 268]
    for line, offset in zip(impacts, impact_offsets):
        _draw_metric_line(
            draw,
            impacts_panel.left + 8,
            impacts_panel.right,
            impacts_panel.top + offset,
            line,
            row_height=SECTION_LINE_ROW_HEIGHT,
            label_ratio=SECTION_LABEL_RATIO,
        )

    deployment = [
        MetricLine("Time On Ice", player[f"{METRIC_MAP['Time On Ice']}_percentile"]),
        MetricLine("Competition", player[f"{METRIC_MAP['Competition']}_percentile"]),
        MetricLine("Teammates", player[f"{METRIC_MAP['Teammates']}_percentile"]),
        MetricLine("O-Zone Faceoffs", player[f"{METRIC_MAP['O-Zone Faceoffs']}_percentile"]),
        MetricLine("N-Zone Faceoffs", player[f"{METRIC_MAP['N-Zone Faceoffs']}_percentile"]),
        MetricLine("D-Zone Faceoffs", player[f"{METRIC_MAP['D-Zone Faceoffs']}_percentile"]),
    ]
    deployment_offsets = [8, 36, 64, 102, 130, 158]
    for line, offset in zip(deployment, deployment_offsets):
        _draw_metric_line(
            draw,
            deployment_panel.left + 8,
            deployment_panel.right,
            deployment_panel.top + offset,
            line,
            row_height=SECTION_LINE_ROW_HEIGHT,
            label_ratio=SECTION_LABEL_RATIO,
        )

    section_lines = (
        [
            MetricLine(
                label,
                0 if pd.isna(player.get(f"{METRIC_MAP[label]}_percentile")) else player.get(f"{METRIC_MAP[label]}_percentile"),
            )
            for label in MICROIMPACT_LABELS
        ]
        if microimpacts
        else [
            MetricLine("Shot Rate", player[f"{METRIC_MAP['Shot Rate']}_percentile"]),
            MetricLine("Shot Quality", player[f"{METRIC_MAP['Shot Quality']}_percentile"]),
            MetricLine("Pass Rate", player[f"{METRIC_MAP['Pass Rate']}_percentile"]),
            MetricLine("Pass Quality", player[f"{METRIC_MAP['Pass Quality']}_percentile"]),
            MetricLine("Hits Given", player[f"{METRIC_MAP['Hits Given']}_percentile"]),
            MetricLine("Hits Taken", player[f"{METRIC_MAP['Hits Taken']}_percentile"]),
            MetricLine("Blocked Shots", player[f"{METRIC_MAP['Blocked Shots']}_percentile"]),
            MetricLine("Giveaways", player[f"{METRIC_MAP['Giveaways']}_percentile"]),
            MetricLine("Takeaways", player[f"{METRIC_MAP['Takeaways']}_percentile"]),
            MetricLine("Consistency", player[f"{METRIC_MAP['Consistency']}_percentile"]),
        ]
    )
    section_offsets = (
        [8, 34, 60, 86, 134, 160, 208, 234, 282, 308, 334]
        if microimpacts
        else [8, 34, 84, 110, 160, 186, 212, 260, 286, 334]
    )
    for line, offset in zip(section_lines, section_offsets):
        _draw_metric_line(
            draw,
            tendencies_panel.left + 8,
            tendencies_panel.right,
            tendencies_panel.top + offset,
            line,
            row_height=SECTION_LINE_ROW_HEIGHT,
            label_ratio=SECTION_LABEL_RATIO,
        )

    footer_blue = layout["footer_blue"]
    footer_red = layout["footer_red"]
    footer_text_rect = layout["footer_text"]
    draw.rectangle((footer_blue.left, footer_blue.top, footer_blue.right, footer_blue.bottom), fill=BLUE)
    draw.rectangle((footer_red.left, footer_red.top, footer_red.right, footer_red.bottom), fill=RED)
    footer_font = _font(13)
    draw.text((footer_text_rect.left + 2, footer_text_rect.top), "PERCENTILES ARE RELATIVE TO POSITION AND SEASON", font=footer_font, fill=WHITE)
    draw.text(
        (footer_text_rect.left + 2, footer_text_rect.top + 24),
        "ALL PREREQUISITE DATA IS ACQUIRED FROM THE NHL OR COREY SZNJADER'S ALL THREE ZONES PROJECT.",
        font=_font(12),
        fill=WHITE,
    )

    if WSBA_LOGO_PATH.exists():
        logo = Image.open(WSBA_LOGO_PATH).convert("RGBA")
        logo = ImageOps.contain(logo, (48, 48))
        image.paste(logo, (footer_red.right - 54, footer_text_rect.top - 2), logo)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return image


def wsba_coach_card(
    coach_name: str,
    seasons: list[int],
    output_path: str | Path | None = None,
    card_layout: dict[str, PixelRect] | None = None,
    plotting_metric: str = "fenwick",
) -> Image.Image:
    plotting_metric = _normalize_rapm_plotting_metric(plotting_metric)
    dataset = _coach_card_dataset(seasons)
    coach_key = str(coach_name or "").strip().upper()
    row = dataset.loc[dataset["coach_name"].astype(str).str.upper().eq(coach_key)]
    if row.empty:
        raise ValueError(f"Coach {coach_name} was not found in {COACH_GAR_DATA_PATH}.")

    coach = row.iloc[0]
    output = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_PATH
    layout = _resolve_coach_card_layout(card_layout)
    team_primary, team_secondary = _team_colors(str(coach["team_abbr"]))

    image = Image.new("RGB", CARD_SIZE, BLACK)
    draw = ImageDraw.Draw(image)
    _draw_background(draw)

    name = str(coach["coach_name"]).upper()
    title_rect = layout["title"]
    name_font = _fit_font(draw, name, title_rect.width, 42, 22)
    _draw_centered(draw, (title_rect.center_x, title_rect.center_y), name, name_font)

    top_bar = layout["header_bar"]
    draw.rectangle((top_bar.left, top_bar.top, top_bar.right, top_bar.bottom), fill=team_primary)

    _paste_team_logo(image, layout["identity_logo"], str(coach["team_abbr"]))
    season_rect = layout["season_label"]
    _draw_centered(draw, (season_rect.center_x, season_rect.center_y), str(coach.get("season_label") or _season_label(seasons)), _font(18))

    info_divider = layout["info_divider"]
    draw.rectangle((info_divider.left, info_divider.top, info_divider.right, info_divider.bottom), fill=team_secondary)

    _draw_section_title(
        draw,
        layout["systems_title"].left,
        layout["systems_title"].right,
        layout["systems_title"].center_y,
        "Systems",
        font_size=17,
    )
    _draw_fitted_section_title(
        draw,
        layout["offense_score_title"].left,
        layout["offense_score_title"].right,
        layout["offense_score_title"].center_y,
        "Offensive Score State Impacts",
        font_size=18,
        min_size=12,
    )
    _draw_fitted_section_title(
        draw,
        layout["defense_score_title"].left,
        layout["defense_score_title"].right,
        layout["defense_score_title"].center_y,
        "Defensive Score State Impacts",
        font_size=18,
        min_size=12,
    )

    systems_title_rect = layout["locational_title_systems"]
    _draw_section_title(
        draw,
        systems_title_rect.left,
        systems_title_rect.right,
        systems_title_rect.center_y,
        "Systems Locational Impacts",
        font_size=14,
    )
    overall_title_rect = layout["locational_title_overall"]
    _draw_section_title(
        draw,
        overall_title_rect.left,
        overall_title_rect.right,
        overall_title_rect.center_y,
        "Overall Locational Impacts",
        font_size=14,
    )

    systems_panel = layout["systems_panel"]
    systems_lines = [
        MetricLine("EV Generation", coach.get("even_strength_generation_percentile")),
        MetricLine("EV Suppression", coach.get("even_strength_suppression_percentile")),
        MetricLine("Taken Penalties", coach.get("taken_penalties_percentile")),
        MetricLine("Drawn Penalties", coach.get("drawn_penalties_percentile")),
        MetricLine("GAR", coach.get("goals_above_replacement_percentile")),
    ]
    systems_offsets = [0, 20, 40, 60, 80]
    for line, offset in zip(systems_lines, systems_offsets):
        _draw_metric_line(
            draw,
            systems_panel.left + 4,
            systems_panel.right - 4,
            systems_panel.top + offset,
            line,
            row_height=20,
            label_ratio=0.48,
        )

    offense_panel = layout["offense_score_panel"]
    defense_panel = layout["defense_score_panel"]
    for idx, (label, state) in enumerate(COACH_SCORE_STATE_ROWS):
        y = offense_panel.top + 2 + idx * 43
        _draw_metric_line(
            draw,
            offense_panel.left + 4,
            offense_panel.right - 4,
            y,
            MetricLine(label, coach.get(f"{state}_even_strength_generation_percentile")),
            row_height=26,
            label_ratio=0.4,
        )
        _draw_metric_line(
            draw,
            defense_panel.left + 4,
            defense_panel.right - 4,
            y,
            MetricLine(label, coach.get(f"{state}_even_strength_suppression_percentile")),
            row_height=26,
            label_ratio=0.4,
        )

    try:
        systems_bytes, cbar_bytes, overall_bytes = _coach_rapm_panel_images_cached(
            str(coach["coach_name"]),
            _seasons_key(seasons),
            plotting_metric,
            (layout["locational_panel_systems"].width, layout["locational_panel_systems"].height),
            (layout["locational_panel_overall"].width, layout["locational_panel_overall"].height),
            (layout["locational_cbar"].width, layout["locational_cbar"].height),
        )
        systems_image = _image_from_png_bytes(systems_bytes)
        cbar_image = _image_from_png_bytes(cbar_bytes)
        overall_image = _image_from_png_bytes(overall_bytes)
        image.paste(systems_image, (layout["locational_panel_systems"].left, layout["locational_panel_systems"].top), systems_image)
        image.paste(cbar_image, (layout["locational_cbar"].left, layout["locational_cbar"].top), cbar_image)
        image.paste(overall_image, (layout["locational_panel_overall"].left, layout["locational_panel_overall"].top), overall_image)
    except Exception:
        pass

    footer_blue = layout["footer_blue"]
    footer_red = layout["footer_red"]
    footer_text_rect = layout["footer_text"]
    draw.rectangle((footer_blue.left, footer_blue.top, footer_blue.right, footer_blue.bottom), fill=BLUE)
    draw.rectangle((footer_red.left, footer_red.top, footer_red.right, footer_red.bottom), fill=RED)
    footer_font = _font(13)
    draw.text((footer_text_rect.left + 2, footer_text_rect.top), "PERCENTILES ARE RELATIVE TO POSITION AND SEASON", font=footer_font, fill=WHITE)
    draw.text(
        (footer_text_rect.left + 2, footer_text_rect.top + 24),
        "ALL PREREQUISITE DATA IS ACQUIRED FROM THE NHL OR COREY SZNJADER'S ALL THREE ZONES PROJECT.",
        font=_font(12),
        fill=WHITE,
    )

    if WSBA_LOGO_PATH.exists():
        logo = Image.open(WSBA_LOGO_PATH).convert("RGBA")
        logo = ImageOps.contain(logo, (48, 48))
        image.paste(logo, (footer_red.right - 54, footer_text_rect.top - 2), logo)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return image
    fenwick_weights = rate_frame["fenwick_per_sixty"] * toi / 60.0
    fenwick_assist_weights = rate_frame["fenwick_assists_per_sixty"] * toi / 60.0
