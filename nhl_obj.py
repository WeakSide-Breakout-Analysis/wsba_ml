import random
import string
from pathlib import Path
import pandas as pd
import numpy as np
import wsba_hockey as wsba

### NHL ENTITY OBJECTS ###

## PLAYER DATA ##
SEASONS = wsba.SEASON_NAMES.keys()
ROSTERS = wsba.repo_load_rosters(SEASONS)
if not isinstance(ROSTERS, pd.DataFrame):
    ROSTERS = pd.DataFrame(ROSTERS.to_dicts())
PLAYER_LOOKUP = ROSTERS.drop_duplicates('player_id').set_index('player_id').to_dict('index')
SIMULATION_NAMES_PATH = Path(__file__).resolve().parent / "simulation" / "sim_data" / "names.csv"


def _load_placeholder_name_pool():
    try:
        frame = pd.read_csv(SIMULATION_NAMES_PATH)
    except Exception:
        return []

    if "first_name" not in frame.columns or "last_name" not in frame.columns:
        return []

    out = []
    for first_name, last_name in zip(frame["first_name"], frame["last_name"]):
        first = str(first_name or "").strip()
        last = str(last_name or "").strip()
        if first and first.lower() != "nan" and last and last.lower() != "nan":
            out.append((first.upper(), last.upper()))
    return out


PLACEHOLDER_NAME_POOL = _load_placeholder_name_pool()
LEAGUE_AVERAGE_SIM_RATES = {
    "shot_rate": 12.0,
    "fenwick_per_corsi": 0.695,
    "shots_per_fenwick": 0.662,
    "xg_per_fenwick": 0.064,
    "primary_pass_rate": 4.5,
    "secondary_pass_rate": 3.5,
    "block_rate": 2.2,
    "penalties_taken_rate": 0.45,
    "penalties_drawn_rate": 0.45,
}


def _sim_positive_value(value, default):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric) or numeric <= 0.0:
        return float(default)
    return numeric


def _sim_probability_value(value, default):
    numeric = _sim_positive_value(value, default)
    if numeric > 1.0 and numeric <= 100.0:
        numeric /= 100.0
    return float(np.clip(numeric, 0.0, 1.0))
COACH_SCORE_STATES = ['tied', 'lead_1', 'lead_2', 'lead_3_plus', 'behind_1', 'behind_2', 'behind_3_plus']
COACH_SCORE_METRICS = ['corsi', 'fenwick', 'expected_goals', 'goals']
COACH_SCORE_SIDES = ['for', 'against']
COACH_SCORE_STRENGTH_PREFIXES = {
    'even_strength': 'ev',
    'power_play': 'pp',
    'penalty_kill': 'pk',
}
COACH_SCORE_COLUMNS = [
    f'rapm_{state}_{metric}_{side}_per_sixty'
    for state in COACH_SCORE_STATES
    for metric in COACH_SCORE_METRICS
    for side in COACH_SCORE_SIDES
]

# Helper function to quickly generate a lineup template
def build_lineup_template(
    df,
    group_col="group",
    line_col="line",
    player_col="player_id",
    pct_col="time_on_ice_percentage",
):
    out = {}

    group_order = ["EV_F", "EV_D", "PP", "PK", "G"]

    for group_name in group_order:
        g = df.loc[df[group_col] == group_name].copy()
        if g.empty:
            continue

        group_dict = {}
        for line_num, line_df in g.groupby(line_col, sort=True):
            line_df = line_df.sort_values(player_col)

            player_ids = line_df[player_col].astype(int).tolist()

            pct_vals = line_df[pct_col].dropna().unique().tolist()
            pct = float(pct_vals[0]) if pct_vals else None

            group_dict[int(line_num)] = player_ids + [pct]

        out[group_name] = group_dict

    return out

class NHLPlayer:
    def __init__(self, player_id, data):
        self.player_id = int(player_id)

        # Search in package database to save time finding player info online
        player = PLAYER_LOOKUP.get(self.player_id, {})

        self.player_name = player.get('player_name', 'Not Found')
        self.position = player.get('position', 'Not Found')
        self.handedness = player.get('handedness', 'Not Found')
        self.height_in = player.get('height_in', 'Not Found')
        self.weight_lbs = player.get('weight_lbs', 'Not Found')
        self.birth_country = player.get('birth_country', 'Not Found')

        birth_date = player.get('birth_date')
        self.birth_date = pd.to_datetime(birth_date) if birth_date else None
        self.age = self._calc_age(self.birth_date)

        if data is not None and 'player_id' in data.columns:
            player_data = data[data['player_id'] == self.player_id]
            if not player_data.empty:
                for attr, column in (
                    ('player_name', 'player_name'),
                    ('position', 'position'),
                    ('handedness', 'handedness'),
                    ('height_in', 'height_in'),
                    ('weight_lbs', 'weight_lbs'),
                    ('birth_country', 'birth_country'),
                ):
                    if column in player_data.columns:
                        value = player_data[column].iloc[0]
                        if value is not None and not pd.isna(value):
                            setattr(self, attr, value)

        self.obtain_data(data)

    def _per_sixty(self, col, toi):
        return (col / toi) * 60

    # Our input data for individual RAPM stats (such as goals) are per-sixty impacts
    # We would like per-shot impact for the simulation (modifying the league-average fenwick shot percentage)
    def _per_event_impact(self, player_data, impact_col, volume_col, fallback_col=None):
        if not player_data.empty and impact_col in player_data.columns:
            value = player_data[impact_col].iloc[0]
            if value is not None and not pd.isna(value):
                return float(value)

        if player_data.empty or fallback_col is None:
            return 0.0

        impact_rate = player_data[fallback_col].iloc[0] if fallback_col in player_data.columns else None
        volume_rate = player_data[volume_col].iloc[0] if volume_col in player_data.columns else None

        if (
            impact_rate is None or pd.isna(impact_rate)
            or volume_rate is None or pd.isna(volume_rate)
            or float(volume_rate) <= 0
        ):
            return 0.0

        return float(impact_rate) / float(volume_rate)

    # Helper to obtain a given column value for the player
    def _select_data(self, player_data, column, default=0.0):
        if player_data.empty or column not in player_data.columns:
            return default

        value = player_data[column].iloc[0]
        if value is None or pd.isna(value):
            return default

        return value

    # Calculate age as of today
    def _calc_age(self, birth_date):
        if pd.isnull(birth_date):
            return None
        today = pd.Timestamp.today()
        age = today.year - birth_date.year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1
        return age

    # Obtain performance data on the specified player
    def obtain_data(self, data):
        player_data = data[data['player_id'] == self.player_id]

        if self.position == 'G':
            self.time_on_ice = self._select_data(player_data, 'time_on_ice')

            self.goaltending = self._select_data(player_data, 'rapm_goals_against_per_sixty')
            self.shot_rate_against = self._select_data(player_data, 'fenwick_against_per_sixty')
            self.goaltending_impact = self._per_event_impact(
                player_data,
                'goaltending_impact',
                'fenwick_against_per_sixty',
                'rapm_goals_against_per_sixty'
            )
        else:
            ## USAGE
            self.time_on_ice = self._select_data(player_data, 'time_on_ice')

            self.ev_time_on_ice = self._select_data(player_data, 'even_strength_time_on_ice')
            self.pp_time_on_ice = self._select_data(player_data, 'power_play_time_on_ice')
            self.pk_time_on_ice = self._select_data(player_data, 'penalty_kill_time_on_ice')

            ## ABILITIES
            self.ev_corsi_offense = self._select_data(player_data, 'even_strength_rapm_corsi_for_per_sixty')
            self.ev_corsi_defense = self._select_data(player_data, 'even_strength_rapm_corsi_against_per_sixty')
            self.ev_fenwick_offense = self._select_data(player_data, 'even_strength_rapm_fenwick_for_per_sixty')
            self.ev_fenwick_defense = self._select_data(player_data, 'even_strength_rapm_fenwick_against_per_sixty')
            self.ev_xg_offense = self._select_data(player_data, 'even_strength_rapm_expected_goals_for_per_sixty')
            self.ev_xg_defense = self._select_data(player_data, 'even_strength_rapm_expected_goals_against_per_sixty')
            
            self.pp_corsi_offense = self._select_data(player_data, 'power_play_rapm_corsi_for_per_sixty')
            self.pk_corsi_defense = self._select_data(player_data, 'penalty_kill_rapm_corsi_against_per_sixty')
            self.pp_fenwick_offense = self._select_data(player_data, 'power_play_rapm_fenwick_for_per_sixty')
            self.pk_fenwick_defense = self._select_data(player_data, 'penalty_kill_rapm_fenwick_against_per_sixty')
            self.pp_xg_offense = self._select_data(player_data, 'power_play_rapm_expected_goals_for_per_sixty')
            self.pk_xg_defense = self._select_data(player_data, 'penalty_kill_rapm_expected_goals_against_per_sixty')

            self.ev_fenwick_offense_map = self._select_data(player_data, 'even_strength_rapm_fenwick_for_per_sixty_map', None)
            self.ev_fenwick_defense_map = self._select_data(player_data, 'even_strength_rapm_fenwick_against_per_sixty_map', None)
            self.pp_fenwick_offense_map = self._select_data(player_data, 'power_play_rapm_fenwick_for_per_sixty_map', None)
            self.pk_fenwick_defense_map = self._select_data(player_data, 'penalty_kill_rapm_fenwick_against_per_sixty_map', None)

            self.finishing = self._select_data(player_data, 'rapm_goals_per_sixty')
            self.induction = self._select_data(player_data, 'rapm_primary_assists_per_sixty')
            self.secondary_induction = self._select_data(player_data, 'rapm_secondary_assists_per_sixty')
            self.finishing_impact = self._per_event_impact(
                player_data,
                'finishing_impact',
                'fenwick_per_sixty',
                'rapm_goals_per_sixty'
            )
            self.induction_impact = self._per_event_impact(
                player_data,
                'induction_impact',
                'primary_fenwick_assists_per_sixty',
                'rapm_primary_assists_per_sixty'
            )
            self.secondary_induction_impact = self._per_event_impact(
                player_data,
                'secondary_induction_impact',
                'secondary_fenwick_assists_per_sixty',
                'rapm_secondary_assists_per_sixty'
            )

            self.taken_pens = self._select_data(player_data, 'rapm_penalty_minutes_for_per_sixty')
            self.drawn_pens = self._select_data(player_data, 'rapm_penalty_minutes_against_per_sixty')

            ## TENDENCIES
            self.shot_rate = _sim_positive_value(
                self._select_data(player_data, 'corsi_per_sixty'),
                LEAGUE_AVERAGE_SIM_RATES["shot_rate"],
            )
            self.fenwick_per_corsi = _sim_probability_value(
                self._select_data(player_data, 'fenwick_per_corsi'),
                LEAGUE_AVERAGE_SIM_RATES["fenwick_per_corsi"],
            )
            self.shots_per_fenwick = _sim_probability_value(
                self._select_data(player_data, 'shots_per_fenwick'),
                LEAGUE_AVERAGE_SIM_RATES["shots_per_fenwick"],
            )
            self.xg_per_fenwick = _sim_positive_value(
                self._select_data(player_data, 'expected_goals_per_fenwick', self._select_data(player_data, 'xg_per_fenwick')),
                LEAGUE_AVERAGE_SIM_RATES["xg_per_fenwick"],
            )
            self.primary_pass_rate = _sim_positive_value(
                self._select_data(player_data, 'primary_fenwick_assists_per_sixty'),
                LEAGUE_AVERAGE_SIM_RATES["primary_pass_rate"],
            )
            self.secondary_pass_rate = _sim_positive_value(
                self._select_data(player_data, 'secondary_fenwick_assists_per_sixty'),
                LEAGUE_AVERAGE_SIM_RATES["secondary_pass_rate"],
            )
            self.block_rate = _sim_positive_value(
                self._select_data(player_data, 'blocked_shots_per_sixty'),
                LEAGUE_AVERAGE_SIM_RATES["block_rate"],
            )
            self.penalties_taken_rate = _sim_positive_value(
                self._select_data(player_data, 'minor_penalties_per_sixty'),
                LEAGUE_AVERAGE_SIM_RATES["penalties_taken_rate"],
            )
            self.penalties_drawn_rate = _sim_positive_value(
                self._select_data(player_data, 'minor_penalties_drawn_per_sixty'),
                LEAGUE_AVERAGE_SIM_RATES["penalties_drawn_rate"],
            )
            self.giveaways_rate = self._select_data(player_data, 'giveaways_per_sixty')
            self.takeaways_rate = self._select_data(player_data, 'takeaways_per_sixty')
            self.hits_for_rate = self._select_data(player_data, 'hits_for_per_sixty')
            self.hits_against_rate = self._select_data(player_data, 'hits_against_per_sixty')
            self.faceoff_rate = self._select_data(player_data, 'faceoffs_per_sixty')
            self.faceoff_percentage = self._select_data(player_data, 'faceoff_percentage')
            self.offensive_zone_faceoffs = self._select_data(player_data, 'offensive_zone_faceoffs_per_sixty')
            self.neutral_zone_faceoffs = self._select_data(player_data, 'neutral_zone_faceoffs_per_sixty')
            self.defensive_zone_faceoffs = self._select_data(player_data, 'defensive_zone_faceoffs_per_sixty')

class NHLCoach:
    def __init__(self, team_abbr=None, coach_name=None, data=None):
        self.team_abbr = team_abbr
        self.coach_name = coach_name or 'Unknown Coach'
        self.position = 'coach'
        self.obtain_data(data)

    # Helper to obtain a given column value for the coach
    def _select_data(self, player_data, column, default=0.0):
        if player_data.empty or column not in player_data.columns:
            return default

        value = player_data[column].iloc[0]
        if value is None or pd.isna(value):
            return default

        return value

    def obtain_data(self, data):
        coach_data = data[data['position'] == 'coach'] if data is not None and not data.empty else pd.DataFrame()

        if self.coach_name != 'Unknown Coach' and 'coach_name' in coach_data.columns:
            coach_data = coach_data[coach_data['coach_name'] == self.coach_name]

        self.coach_name = self._select_data(coach_data, 'coach_name', self.coach_name)
        self.team_abbr = self._select_data(coach_data, 'team_abbr', self.team_abbr)

        ## USAGE
        self.time_on_ice = self._select_data(coach_data, 'time_on_ice')
        self.ev_time_on_ice = self._select_data(coach_data, 'even_strength_time_on_ice')
        self.pp_time_on_ice = self._select_data(coach_data, 'power_play_time_on_ice')
        self.pk_time_on_ice = self._select_data(coach_data, 'penalty_kill_time_on_ice')

        ## ABILITIES
        self.ev_corsi_offense = self._select_data(coach_data, 'even_strength_rapm_corsi_for_per_sixty')
        self.ev_corsi_defense = self._select_data(coach_data, 'even_strength_rapm_corsi_against_per_sixty')
        self.ev_fenwick_offense = self._select_data(coach_data, 'even_strength_rapm_fenwick_for_per_sixty')
        self.ev_fenwick_defense = self._select_data(coach_data, 'even_strength_rapm_fenwick_against_per_sixty')
        self.ev_xg_offense = self._select_data(coach_data, 'even_strength_rapm_expected_goals_for_per_sixty')
        self.ev_xg_defense = self._select_data(coach_data, 'even_strength_rapm_expected_goals_against_per_sixty')

        self.pp_corsi_offense = self._select_data(coach_data, 'power_play_rapm_corsi_for_per_sixty')
        self.pk_corsi_defense = self._select_data(coach_data, 'penalty_kill_rapm_corsi_against_per_sixty')
        self.pp_fenwick_offense = self._select_data(coach_data, 'power_play_rapm_fenwick_for_per_sixty')
        self.pk_fenwick_defense = self._select_data(coach_data, 'penalty_kill_rapm_fenwick_against_per_sixty')
        self.pp_xg_offense = self._select_data(coach_data, 'power_play_rapm_expected_goals_for_per_sixty')
        self.pk_xg_defense = self._select_data(coach_data, 'penalty_kill_rapm_expected_goals_against_per_sixty')

        self.ev_fenwick_offense_map = self._select_data(coach_data, 'even_strength_rapm_fenwick_for_per_sixty_map', None)
        self.ev_fenwick_defense_map = self._select_data(coach_data, 'even_strength_rapm_fenwick_against_per_sixty_map', None)
        self.pp_fenwick_offense_map = self._select_data(coach_data, 'power_play_rapm_fenwick_for_per_sixty_map', None)
        self.pk_fenwick_defense_map = self._select_data(coach_data, 'penalty_kill_rapm_fenwick_against_per_sixty_map', None)

        self.taken_pens = self._select_data(coach_data, 'rapm_penalty_minutes_for_per_sixty')
        self.drawn_pens = self._select_data(coach_data, 'rapm_penalty_minutes_against_per_sixty')
        for column in COACH_SCORE_COLUMNS:
            setattr(self, column, self._select_data(coach_data, column))
        for strength_state, prefix in COACH_SCORE_STRENGTH_PREFIXES.items():
            # RAPM can arrive either as one row per strength (raw model output)
            # or as one flattened roster row with explicit ev_/pp_/pk_ aliases.
            # Prefer the explicit aliases for the flattened form; otherwise the
            # generic columns would incorrectly be copied into every strength.
            has_strength_rows = 'strength_state' in coach_data.columns
            strength_data = coach_data
            if has_strength_rows:
                strength_data = strength_data.loc[strength_data['strength_state'].astype(str).eq(strength_state)]
            for column in COACH_SCORE_COLUMNS:
                alias = f'{prefix}_{column}'
                if has_strength_rows:
                    value = self._select_data(strength_data, column, None)
                    if value is None:
                        value = self._select_data(coach_data, alias)
                else:
                    value = self._select_data(coach_data, alias, None)
                    if value is None:
                        value = self._select_data(coach_data, column)
                setattr(self, alias, value)

class NHLTeam:
    def __init__(self, abbr, player_ids=None, data=None, name=None, lineup=None, coach_name=None):
        self.team_abbr = abbr
        self.team_name = name
        self.coach = NHLCoach(team_abbr=abbr, coach_name=coach_name, data=data)

        if player_ids and data is not None:
            self.roster = [NHLPlayer(p, data) for p in player_ids]
            self._populate_placeholder_identity()
            self._fill_missing()
            self._build_index()

            self.assign_lineup(lineup)

    # Allows us to reference players by their NHL API ID directly
    def _build_index(self):
        self._by_id = {p.player_id: p for p in self.roster}

    def __getitem__(self, key):
        return self._by_id[key]

    def get_roster_list(self):
        return self.roster
    
    # Fill in as many necessary roster spots with fake players as possible
    # These are (assumed) league-average players
    def _fake_name(self):
        if PLACEHOLDER_NAME_POOL:
            first, last = random.choice(PLACEHOLDER_NAME_POOL)
            return f"{first} {last}"
        first = ''.join(random.choices(string.ascii_uppercase, k=6))
        last = ''.join(random.choices(string.ascii_uppercase, k=8))
        return f"{first} {last}"

    def _is_placeholder_player(self, player):
        player_id = int(getattr(player, "player_id", 0) or 0)
        player_name = str(getattr(player, "player_name", "") or "").strip().upper()
        return (
            player_id < 0
            or player_name in {"", "PLAYER", "NOT FOUND", "NAN"}
            or (6000000 <= player_id < 8000000 and player_name in {"", "PLAYER", "NOT FOUND", "NAN"})
        )

    def _populate_placeholder_identity(self):
        existing_names = {
            str(getattr(player, "player_name", "") or "").strip().upper()
            for player in getattr(self, "roster", [])
            if str(getattr(player, "player_name", "") or "").strip()
        }

        for player in getattr(self, "roster", []):
            if not self._is_placeholder_player(player):
                continue
            fake_name = self._fake_name()
            while fake_name.upper() in existing_names:
                fake_name = self._fake_name()
            player.player_name = fake_name
            existing_names.add(fake_name.upper())

    def _fill_missing(self):
        target = {"F": 13, "D": 7, "G": 3}

        counts = {"F": 0, "D": 0, "G": 0}

        def map_pos(pos):
            if pos in ['C', 'L', 'R', 'F']:
                return "F"
            if pos == "D":
                return "D"
            if pos == "G":
                return "G"
            return None

        for p in self.roster:
            pos = map_pos(getattr(p, "position", None))
            if pos:
                counts[pos] += 1

        fake_id = 8500000

        for pos in ["F", "D", "G"]:
            missing = target[pos] - counts[pos]

            for idx in range(max(0, missing)):
                fake = type("FakePlayer", (), {})()
                fake.player_id = fake_id
                fake.player_name = self._fake_name()
                fake.position = pos
                fake.handedness = random.choice(["L", "R"])

                fake.time_on_ice = 0
                fake.ev_time_on_ice = 0
                fake.pp_time_on_ice = 0
                fake.pk_time_on_ice = 0

                self.roster.append(fake)
                fake_id += 1

    def _roster_table(self):
        return pd.DataFrame([{
            "player_id": p.player_id,
            "player_name": getattr(p, "player_name", None),
            "position": getattr(p, "position", None),
            "shoots": getattr(p, "handedness", None)
        } for p in self.roster])

    # Algorithm to determine player usage in the simulation
    def assign_lineup(self, lineup=None):
        if lineup:
            forwards = [self[p] for unit in lineup['EV_F'].values() for p in unit[:-1]]
            defensemen = [self[p] for unit in lineup['EV_D'].values() for p in unit[:-1]]
            goalies = [self[p] for unit in lineup['G'].values() for p in unit[:-1]]

            f_minutes = [round(unit[-1] * 50, 2) for unit in lineup['EV_F'].values()]
            d_minutes = [round(unit[-1] * 50, 2) for unit in lineup['EV_D'].values()]
            g_minutes = [round(unit[-1] * 60, 2) for unit in lineup['G'].values()]

            # Flatten PP/PK units directly
            pp_units = [
                [self[p] for p in lineup['PP'][1][:-1]],
                [self[p] for p in lineup['PP'][2][:-1]]
            ]
            pk_units = [
                [self[p] for p in lineup['PK'][1][:-1]],
                [self[p] for p in lineup['PK'][2][:-1]]
            ]

            pp_minutes = [lineup['PP'][1][-1] * 5, lineup['PP'][2][-1] * 5]
            pk_minutes = [lineup['PK'][1][-1] * 5, lineup['PK'][2][-1] * 5]

        else:
            forwards = [p for p in self.roster if p.position in ['C', 'L', 'R', 'F']]
            defensemen = [p for p in self.roster if p.position == "D"]
            goalies = [p for p in self.roster if p.position == "G"]

            forwards.sort(key=lambda x: x.time_on_ice, reverse=True)
            defensemen.sort(key=lambda x: x.time_on_ice, reverse=True)
            goalies.sort(key=lambda x: x.time_on_ice, reverse=True)

            forwards = forwards[:12]
            defensemen = defensemen[:6]
            # Keep both goalies available to the simulator.  The notebook or
            # caller may select the game starter; the first goalie remains the
            # default for backwards compatibility.
            goalies = goalies[:2]

            f_minutes = [19, 16, 14, 11]
            d_minutes = [25, 20, 15]
            g_minutes = [60, 60]

            # Build PP/PK units directly
            pp_pool = sorted(self.active_roster if hasattr(self, "active_roster") else forwards + defensemen,
                            key=lambda x: x.pp_time_on_ice, reverse=True)

            pk_pool = sorted(forwards + defensemen,
                            key=lambda x: x.pk_time_on_ice, reverse=True)

            pp_units = [pp_pool[:5], pp_pool[5:10]]
            pk_units = [pk_pool[:4], pk_pool[4:8]]

            pp_minutes = [0.12 * 60, 0.08 * 60]
            pk_minutes = [0.10 * 60, 0.08 * 60]

        # --- EVEN STRENGTH ASSIGNMENT ---
        for i, p in enumerate(forwards):
            line_idx = i // 3
            p.ev_time_on_ice = f_minutes[line_idx] if line_idx < len(f_minutes) else 8
            p.line = line_idx + 1

        for i, p in enumerate(defensemen):
            pair_idx = i // 2
            p.ev_time_on_ice = d_minutes[pair_idx] if pair_idx < len(d_minutes) else 10
            p.line = pair_idx + 1

        for i, p in enumerate(goalies):
            p.time_on_ice = g_minutes[i] if i < len(g_minutes) else 0

        # --- SPECIAL TEAMS ASSIGNMENT (UNIFIED) ---
        skaters = forwards + defensemen

        for p in skaters:
            p.pp_time_on_ice = 0
            p.pk_time_on_ice = 0

            for i, unit in enumerate(pp_units):
                if p in unit:
                    p.pp_time_on_ice = pp_minutes[i]

            for i, unit in enumerate(pk_units):
                if p in unit:
                    p.pk_time_on_ice = pk_minutes[i]

            p.time_on_ice = p.ev_time_on_ice + p.pp_time_on_ice + p.pk_time_on_ice

        for p in goalies:
            p.pp_time_on_ice = 0
            p.pk_time_on_ice = 0
            p.ev_time_on_ice = p.time_on_ice

        if self.coach is not None:
            self.coach.time_on_ice = g_minutes[0] if g_minutes else 0
            self.coach.pp_time_on_ice = 0
            self.coach.pk_time_on_ice = 0
            self.coach.ev_time_on_ice = self.coach.time_on_ice

        self.active_roster = forwards + defensemen + goalies
        self.starting_goalie = goalies[0] if goalies else None

    def __repr__(self):
        header = self.team_name if self.team_name else self.team_abbr
        table = self.roster_to_df().to_string(index=False)
        return f"\n{header}\nTeam Abbreviation: {self.team_abbr}\n\n{table}"

    def coach_to_df(self):
        if self.coach is None:
            return pd.DataFrame()

        row = {
            'player_id': None,
            'player_name': None,
            'coach_name': getattr(self.coach, 'coach_name', None),
            'position': 'coach',
            'team_abbr': self.team_abbr,
            'time_on_ice': getattr(self.coach, 'time_on_ice', None),
            'ev_time_on_ice': getattr(self.coach, 'ev_time_on_ice', None),
            'pp_time_on_ice': getattr(self.coach, 'pp_time_on_ice', None),
            'pk_time_on_ice': getattr(self.coach, 'pk_time_on_ice', None),
            'ev_corsi_offense': getattr(self.coach, 'ev_corsi_offense', None),
            'ev_corsi_defense': getattr(self.coach, 'ev_corsi_defense', None),
            'ev_fenwick_offense': getattr(self.coach, 'ev_fenwick_offense', None),
            'ev_fenwick_defense': getattr(self.coach, 'ev_fenwick_defense', None),
            'ev_xg_offense': getattr(self.coach, 'ev_xg_offense', None),
            'ev_xg_defense': getattr(self.coach, 'ev_xg_defense', None),
            'pp_corsi_offense': getattr(self.coach, 'pp_corsi_offense', None),
            'pp_fenwick_offense': getattr(self.coach, 'pp_fenwick_offense', None),
            'pp_xg_offense': getattr(self.coach, 'pp_xg_offense', None),
            'pk_corsi_defense': getattr(self.coach, 'pk_corsi_defense', None),
            'pk_fenwick_defense': getattr(self.coach, 'pk_fenwick_defense', None),
            'pk_xg_defense': getattr(self.coach, 'pk_xg_defense', None),
            'taken_pens': getattr(self.coach, 'taken_pens', None),
            'drawn_pens': getattr(self.coach, 'drawn_pens', None),
            'ev_fenwick_offense_map': getattr(self.coach, 'ev_fenwick_offense_map', None),
            'ev_fenwick_defense_map': getattr(self.coach, 'ev_fenwick_defense_map', None),
            'pp_fenwick_offense_map': getattr(self.coach, 'pp_fenwick_offense_map', None),
            'pk_fenwick_defense_map': getattr(self.coach, 'pk_fenwick_defense_map', None)
        }
        row.update({column: getattr(self.coach, column, None) for column in COACH_SCORE_COLUMNS})
        for prefix in COACH_SCORE_STRENGTH_PREFIXES.values():
            row.update({f'{prefix}_{column}': getattr(self.coach, f'{prefix}_{column}', None) for column in COACH_SCORE_COLUMNS})
        return pd.DataFrame([row])

    # Export player data in the team to a dataframe for reference and restoration

    def roster_to_df(self):
        rows = []

        for p in self.active_roster:
            rows.append({
                # IDENTITY
                "player_id":    p.player_id,
                "player_name":  p.player_name,
                "age":          getattr(p, "age",           None),
                "position":     getattr(p, "position",      None),
                "handedness":   getattr(p, "handedness",    None),
                "height_in":    getattr(p, "height_in",     None),
                "weight_lbs":   getattr(p, "weight_lbs",    None),
                "birth_country":getattr(p, "birth_country", None),
                "team_abbr": self.team_abbr,
                # USAGE
                "time_on_ice":      getattr(p, "time_on_ice",      None),
                "ev_time_on_ice":   getattr(p, "ev_time_on_ice",   None),
                "pp_time_on_ice":   getattr(p, "pp_time_on_ice",   None),
                "pk_time_on_ice":   getattr(p, "pk_time_on_ice",   None),
                # ABILITIES
                "ev_corsi_offense":   getattr(p, "ev_corsi_offense",  None),
                "ev_corsi_defense":   getattr(p, "ev_corsi_defense",  None),
                "ev_fenwick_offense":   getattr(p, "ev_fenwick_offense",  None),
                "ev_fenwick_defense":   getattr(p, "ev_fenwick_defense",  None),
                "ev_xg_offense":   getattr(p, "ev_xg_offense",  None),
                "ev_xg_defense":   getattr(p, "ev_xg_defense",  None),
                "pp_corsi_offense":   getattr(p, "pp_corsi_offense",  None),
                "pp_fenwick_offense":   getattr(p, "pp_fenwick_offense",  None),
                "pp_xg_offense":   getattr(p, "pp_xg_offense",  None),
                "pk_corsi_defense":   getattr(p, "pk_corsi_defense",  None),
                "pk_fenwick_defense":   getattr(p, "pk_fenwick_defense",  None),
                "pk_xg_defense":   getattr(p, "pk_xg_defense",  None),
                "finishing":    getattr(p, "finishing",   None),
                "induction":      getattr(p, "induction",     None),
                "secondary_induction": getattr(p, "secondary_induction", None),
                "finishing_impact": getattr(p, "finishing_impact", None),
                "induction_impact": getattr(p, "induction_impact", None),
                "secondary_induction_impact": getattr(p, "secondary_induction_impact", None),
                "taken_pens":   getattr(p, "taken_pens",  None),
                "drawn_pens":   getattr(p, "drawn_pens",  None),
                "goaltending":   getattr(p, "goaltending",  None),
                "goaltending_impact": getattr(p, "goaltending_impact", None),
                # TENDENCIES
                "shot_rate":    getattr(p, "shot_rate",   None),
                "fenwick_per_corsi": getattr(p, "fenwick_per_corsi", None),
                "shots_per_fenwick": getattr(p, "shots_per_fenwick", None),
                "xg_per_fenwick": getattr(p, "xg_per_fenwick", None),
                "primary_pass_rate": getattr(p, "primary_pass_rate", None),
                "secondary_pass_rate": getattr(p, "secondary_pass_rate", None),
                "block_rate": getattr(p, "block_rate", None),
                "penalties_taken_rate": getattr(p, "penalties_taken_rate", None),
                "penalties_drawn_rate": getattr(p, "penalties_drawn_rate", None),
                "giveaways_rate": getattr(p, "giveaways_rate", None),
                "takeaways_rate": getattr(p, "takeaways_rate", None),
                "hits_for_rate": getattr(p, "hits_for_rate", None),
                "hits_against_rate": getattr(p, "hits_against_rate", None),
                "faceoff_rate": getattr(p, "faceoff_rate", None),
                "faceoff_percentage": getattr(p, "faceoff_percentage", None),
                "shot_rate_against": getattr(p, "shot_rate_against", None),
                "ev_fenwick_offense_map": getattr(p, "ev_fenwick_offense_map", None),
                "ev_fenwick_defense_map": getattr(p, "ev_fenwick_defense_map", None),
                "pp_fenwick_offense_map": getattr(p, "pp_fenwick_offense_map", None),
                "pk_fenwick_defense_map": getattr(p, "pk_fenwick_defense_map", None)
            })

        df = pd.DataFrame(rows)
        coach_df = self.coach_to_df()
        if not coach_df.empty:
            df = pd.concat([df, coach_df], ignore_index=True, sort=False)

        return df

    # Load a roster from a dataframe
    # Assuming any dataframe has these columns with logical levels/values, none of the functions or classes developed so far are necessarily required
    def roster_from_df(self, df):
        roster = []

        # Rename the output columns back to the input columns to re-calculate the team components with
        df = df.rename(columns={
            'ev_time_on_ice': 'even_strength_time_on_ice',
            'pp_time_on_ice': 'power_play_time_on_ice',
            'pk_time_on_ice': 'penalty_kill_time_on_ice',
            'ev_corsi_offense': 'even_strength_rapm_corsi_for_per_sixty',
            'ev_corsi_defense': 'even_strength_rapm_corsi_against_per_sixty',
            'ev_fenwick_offense': 'even_strength_rapm_fenwick_for_per_sixty',
            'ev_fenwick_defense': 'even_strength_rapm_fenwick_against_per_sixty',
            'ev_xg_offense': 'even_strength_rapm_expected_goals_for_per_sixty',
            'ev_xg_defense': 'even_strength_rapm_expected_goals_against_per_sixty',
            'pp_corsi_offense': 'power_play_rapm_corsi_for_per_sixty',
            'pp_fenwick_offense': 'power_play_rapm_fenwick_for_per_sixty',
            'pp_xg_offense': 'power_play_rapm_expected_goals_for_per_sixty',
            'pk_corsi_defense': 'penalty_kill_rapm_corsi_against_per_sixty',
            'pk_fenwick_defense': 'penalty_kill_rapm_fenwick_against_per_sixty',
            'pk_xg_defense': 'penalty_kill_rapm_expected_goals_against_per_sixty',
            'taken_pens': 'rapm_penalty_minutes_for_per_sixty',
            'drawn_pens': 'rapm_penalty_minutes_against_per_sixty',
            "ev_fenwick_offense_map": "even_strength_rapm_fenwick_for_per_sixty_map",
            "ev_fenwick_defense_map": "even_strength_rapm_fenwick_against_per_sixty_map",
            "pp_fenwick_offense_map": "power_play_rapm_fenwick_for_per_sixty_map",
            "pk_fenwick_defense_map": "penalty_kill_rapm_fenwick_against_per_sixty_map",
            'finishing': 'rapm_goals_per_sixty',
            'induction': 'rapm_primary_assists_per_sixty',
            'secondary_induction': 'rapm_secondary_assists_per_sixty',
            'goaltending': 'rapm_goals_against_per_sixty',
            'shot_rate': 'corsi_per_sixty',
            'fenwick_per_corsi': 'fenwick_per_corsi',
            'shots_per_fenwick': 'shots_per_fenwick',
            'xg_per_fenwick': 'expected_goals_per_fenwick',
            'primary_pass_rate': 'primary_fenwick_assists_per_sixty',
            'secondary_pass_rate': 'secondary_fenwick_assists_per_sixty',
            'block_rate': 'blocked_shots_per_sixty',
            "penalties_taken_rate": 'minor_penalties_per_sixty',
            "penalties_drawn_rate": 'minor_penalties_drawn_per_sixty',
            "giveaways_rate": 'giveaways_per_sixty',
            "takeaways_rate": 'takeaways_per_sixty',
            "hits_for_rate": 'hits_for_per_sixty',
            "hits_against_rate": 'hits_against_per_sixty',
            "faceoff_rate": 'faceoffs_per_sixty',
            "faceoff_percentage": 'faceoff_percentage',
            "shot_rate_against": "fenwick_against_per_sixty",
            "offensive_zone_faceoffs": "offensive_zone_faceoffs_per_sixty",
            "defensive_zone_faceoffs": "defensive_zone_faceoffs_per_sixty",
            "neutral_zone_faceoffs": "neutral_zone_faceoffs_per_sixty"
        })
        
        self.coach = NHLCoach(team_abbr=self.team_abbr, data=df)

        player_df = df.copy()
        if 'position' in player_df.columns:
            player_df = player_df.loc[~player_df['position'].astype(str).str.lower().eq('coach')].copy()
        player_df = player_df.loc[player_df['player_id'].notna()].copy()

        for _, row in player_df.iterrows():
            player = NHLPlayer(row['player_id'], player_df)
            roster.append(player)

        self.roster = roster
        self._build_index()
        self.assign_lineup()

        return self
