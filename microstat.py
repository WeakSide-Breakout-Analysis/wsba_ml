import os
import pandas as pd
import datetime as dt
import wsba_hockey as wsba
from typing import Literal, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from wsba_hockey.tools.scraping import *
from wsba_hockey.tools.xg_model import *
from wsba_hockey.tools.agg import *
from wsba_hockey.tools.plotting import *
from wsba_hockey.tools.globals import *

### MICROSTAT FUNCTIONS ###
# Provided in this file are functions vital to parsing, cleaning, and analyzing data generated with this package in conjunction with hand-tracked microstat data from Corey Sznajder #
# Subscribe to Corey's patreon and obtain access to his raw data below #
# https://www.patreon.com/c/CSznajder/posts #

def nhl_scrape_a3z_pbp(a3z_path:str, simple = False, seasons:int | list[int] = []) -> pd.DataFrame:
    #Clean and create play-by-play dataset with handtracked data from Corey Sznajder
    a3z_map = COL_MAP['a3z_simple' if simple else 'a3z_pbp']

    if isinstance(seasons, int):
        seasons = [seasons]

    if simple:
        #Use the full game-by-game dataset (skims a lot of the cleaning and aggregating processes at the expense of blending our datasets together)
        a3z_file = f'{a3z_path}/a3z_simple.csv'

        a3z = pd.read_csv(a3z_file).rename(columns=a3z_map)[[col for col in a3z_map.values()]]

        #The simple stats are loaded in the game-by-game format ordinarily produced with our pbp parsing and calc_a3z functions
        #Most of the cleaning will be rearranging the data to match the stat conventions across the package and ensuring game_ids and player_ids are appended
        
        #Remove playoff stats (only exist for 2024-25) and goalies
        a3z = a3z.loc[(a3z['season']!='2024-25p')&(a3z['position']!='G')]

        #Convert season and team names as necessary and use same player_name convention (all uppercase)
        a3z['season'] = a3z['season'].replace({v: k for k, v in SEASON_NAMES.items()})
        a3z['team_abbr'] = a3z['team_abbr'].replace(CONVERT_TEAM_ABBR)

        #Any columns with the strength encoded are split and sprung out to a new row
        ev_df = a3z[[col for col in a3z.columns if col in FRONT_COL+['game_title', 'player_num'] or ('5v4' not in col and '4v5' not in col)]]
        pp_df = a3z[[col for col in a3z.columns if col in FRONT_COL+['game_title', 'player_num'] or '5v4' in col]]
        pk_df = a3z[[col for col in a3z.columns if col in FRONT_COL+['game_title', 'player_num'] or '4v5' in col]]

        ev_df = ev_df.rename(columns={col: col.replace('5v5_', '') for col in ev_df if '5v5' in col})
        pp_df = pp_df.rename(columns={col: col.replace('5v4_', '') for col in pp_df if '5v4' in col})
        pk_df = pk_df.rename(columns={col: col.replace('4v5_', '') for col in pk_df if '4v5' in col})

        ev_df['strength_state'] = '5v5'
        pp_df['strength_state'] = '5v4'
        pk_df['strength_state'] = '4v5'

        #Add back together
        df = pd.concat([ev_df, pp_df, pk_df])

        #Find the game id
        df['player_name'] = df['player_name'].str.upper()
        
        df[['game_date', 'team1', 'team2']] = df['game_title'].str.extract(r'(\d+/\d+/\d+)\s+([A-Za-z.]+)\s+vs\.?\s*([A-Za-z.]+)')
        df['game_date'] = pd.to_datetime(df['game_date'].str.strip())
        df['team1'] = df['team1'].str.strip().str.upper().replace(CONVERT_TEAM_ABBR | {'NJ': 'NJD', 'SJ': 'SJS', 'LA':'LAK', 'TB': 'TBL'})
        df['team2'] = df['team2'].str.strip().str.upper().replace(CONVERT_TEAM_ABBR | {'NJ': 'NJD', 'SJ': 'SJS', 'LA':'LAK', 'TB': 'TBL'})

        #Using the repository schedule dataframe
        schedule = pd.read_csv(SCHEDULE_PATH)[['game_date', 'away_team_abbr', 'home_team_abbr', 'game_id', 'season_type']]
        schedule['game_date'] = pd.to_datetime(schedule['game_date'].str.strip().str.replace('//', '/', regex=False))

        #Try both team orders
        df = pd.merge(
            df, 
            schedule, 
            left_on=['game_date', 'team1', 'team2'],
            right_on=['game_date', 'away_team_abbr', 'home_team_abbr'],
            how='left'
        )

        #For rows that didn't match, try reversed order
        unmatched = df['game_id'].isna()
        df.loc[unmatched] = df.loc[unmatched].drop(columns=['game_id', 'away_team_abbr', 'home_team_abbr']).merge(
            schedule,
            left_on=['game_date', 'team1', 'team2'],
            right_on=['game_date', 'home_team_abbr', 'away_team_abbr'],
            how='left'
        )

        #Additional stats
        df['entry_attempts'] = df['entries']+df['failed_entries']
        df['entry_attempts_against'] = df['entries_against']+df['disrupted_entries']
        df['exit_attempts'] = df['exits']+df['failed_exits']

        df['retrieval_attempts'] = df['retrievals']+df['botched_retrievals']

        df['controlled_entries'] = df['pass_entries']+df['carry_entries']
        df['controlled_entries_against'] = df['pass_entries_against']+df['carry_entries_against']
        df['controlled_exits'] = df['pass_exits']+df['carry_exits']
        
        df['dump_ins'] = df['entries']-df['carry_entries']
        df['dump_ins_against'] = df['entries_against']-df['carry_entries_against']
        
        df['passes'] = df['fenwick_assists']+df['pass_entries']+df['pass_exits']
        df['carries'] = df['carry_exits']+df['carry_entries']

        #Convert per-sixty stats to totals
        per_sixty_stats = [col for col in df.columns if 'per_sixty' in col]
        for col in per_sixty_stats:
            df[f'{col.replace('_per_sixty', '')}'] = df[col] * (df['time_on_ice']/60)
        
        df = df.drop(columns=per_sixty_stats)

        return df

    else:
        by_cols = ['entry_by', 'defended_by', 'dump_recovered_by', 'retrieval_by', 'pressured_by', 'exit_by']
        sort_col = ['game_id','period','period_seconds_remaining']

        dfs = []

        for season in seasons:            
            #Lets grab each game individually (should be safer for some future data analysis)
            a3z_folder = f'{a3z_path}/{season}/Game Sheets'

            season_games = []
            errors = []
            game_logs = os.listdir(a3z_folder)

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {}
                
                for game_log in game_logs:
                    game_id = int(str(season)[:4] + '0' + str(game_log[:5]))
                    
                    def process_game(gl=game_log, gid=game_id):
                        print(f'Loading game {gid}...')
                        try:
                            game_df = pd.read_excel(
                                os.path.join(a3z_folder, gl), 
                                'Tracking'
                            ).drop(columns=['Game ID'], errors='ignore')
                            
                            game_df['game_id'] = gid
                            game_df['season'] = season
                            game_df['season_type'] = int(gl[:1])
                            
                            return game_df.dropna(subset=['Time']).rename(columns=a3z_map), None
                        except:
                            print(f'No data found for game {gid}...')
                            return None, gid
                    
                    futures[executor.submit(process_game)] = game_log
                
                for future in as_completed(futures):
                    result, error = future.result()
                    if result is not None:
                        season_games.append(result)
                    if error is not None:
                        errors.append(error)

            print(f'Errors: {errors}')
            a3z_data = pd.concat(season_games)

            a3z_data_non_shot = a3z_data.loc[a3z_data['event_player_1_num'].isna()]
            a3z_data = a3z_data.loc[a3z_data['event_player_1_num'].notna()]

            a3z_data['event_player_1_num'] = a3z_data['event_player_1_num'].replace({
                '20-': 20
            })

            for key, col in a3z_map.items():
                if '?' in key:
                    a3z_data[col] = a3z_data[col].str.lower().str.strip().replace({'y': 1, 'n': 0}).fillna(0)

            a3z_data['is_goal'] = a3z_data['is_goal'].replace({
                'y': 1,
                'n': 0
            })

            try:    
                a3z_data['event_origin'] = a3z_data['event_origin'].str.upper()
            except KeyError:
                pass

            a3z_data['shot_category'] = a3z_data['shot_category'].astype(str).str.strip().str.lower().replace({
                'w': 'wrist/snap',
                'o': 'one-timer', 
                's': 'slap',
                'r': 'rebound',
                't': 'deflection',
                'a': 'wrap-around',
                'b': 'backhand'
            })

            a3z_data['event_type'] = np.where(
                a3z_data['is_goal']==1, 'goal',
                    np.where(a3z_data['is_shot']==1, 'shot-on-goal', 'missed-shot')
                )

            #Found out the hard way that a fallback is good here 
            a3z_data['is_fenwick'] = 1

            a3z_data['oddman'] = a3z_data['oddman'].str.lower().str.strip().str.replace(' on ', 'v')

            a3z_data_non_shot['event_type'] = 'entry-sequence'

            for col in ['pass', 'chance']:
                a3z_data_non_shot[col] = a3z_data_non_shot[col].str.lower().replace({'y':1, 'r':1, 'l':1, 'n':0})
                a3z_data_non_shot[col] = np.where(a3z_data_non_shot[col].isin([0,1]),
                                                a3z_data_non_shot[col],
                                                None)

            a3z_data_non_shot[by_cols] = a3z_data_non_shot[by_cols].replace('N', None)

            for col in by_cols:
                a3z_data_non_shot[[f'{col}_num', f'{col}_team_abbr']] = (
                    a3z_data_non_shot[col]
                    .astype(str)
                    .str.strip()
                    .str.extract(r'^(\d+)\s*([A-Za-z\.]+)$')
                )

                a3z_data_non_shot[f'{col}_num'] = (
                    a3z_data_non_shot[f'{col}_num']
                    .astype('Int64')
                )

            a3z_data_non_shot['is_fenwick'] = 0
            a3z_data_non_shot['event_player_1_num'] = pd.to_numeric(a3z_data_non_shot['event_player_1_num'], errors='coerce')

            a3z_pbp = pd.concat([a3z_data, a3z_data_non_shot])

            def mmssms_to_seconds(t):
                try:
                    t = str(t)[-8:]

                    t = t.translate({
                        '!': '1',
                        '@': '2',
                        '#': '3',
                        '$': '4',
                        '%': '5',
                        '^': '6',
                        '&': '7',
                        '*': '8',
                        '(': '9',
                        ')': '0'
                    })
                    
                    t = re.sub(r'[^0-9:]', '', t)
                    
                    if ':' in t:
                        parts = (t.replace('.', ':')
                                .replace(';', ':')
                                .split(':')
                                )
                        minutes = int(parts[0])
                        seconds = int(parts[1])
                        return minutes*60 + seconds
                except:
                    return None

            a3z_pbp['period_seconds_remaining'] = a3z_pbp['period_time_remaining'].apply(mmssms_to_seconds)

            a3z_pbp = a3z_pbp[[col for col in a3z_pbp.columns if not any(c.isupper() for c in col)]]
                    
            #Plenty to clean up after before the data is "useable"
            a3z_pbp = a3z_pbp.replace([np.nan, 'nan', pd.NA], None)

            a3z_pbp['period'] = a3z_pbp['period'].replace({
                dt.datetime(1900, 1, (i+1), 0, 0): (i+1) for i in range(10)
            } | {
                dt.time(0, (i+1)): (i+1) for i in range(10)
            } | {
                dt.time(0, 0): None,
                'c': None,
                'F': None,
                'L': None,
            })
            
            #If a period value is somehow not sanitized then just remove it
            a3z_pbp['period'] = (pd.to_numeric(a3z_pbp['period'], errors='coerce')
                .where(lambda s: s.isna() | (s % 1 == 0))
                .astype('Int64')
                .ffill()
            )

            a3z_pbp['period_seconds_remaining'] = a3z_pbp['period_seconds_remaining'].astype('Int64')

            try:    
                a3z_pbp['event_origin'] = a3z_pbp['event_origin'].str.strip().replace({
                    'F': None,
                    'R': None
                })
            except KeyError:
                #This doesn't exist for some older a3z datasets
                a3z_pbp['event_origin'] = None

            try:
                a3z_pbp['event_context'] = a3z_pbp['event_context'].str.strip().str.lower().replace(CONTEXT_FIXES).str.lower().str.replace(' ', '-')
            except KeyError:
                #Same here
                a3z_pbp['event_context'] = None

            a3z_pbp['strength_state'] = a3z_pbp['strength_state'].str.lower().replace(A3Z_STRENGTH_FIXES)

            non_cols = a3z_pbp.columns.difference(['strength_state'])
            for col in non_cols:
                a3z_pbp.loc[a3z_pbp[col]=='5v5', col] = None

            a3z_pbp[[col for col in a3z_pbp if 'team_abbr' in col]] = a3z_pbp[[col for col in a3z_pbp if 'team_abbr' in col]].replace(A3Z_TEAM_FIXES).replace(CONVERT_TEAM_ABBR)

            a3z_pbp['retrieval_result'] = a3z_pbp['retrieval_result'].str.strip().str.upper()
            a3z_pbp['exit_result'] = a3z_pbp['exit_result'].str.strip().str.upper()

            a3z_pbp[['retrieval_result', 'exit_result']] = a3z_pbp[['retrieval_result', 'exit_result']].replace(RESULT_FIXES).replace(RESULT_NAMES)
            
            a3z_pbp['entry_type'] = a3z_pbp['entry_type'].str.strip().str.lower().replace({
                '': None,
                '*': None,
                'l': None,
                '3c':'c'
            })
            a3z_pbp['entry_type'] = a3z_pbp['entry_type'].replace(ENTRY_TYPES)
            
            a3z_pbp['rush_attempt'] = a3z_pbp['rush_attempt'].str.lower().str.strip()
            a3z_pbp['lane'] = a3z_pbp['lane'].str.strip().str.upper().replace(LANE_FIXES)

            #utility_get_unique(a3z_pbp).to_csv(f'{season}_a3z_unique.csv', index=False)
            
            dfs.append(a3z_pbp)

        wsba_x_a3z = pd.concat(dfs).reset_index().sort_values(by=sort_col, ascending=[True, True, False])

        wsba.utility_get_unique(wsba_x_a3z).to_csv(f'{season}_a3z_unique.csv',index=False)

        num_cols = wsba_x_a3z.select_dtypes(include='number').columns
        wsba_x_a3z[num_cols] = wsba_x_a3z[num_cols].astype('Int64')

        return wsba_x_a3z

def nhl_join_plays(pbp:pd.DataFrame, a3z:pd.DataFrame) -> pd.DataFrame:
    #Join NHL pbp data to allthreezones pbp data
    
    def seconds_remaining(row):
        # convert elapsed MM:SS to seconds
        if row['season_type'] == 2 and row['period'] == 4:
            period_len = 5 * 60  # 5 min OT
        else:
            period_len = 20 * 60  # 20 min regular

        # seconds elapsed **within this period**
        period_start = (row['period'] - 1) * 20 * 60
        if row['season_type'] == 2 and row['period'] == 4:
            period_start = 3 * 20 * 60  # 4th period in OT
        elapsed_in_period = row['seconds_elapsed'] - period_start

        # remaining seconds
        remaining = period_len - elapsed_in_period
        return max(remaining, 0)
    
    pbp['period_seconds_remaining'] = pbp.apply(seconds_remaining, axis=1)

    #Remove events highly unlikely to be included in a3z
    pbp = pbp.loc[~pbp['event_reason'].isin(['short'])]

    merge_col = [col for col in ['game_id', 'period', 'event_type','strength_state','event_player_1_id','event_team_abbr', 'event_goalie_id'] if col in pbp.columns and col in a3z.columns]
    merge_sort = ['period_seconds_remaining']+merge_col

    for col in merge_col+['period_seconds_remaining']:
        try:
            pbp[col] = pbp[col].astype("Int64")
            a3z[col] = a3z[col].astype("Int64")
        except:
            continue

    pbp = pbp.drop(columns=['season_type', 'season'])
    a3z = a3z.drop(columns=['event_index','index','is_goal'],errors='ignore')

    pbp = pbp.loc[pbp['event_type'].isin(FENWICK_EVENTS)].dropna(subset=merge_col+['period_seconds_remaining'])\
        .sort_values(merge_sort)
    
    a3z_non_shot = a3z.loc[a3z['is_fenwick']!=1] #Hear me out...
    a3z = a3z.loc[a3z['is_fenwick']==1]\
        .dropna(subset=merge_col+['period_seconds_remaining'])\
        .sort_values(merge_sort)

    #Primary join
    data = pd.merge_asof(a3z, pbp,
        on='period_seconds_remaining',
        by=merge_col,
        direction='nearest',
        tolerance=5
    )

    data = pd.concat([data, a3z_non_shot]).sort_values(by=['game_id','period','seconds_elapsed'])

    return data

def calc_a3z(pbp,game_strength,second_group,roster):
    #Calculate stats with microstat data
    pbp = pbp.loc[(pbp['event_type'].isin(FENWICK_EVENTS+['entry-sequence']))]
    
    num_cols = [col for col in pbp.columns if 'num' in col]+['season', 'game_id']
    for col in num_cols:
        pbp[col] = pd.to_numeric(pbp[col], errors='coerce').astype('Int64')
    
    #Now for the fun stuff

    # Filter by game strength if not "all"
    if game_strength != "all":
        pbp = pbp.loc[pbp['strength_state'].isin(game_strength)]
    
    #Add second event-team column for necessary situations
    pbp['event_team_abbr_2'] = np.where(pbp['event_team_abbr'].notna(),
        np.where(pbp['event_team_abbr']==pbp['home_team_abbr'],pbp['away_team_abbr'],pbp['home_team_abbr']),np.nan)

    #Change second event team to goal-scoring team for goal events
    pbp['event_team_abbr_2'] = np.where(pbp['event_type']=='goal',pbp['event_team_abbr'],pbp['event_team_abbr_2'])
    
    pbp['screen_fenwick'] = (pbp['screen']==1).astype(int)
    pbp['rush_fenwick'] = (pbp['rush_attempt']=='r').astype(int)
    pbp['forecheck_fenwick'] = (pbp['rush_attempt']=='f').astype(int)
    pbp['cycle_fenwick'] = (pbp['rush_attempt']=='c').astype(int)
    pbp['screen_goal'] = ((pbp['screen']==1)*(pbp['event_type']=='goal')).astype(int)
    pbp['rush_goal'] = ((pbp['rush_attempt']=='r')*(pbp['event_type']=='goal')).astype(int)
    pbp['forecheck_goal'] = ((pbp['rush_attempt']=='f')*(pbp['event_type']=='goal')).astype(int)
    pbp['cycle_goal'] = ((pbp['rush_attempt']=='c')*(pbp['event_type']=='goal')).astype(int)
    pbp['screen_xG'] = np.where(pbp['screen']==1, pbp['xG'], 0)
    pbp['rush_xG'] = np.where(pbp['rush_attempt']=='r', pbp['xG'], 0)
    pbp['forecheck_xG'] = np.where(pbp['rush_attempt']=='f', pbp['xG'], 0)
    pbp['cycle_xG'] = np.where(pbp['rush_attempt']=='c', pbp['xG'], 0)
    pbp['screen_scoring_chance'] = ((pbp['screen']==1)*(pbp['scoring_chance']==1)).astype(int)
    pbp['rush_scoring_chance'] = ((pbp['rush_attempt']=='r')*(pbp['scoring_chance']==1)).astype(int)
    pbp['forecheck_scoring_chance'] = ((pbp['rush_attempt']=='f')*(pbp['scoring_chance']==1)).astype(int)
    pbp['cycle_scoring_chance'] = ((pbp['rush_attempt']=='c')*(pbp['scoring_chance']==1)).astype(int)
    
    pbp['forecheck_recovery'] = (pbp['entry_by_team_abbr']==pbp['dump_recovered_by_team_abbr']).astype(int)
    pbp['disrupted_exit'] = (pbp['exit_result'].isin(['missed-pass', 'failed-exit'])).astype(int)
    
    event_contexts = ['nz-regroup','nz-turnover','dz-counter','oz-faceoff','dz-exit','nz-faceoff','nz-reload','controlled-breakout','broken-play','sling-shot','nz-counter','scramble','nz-recovery','chip','oddman','lead-in']

    for context in event_contexts:
        pbp[f'{context}_expected_goals'] = np.where(pbp['event_context']==context, pbp['xG'], 0)

    agg_pbp = pbp.loc[(pbp['event_type'].isin(["goal", "shot-on-goal", "missed-shot"]))&(pbp['xG'].notna())]
    agg_pbp_2 = pbp.loc[pbp['event_type']=='entry-sequence']

    complete = (
        agg_pbp.groupby(by=[f'event_player_1_id', 'event_team_abbr']+second_group).agg(
            **{f'{context}_expected_goals': (f'{context}_expected_goals', 'sum') for context in event_contexts},
            screen_fenwick=('screen_fenwick', 'sum'),
            rush_fenwick=('rush_fenwick', 'sum'),
            forecheck_fenwick=('forecheck_fenwick', 'sum'),
            cycle_fenwick=('cycle_fenwick', 'sum'),

            screen_goals=('screen_goal', 'sum'),
            rush_goals=('rush_goal', 'sum'),
            forecheck_goals=('forecheck_goal', 'sum'),
            cycle_goals=('cycle_goal', 'sum'),

            screen_scoring_chances = ('screen_scoring_chance', 'sum'),
            rush_scoring_chances = ('rush_scoring_chance', 'sum'),
            forecheck_scoring_chances = ('forecheck_scoring_chance', 'sum'),
            cycle_scoring_chances = ('cycle_scoring_chance', 'sum'),

            screen_expected_goals=('screen_xG', 'sum'),
            rush_expected_goals=('rush_xG', 'sum'),
            forecheck_expected_goals=('forecheck_xG', 'sum'),
            cycle_expected_goals=('cycle_xG', 'sum')
        )
    ).reset_index().rename(columns={'event_player_1_id': 'player_id', 'event_team_abbr': 'team_abbr'})
    
    #Determine how to group
    iter_title = {1: 'primary', 2: 'secondary', 3: 'tertiary'}
    for i in range(3):
        data = (
            agg_pbp.groupby(by=[f'shot_assist_player_{i+1}_num', 'event_team_abbr']+second_group).agg(
                **{f'{context}_expected_assists': (f'{context}_expected_goals', 'sum') for context in event_contexts},
                fenwick_assists = ('event_type', 'count'),
                shot_assists = ('is_shot', 'sum'),
                scoring_chance_assists = ('scoring_chance', 'sum'),
                expected_assists = ('xG', 'sum'),

                screen_fenwick_assists = ('screen_fenwick', 'sum'),
                rush_fenwick_assists = ('rush_fenwick', 'sum'),
                forecheck_fenwick_assists = ('forecheck_fenwick', 'sum'),
                cycle_fenwick_assists = ('cycle_fenwick', 'sum'),
                
                screen_expected_assists = ('screen_xG', 'sum'),
                rush_expected_assists = ('rush_xG', 'sum'),
                forecheck_expected_assists = ('forecheck_xG', 'sum'),
                cycle_expected_assists = ('cycle_xG', 'sum'),

                screen_scoring_chance_assists = ('screen_scoring_chance', 'sum'),
                rush_scoring_chance_assists = ('rush_scoring_chance', 'sum'),
                forecheck_scoring_chance_assists = ('forecheck_scoring_chance', 'sum'),
                cycle_scoring_chance_assists = ('cycle_scoring_chance', 'sum'),

                screen_assists=('screen_goal', 'sum'),
                rush_assists=('rush_goal', 'sum'),
                forecheck_assists=('forecheck_goal', 'sum'),
                cycle_assists=('cycle_goal', 'sum'),
            )
        ).reset_index()
        
        data = data.rename(columns=({col: f'{iter_title[i+1]}_{col}' for col in data.columns if 'assists' in col} | {f'shot_assist_player_{i+1}_num': 'player_num', 'event_team_abbr': 'team_abbr'}))
        
        if i > 0:
            df = pd.merge(df, data, how='left')
        else:
            df = data

    #Non-shot microstats
    entry_offense = (
        agg_pbp_2.groupby(by=['entry_by_num', 'entry_by_team_abbr', 'game_id']).agg(
            zone_entries=('entry_by_num', 'size'),
            carries=('entry_type', lambda x: (x == 'carry-in').sum()),
            dump_ins=('entry_type', lambda x: (x == 'dump-in').sum()),
            failed_entries=('entry_type', lambda x: (x == 'failed-entry').sum()),
            entry_passes=('pass', 'sum'),
            entry_chances=('chance', 'sum'),
            forechecking_recoveries=('forecheck_recovery', 'sum')
        )
    ).reset_index().rename(columns=({'entry_by_num': 'player_num', 'entry_by_team_abbr': 'team_abbr'}))

    dump_recovery = (
        agg_pbp_2.groupby(by=['dump_recovered_by_num', 'dump_recovered_by_team_abbr', 'game_id']).agg(
            dump_ins_recovered=('entry_type', lambda x: (x == 'dump-in').sum())
        )
    ).reset_index().rename(columns=({'dump_recovered_by_num': 'player_num', 'dump_recovered_by_team_abbr': 'team_abbr'}))

    entry_defense = (
        agg_pbp_2.groupby(by=['defended_by_num', 'defended_by_team_abbr', 'game_id']).agg(
            zone_entries_against=('defended_by_num', 'size'),
            carries_against=('entry_type', lambda x: (x == 'carry-in').sum()),
            dump_ins_against=('entry_type', lambda x: (x == 'dump-in').sum()),
            disrupted_entries=('entry_type', lambda x: (x == 'failed-entry').sum()),
            entry_passes_against=('pass', 'sum'),
            entry_chances_against=('chance', 'sum')
        )
    ).reset_index().rename(columns=({'defended_by_num': 'player_num', 'defended_by_team_abbr': 'team_abbr'}))

    by_events = list(RESULT_NAMES.values())

    retrieval = (
        agg_pbp_2.groupby(by=['retrieval_by_num', 'retrieval_by_team_abbr', 'game_id']).agg(
            retrievals=('retrieval_by_num', 'size'),
            **{f'retrievals_leading_to_{result}': ('retrieval_result', lambda x, r=result: (x == r).sum()) for result in by_events}
        )
    ).reset_index().rename(columns=({'retrieval_by_num': 'player_num', 'retrieval_by_team_abbr': 'team_abbr'}))

    forechecking = (
        agg_pbp_2.groupby(by=['pressured_by_num', 'pressured_by_team_abbr', 'game_id']).agg(
            forechecking_pressures=('pressured_by_num', 'size'),
            disrupted_exits=('disrupted_exit', 'sum')
        )
    ).reset_index().rename(columns=({'pressured_by_num': 'player_num', 'pressured_by_team_abbr': 'team_abbr'}))
    
    zone_exit = (
        agg_pbp_2.groupby(by=['exit_by_num', 'exit_by_team_abbr', 'game_id']).agg(
            zone_exits=('exit_by_num', 'size'),
            **{f'zone_exits_leading_to_{result}': ('exit_result', lambda x, r=result: (x == r).sum()) for result in by_events}
        )
    ).reset_index().rename(columns=({'exit_by_num': 'player_num', 'exit_by_team_abbr': 'team_abbr'}))

    for non_shot in [entry_offense, dump_recovery, entry_defense, retrieval, forechecking, zone_exit]:
        df = pd.merge(df, non_shot, how='left')

    #Add roster
    df = pd.merge(
        df, roster[['game_id', 'sweater_number', 'player_id', 'position']],
        how='left',
        left_on=['game_id', 'player_num'],
        right_on=['game_id', 'sweater_number']
    ).drop(columns=['sweater_number'])

    complete = pd.merge(df, complete, how='left')

    #Field calculations/percentage stats
    for sit in ['forecheck', 'cycle', 'rush']:
        for stat in ['expected', 'goal', 'fenwick', 'scoring_chance']:
            if stat == 'goal':
                complete[f'{sit}_assists'] = (
                    complete[f'primary_{sit}_assists']
                    + complete[f'secondary_{sit}_assists']
                )
            else:
                complete[f'{sit}_{stat}_assists'] = (
                    complete[f'primary_{sit}_{stat}_assists']
                    + complete[f'secondary_{sit}_{stat}_assists']
                )

            if stat not in ['expected', 'goal']:
                complete[f'{sit}_{stat}_contributions'] = complete[f'{sit}_{stat}{'s' if stat == 'scoring_chance' else ''}']+complete[f'{sit}_{stat}_assists']

        complete[f'{sit}_points'] = complete[f'{sit}_goals']+complete[f'{sit}_assists']
        complete[f'{sit}_expected_points'] = complete[f'{sit}_expected_goals']+complete[f'{sit}_expected_assists']

    complete['fenwick_assists'] = (
        complete['primary_fenwick_assists']
        + complete['secondary_fenwick_assists']
    )

    cats = ['screen', 'rush', 'forecheck', 'cycle']

    complete['goals'] = complete[[f'{cat}_goals' for cat in cats]].fillna(0).sum(axis=1)
    complete['expected_goals'] = complete[[f'{cat}_expected_goals' for cat in cats]].fillna(0).sum(axis=1)
    complete['fenwick'] = complete[[f'{cat}_fenwick' for cat in cats]].fillna(0).sum(axis=1)
    complete['assists'] = complete[[f'{cat}_assists' for cat in cats]].fillna(0).sum(axis=1)

    complete['in_zone_fenwick_contributions'] = complete['forecheck_fenwick_contributions']+complete['cycle_fenwick_contributions']
    complete['in_zone_scoring_chance_contributions'] = complete['forecheck_scoring_chance_contributions']+complete['cycle_scoring_chance_contributions']
    
    complete['in_zone_expected_goals'] = complete['forecheck_expected_goals']+complete['cycle_expected_goals']
    complete['in_zone_expected_assists'] = complete['forecheck_expected_assists']+complete['cycle_expected_assists']
    complete['in_zone_expected_points'] = complete['in_zone_expected_goals']+complete['in_zone_expected_assists']

    #Add back in season
    complete['season_year'] = complete['game_id'].astype(str).str[:4].astype(int)
    complete['season'] = complete['season_year'].astype(str) + (complete['season_year']+1).astype(str)

    return complete.replace([np.inf, -np.inf], 0)

def nhl_calculate_microstats(
        pbp:pd.DataFrame, 
        game_strength:Union[Literal['all'], str, list[str]] = 'all', 
        season_types:int | list[int] = [2,3],
        roster_path:str = 'rosters'
    ):

    #Calculate stats from pbp generated with _nhl_join_plays
    
    #Find the seasons covered in the play-by-play data
    seasons = pbp['season'].drop_duplicates().dropna().astype(int)
    
    print(f'''Calculating micro-statistics for {'regular season' if season_types == 2 else
                                            'playoff' if season_types == 3 else
                                            'regular season and playoff' if season_types == [2,3] else
                                            'unknown selection of'} games in the provided play-by-play data at {game_strength}...\nSeasons included: {seasons.to_list()}...'''
    )
    start = time.perf_counter()

    #If single values provided for columns typically in a list then place them into a list
    if isinstance(season_types, int):
        season_types = [season_types]
    if isinstance(game_strength, str) and game_strength != 'all':
        game_strength = [game_strength]

    #Apply season_type filter, remove shootouts, and remove invalid strengths
    pbp = pbp.loc[(pbp['season_type'].isin(season_types))&(pbp['period_type']!='SO')&(pbp['strength_state'].isin(STRENGTHS))&(~pbp['event_reason'].isin(['short', 'failed-bank-attempt']))]

    #Convert all columns with player ids to float in order to avoid merging errors
    id_cols = [col for col in pbp.columns if '_id' in col]
    pbp[id_cols] = pbp[id_cols].apply(pd.to_numeric)

    second_group = ['season', 'game_id']

    #Load game-by-game rosters to bind player numbers (primary player indentifier in a3z data) to NHL api player id
    roster = pd.concat([pd.read_parquet(f'{roster_path}/nhl_rosters_{season}.parquet') for season in seasons])
    complete = calc_a3z(pbp, game_strength, second_group, roster)

    #Remove entries with no ID listed
    if 'player_id' in complete.columns:
        complete = complete.loc[complete['player_id'].notna()]
    
    #Add strength and season type columns to the end of the df
    complete['strength_state'] = game_strength if isinstance(game_strength, str) else ', '.join(game_strength)
    complete['season_type'] = 'all' if season_types == [2,3] else season_types if isinstance(season_types, int) else ', '.join([str(s) for s in season_types])
    
    end = time.perf_counter()
    length = end-start
    print(f'...finished in {(length if length <60 else length/60):.2f} {'seconds' if length <60 else 'minutes'}.')

    #Return: game-by-game microstat data
    #Use nhl_agg_stats in the main module to condense as desired
    return complete
