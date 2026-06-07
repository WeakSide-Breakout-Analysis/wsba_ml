import numpy as np
import pandas as pd

def qoc_qot(pbp, stats):
    seasons = pbp['season'].drop_duplicates().to_list()
    print(f'QOC/QOT Estimates for {seasons} season...')

    away_cols = [f'away_on_{i}_id' for i in range(1, 7)]
    home_cols = [f'home_on_{i}_id' for i in range(1, 7)]

    # Convert columns to numpy arrays
    season_arr = pbp['season'].to_numpy()
    event_len_arr = pbp['event_length'].to_numpy()
    away_arr = pbp[away_cols].to_numpy()
    home_arr = pbp[home_cols].to_numpy()

    # Dunction to generate pairs efficiently
    def generate_pairs(arr1, arr2, interaction):
        n_events = arr1.shape[0]
        idx1, idx2 = np.meshgrid(np.arange(arr1.shape[1]), np.arange(arr2.shape[1]), indexing='ij')
        idx1 = idx1.flatten()
        idx2 = idx2.flatten()

        player = arr1[:, idx1].flatten()
        partner = arr2[:, idx2].flatten()
        seasons_expanded = np.repeat(season_arr, len(idx1))
        toi_expanded = np.repeat(event_len_arr, len(idx1))
        interaction_arr = np.repeat(interaction, len(player))

        mask = player != partner
        return pd.DataFrame({
            'player_id': player[mask],
            'partner_id': partner[mask],
            'season': seasons_expanded[mask],
            'interaction': interaction_arr[mask],
            'time_on_ice': toi_expanded[mask] / 60
        })

    # Teammate pairs (exclude diagonal)
    def teammate_pairs(arr):
        n = arr.shape[1]
        idx1, idx2 = np.triu_indices(n, k=1)
        player = arr[:, idx1].flatten()
        partner = arr[:, idx2].flatten()
        seasons_expanded = np.repeat(season_arr, len(idx1))
        toi_expanded = np.repeat(event_len_arr, len(idx1))
        interaction_arr = np.repeat('teammate', len(player))
        return pd.DataFrame({
            'player_id': player,
            'partner_id': partner,
            'season': seasons_expanded,
            'interaction': interaction_arr,
            'time_on_ice': toi_expanded / 60
        })

    edges = [
        teammate_pairs(away_arr),
        teammate_pairs(home_arr),
        generate_pairs(away_arr, home_arr, 'competition'),
        generate_pairs(home_arr, away_arr, 'competition')
    ]

    combos = pd.concat(edges, ignore_index=True)
    combos = combos.groupby(['player_id', 'partner_id', 'season', 'interaction'], as_index=False)['time_on_ice'].sum()
    combos = combos[combos['time_on_ice'] > 0]

    stats_renamed = stats[['player_id', 'season', 'goals_above_replacement']].rename(columns={'player_id': 'partner_id'})
    combos = combos.merge(stats_renamed, on=['partner_id', 'season'], how='left')
    combos = combos[combos['goals_above_replacement'].notna()]

    total_toi = combos.groupby(['player_id', 'season'])['time_on_ice'].transform('sum')
    combos['weighted_gar'] = combos['goals_above_replacement'] * (combos['time_on_ice'] / total_toi)

    agg = combos.groupby(['player_id', 'season', 'interaction'], as_index=False)['weighted_gar'].sum()
    qoc_qot_df = agg.pivot_table(index=['player_id', 'season'],
                                 columns='interaction',
                                 values='weighted_gar',
                                 fill_value=0).reset_index().rename(columns={
                                     'competition': 'quality_of_competition',
                                     'teammate': 'quality_of_teammates'
                                 })
    return qoc_qot_df
