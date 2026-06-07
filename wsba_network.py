import itertools
import pandas as pd
import numpy as np
import networkx as nx

NETWORK_WEIGHTS = {
    'primary': 0.75,
    'secondary': 0.5,
    'tertiary': 0.25
}

def nhl_create_network_edges(pbp, weights=NETWORK_WEIGHTS, net=False):
    # Conduct a network analysis on projected expected assists for fenwick shots
    seasons = pbp['season'].drop_duplicates().to_list()
    print(f'Network analysis on projected expected assists from pbp with seasons: {seasons}...')

    pbp = pbp.loc[pbp['xG'] > 0].copy()
    prob_col = 'fenwick_assist_probability'

    # Orient on-ice IDs and probabilities
    for i in range(1, 7):
        pbp[f'event_on_{i}_id'] = np.where(
            pbp['event_team_venue']=='away',
            pbp[f'away_on_{i}_id'],
            pbp[f'home_on_{i}_id']
        )
        for role in ['primary','secondary','tertiary']:
            pbp[f'event_on_{i}_{role}_{prob_col}'] = np.where(
                pbp['event_team_venue']=='away',
                pbp[f'away_on_{i}_{role}_{prob_col}'],
                pbp[f'home_on_{i}_{role}_{prob_col}']
            )

    edges = []
    context_cols = ['game_id','season','strength_state','event_team_abbr','period','seconds_elapsed','event_type','xG']

    # Iterate row by row
    for i in range(1, 6):  # max chain length 6
        # Tertiary -> Secondary
        t_s = pbp[context_cols].copy()
        t_s['source_player_id'] = pbp[f'event_on_{i}_id']
        t_s['target_player_id'] = pbp[f'event_on_{i+1}_id']
        t_s['source_event_probability'] = pbp[f'event_on_{i}_tertiary_{prob_col}']
        t_s['target_event_probability'] = pbp[f'event_on_{i+1}_secondary_{prob_col}']
        t_s['weight'] = t_s['xG'] * t_s['source_event_probability'] * t_s['target_event_probability'] * weights['tertiary'] * weights['secondary']
        t_s['source_player_event'] = 'tertiary_fenwick_assist'
        t_s['target_player_event'] = 'secondary_fenwick_assist'
        edges.append(t_s)

        # Secondary -> Primary
        s_p = pbp[context_cols].copy()
        s_p['source_player_id'] = pbp[f'event_on_{i}_id']
        s_p['target_player_id'] = pbp[f'event_on_{i+1}_id']
        s_p['source_event_probability'] = pbp[f'event_on_{i}_secondary_{prob_col}']
        s_p['target_event_probability'] = pbp[f'event_on_{i+1}_primary_{prob_col}']
        s_p['weight'] = s_p['xG'] * s_p['source_event_probability'] * s_p['target_event_probability'] * weights['secondary'] * weights['primary']
        s_p['source_player_event'] = 'secondary_fenwick_assist'
        s_p['target_player_event'] = 'primary_fenwick_assist'
        edges.append(s_p)

        # Primary -> Shooter
        p_f = pbp[context_cols].copy()
        p_f['source_player_id'] = pbp[f'event_on_{i}_id']
        p_f['target_player_id'] = pbp['event_player_1_id']
        p_f['source_event_probability'] = pbp[f'event_on_{i}_primary_{prob_col}']
        p_f['target_event_probability'] = 1.0
        p_f['weight'] = p_f['xG'] * p_f['source_event_probability'] * weights['primary']
        p_f['source_player_event'] = 'primary_fenwick_assist'
        p_f['target_player_event'] = 'fenwick'
        edges.append(p_f)

    if net:
        # Shooter -> net edges keep shot volume/xG visible as the terminal node.
        shot_net = pbp[context_cols].copy()
        shot_net['source_player_id'] = pbp['event_player_1_id']
        shot_net['target_player_id'] = 'net'
        shot_net['source_event_probability'] = 1.0
        shot_net['target_event_probability'] = 1.0
        shot_net['weight'] = shot_net['xG']
        shot_net['source_player_event'] = 'fenwick'
        shot_net['target_player_event'] = 'net'
        edges.append(shot_net)

    network_edges = pd.concat(edges, ignore_index=True)
    network_edges = network_edges.dropna(subset=['source_player_id', 'target_player_id'])
    network_edges = network_edges.loc[network_edges['weight'] > 0]

    return network_edges

def nhl_create_defense_network(pbp):
    #Construct a network based on on-ice fenwick against
    seasons = pbp['season'].drop_duplicates().to_list()
    print(f'Network analysis on fenwick shots against from pbp with seasons: {seasons}...')

    pbp = pbp.loc[pbp['xG'] > 0].copy()

    #Re-calibrate time from last event
    pbp['seconds_since_last'] = pbp.groupby('game_id')['seconds_elapsed'].diff()

    avg_gap = pbp['seconds_since_last'].mean()

    # Fill first shot of each game with average gap
    pbp['seconds_since_last'] = pbp['seconds_since_last'].fillna(avg_gap)

    for i in range(1, 7):
        pbp[f'event_on_{i}_id'] = np.where(
            pbp['event_team_venue'] == 'away',
            pbp[f'home_on_{i}_id'],
            pbp[f'away_on_{i}_id']
        )
    
    pbp['event_team_abbr'] = np.where(pbp['event_team_abbr']==pbp['home_team_abbr'], pbp['away_team_abbr'], pbp['home_team_abbr'])

    # Calculate the number of valid players on ice for each shot
    def_cols = [f'event_on_{i}_id' for i in range(1, 7)]
    pbp['skater_count'] = pbp[def_cols].notna().sum(axis=1)
    
    # Calculate unique pairs: n * (n - 1) / 2
    pbp['num_pairs'] = (pbp['skater_count'] * (pbp['skater_count'] - 1)) / 2
    
    # Distribute xG across the pairs
    pbp['weight'] = pbp['xG'] / pbp['num_pairs']

    edges = []
    context_cols = ['game_id', 'season', 'strength_state', 'event_team_abbr', 'period', 'seconds_elapsed', 'event_type', 'xG', 'weight']

    pair_indices = list(itertools.combinations(range(len(def_cols)), 2))

    # Iterate over each play
    for i, j in pair_indices:
        # Get all on-ice player IDs
        pair_df = pbp[context_cols]
        pair_df['source_player_id'] = pbp[def_cols[i]]
        pair_df['target_player_id'] = pbp[def_cols[j]]
        edges.append(pair_df)

    defense_network_edges = pd.concat(edges).dropna(subset=['source_player_id', 'target_player_id'])

    #Unique combinations only
    mask = defense_network_edges['source_player_id'] > defense_network_edges['target_player_id']
    defense_network_edges.loc[mask, ['source_player_id', 'target_player_id']] = defense_network_edges.loc[mask, ['target_player_id', 'source_player_id']].values

    defense_network_edges = defense_network_edges.groupby(
        ['game_id', 'season', 'strength_state', 'seconds_elapsed', 'event_team_abbr','source_player_id', 'target_player_id'], 
        as_index=False
    ).agg({'weight': 'first'})

    return defense_network_edges

def nhl_calculate_network(edges, network='for'):
    seasons = edges['season'].drop_duplicates().to_list()
    print(f'Calculating centralities for {"offensive" if network == "for" else "defensive"} network edges in season(s): {seasons}...')
    
    results = []

    group_cols = ['game_id', 'season', 'event_team_abbr', 'strength_state']
    for (game_id, season, team, strength), group in edges.groupby(group_cols):
        # Aggregate duplicate edges (sum weight and recompute distance)
        edge_agg = group.groupby(['source_player_id','target_player_id'], as_index=False).agg(
            weight=('weight','sum'),
        )
        edge_agg['distance'] = 1 / edge_agg['weight']

        # Create graph
        G = nx.from_pandas_edgelist(
            edge_agg,
            'source_player_id',
            'target_player_id',
            edge_attr=['weight','distance'],
            create_using=nx.DiGraph() if network=='for' else nx.Graph()
        )

        nodes = list(G.nodes())

        # Calculate centralities (optimized for disconnected graphs)
        degree = dict(G.degree(weight='weight'))
        in_degree = dict(G.in_degree(weight='weight')) if G.is_directed() else {}
        out_degree = dict(G.out_degree(weight='weight')) if G.is_directed() else {}
        betweenness = nx.betweenness_centrality(G, weight='distance')
        harmonic = nx.harmonic_centrality(G, distance='distance')
        pagerank = nx.pagerank(G, weight='weight')

        suffix = '_against' if network=='against' else ''

        # Bulk append
        results.extend([{
            'game_id': game_id,
            'season': season,
            'team_abbr': team,
            'strength_state': strength,
            'player_id': n,
            f'degree{suffix}': degree.get(n,0),
            f'in_degree{suffix}': in_degree.get(n,0),
            f'out_degree{suffix}': out_degree.get(n,0),
            f'betweenness{suffix}': betweenness.get(n,0),
            f'harmonic{suffix}': harmonic.get(n,0),
            f'pagerank{suffix}': pagerank.get(n,0)
        } for n in nodes])

    return pd.DataFrame(results)
