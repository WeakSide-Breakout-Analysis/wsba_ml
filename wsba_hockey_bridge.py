"""Explicit pandas/Polars boundary for wsba_ml calls into wsba_hockey."""

import pandas as pd
import polars as pl
import wsba_hockey as wsba


def polars_to_pandas(data):
    """Convert a wsba_hockey result to the pandas convention used by wsba_ml."""
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pl.DataFrame):
        try:
            return data.to_pandas()
        except (ImportError, ModuleNotFoundError):
            return pd.DataFrame(data.to_dicts())
    return data


def pandas_to_polars(data):
    """Convert a wsba_ml frame before passing it into wsba_hockey."""
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, pd.DataFrame):
        return pl.from_pandas(data, include_index=False)
    return data


def nhl_agg_stats_pandas(data, *args, **kwargs):
    """Run the Polars wsba_hockey aggregator and return a pandas frame."""
    result = wsba.nhl_agg_stats(pandas_to_polars(data), *args, **kwargs)
    return polars_to_pandas(result)
