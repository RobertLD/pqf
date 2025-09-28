import polars as pl
from datetime import timedelta


def estimate_time_grain(date_series: pl.Series) -> timedelta:
    """Estimate the time grain of a datetime series.

    Args:
        date_series (pl.Series): A polars Series of datetime type.

    Returns:
        str: The estimated time grain as a string (e.g., '1d', '1h', '1min').
    """
    if not isinstance(date_series, pl.Series):
        raise TypeError("Input must be a polars Series.")
    if date_series.dtype != pl.Datetime and date_series.dtype != pl.Date:
        raise ValueError("Series must be of date or datetime type.")

    sorted_dates = date_series.sort()
    diffs = sorted_dates.diff().drop_nulls()
    if diffs.is_empty():
        raise ValueError("Not enough data to estimate time grain.")

    common_diff = diffs.mode()
    if common_diff.is_empty() or common_diff.len() > 1:
        raise ValueError("Could not determine a common time difference.")

    return common_diff.item()
