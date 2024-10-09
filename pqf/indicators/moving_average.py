import polars as pl


def simple_moving_average(
    prices: pl.Expr | pl.Series, window_size: int
) -> pl.Expr | pl.Series:
    return prices.rolling_mean(window_size)


def exponential_moving_average(
    prices: pl.Series | pl.Expr, window_size: int
) -> pl.Series | pl.Expr:
    return prices.ewm_mean(span=window_size, adjust=False)
