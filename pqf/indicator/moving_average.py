import polars as pl


def simple_moving_average(
    prices: pl.Expr | pl.Series, window_size: int
) -> pl.Expr | pl.Series:
    """Calculate the simple moving average of prices over a specified window size.

    Args:
        prices (pl.Expr | pl.Series): The input prices data as a Polars expression or series.
        window_size (int): The size of the window for calculating the moving average.

    Returns:
        pl.Expr | pl.Series: The simple moving average values.
    """
    return prices.rolling_mean(window_size)


def exponential_moving_average(
    prices: pl.Series | pl.Expr, window_size: int
) -> pl.Series | pl.Expr:
    """Calculate the exponential moving average of prices.

    Args:
        prices (pl.Series | pl.Expr): Series or Expression containing the prices.
        window_size (int): Number of periods to consider in the moving average calculation.

    Returns:
        pl.Series | pl.Expr: Series or Expression with the exponential moving average values.
    """
    return prices.ewm_mean(span=window_size, adjust=False)
