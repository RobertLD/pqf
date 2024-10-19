import polars as pl

from pqf.indicator.moving_average import exponential_moving_average


def rsi(data: pl.Series | pl.Expr, period: int) -> pl.Series | pl.Expr:
    """Calculate the Relative Strength Index (RSI) for the given data series over a specified period.

    Args:
        data(pl.Series | pl.Expr): A Polars Series or Expression containing the data for which RSI needs to be calculated.
        period (int): An integer specifying the period for RSI calculation.

    Returns:
        pl.Series | pl.Expr: A Polars Series or Expression representing the RSI values calculated based on the input data and period.
    """
    rsi_expr = (
        pl.lit(100)
        - (
            100
            / (
                1
                + (
                    pl.when(data.diff() >= 0)
                    .then(data.diff())
                    .otherwise(0)
                    .rolling_mean(window_size=period, min_periods=3)
                    / pl.when(data.diff() < 0)
                    .then(data.diff())
                    .otherwise(0)
                    .abs()
                    .rolling_mean(window_size=period, min_periods=3)
                )
            )
        )
    ).replace(0, None)

    if isinstance(data, pl.Expr):
        return rsi_expr.name.keep()
    price_data = pl.LazyFrame(data)
    rsi_data = price_data.select(rsi_expr.alias(data.name))
    return rsi_data.collect().to_series()


def macd(
    data: pl.Series | pl.Expr,
    slow_period: int = 26,
    fast_period: int = 12,
    signal_period: int = 9,
) -> pl.Series | pl.Expr:
    """Calculate the Moving Average Convergence Divergence (MACD) indicator.

    Args:
        data (pl.Series | pl.Expr): Time series data for calculation.
        slow_period (int, optional): Number of periods for the slow EMA. Defaults to 26.
        fast_period (int, optional): Number of periods for the fast EMA. Defaults to 12.
        signal_period (int, optional): Number of periods for the signal line. Defaults to 9.

    Returns:
        pl.Series | pl.Expr: The MACD histogram values.
    """
    fast_ema = exponential_moving_average(data, fast_period)
    slow_ema = exponential_moving_average(data, slow_period)

    macd = fast_ema - slow_ema
    macd_signal = exponential_moving_average(macd, signal_period)
    macd_hist = macd - macd_signal
    return macd_hist
