import polars as pl

from pqf.indicator.moving_average import exponential_moving_average


def rsi(data: pl.Series | pl.Expr, period: int) -> pl.Series | pl.Expr:
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
    fast_ema = exponential_moving_average(data, fast_period)
    slow_ema = exponential_moving_average(data, slow_period)

    macd = fast_ema - slow_ema
    macd_signal = exponential_moving_average(macd, signal_period)
    macd_hist = macd - macd_signal
    return macd_hist
