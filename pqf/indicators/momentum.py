import polars as pl


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
