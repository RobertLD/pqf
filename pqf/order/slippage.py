import polars as pl


def simple_slippage(
    prices: pl.Series | pl.Expr, slippage_rate: float
) -> pl.Series | pl.Expr:
    return prices * slippage_rate
