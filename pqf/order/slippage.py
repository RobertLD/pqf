import polars as pl


def simple_slippage(
    prices: pl.Series | pl.Expr, slippage_rate: float
) -> pl.Series | pl.Expr:
    """Calculate slippage on prices.

    Args:
        prices (pl.Series | pl.Expr): Series or expression representing prices.
        slippage_rate (float): Rate of slippage to apply.

    Returns:
        pl.Series | pl.Expr: Series or expression with slippage applied.
    """
    return prices * slippage_rate
