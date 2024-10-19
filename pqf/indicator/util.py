import polars as pl


def apply_expr_to_series(series: pl.Series, expr: pl.Expr) -> pl.Series:
    """Apply an expression to a series.

    Args:
        series (pl.Series): The input series to apply the expression to.
        expr (pl.Expr): The expression to apply to the series.

    Returns:
        pl.Series: The resulting series after applying the expression.
    """
    return series.to_frame().select(expr).to_series()
