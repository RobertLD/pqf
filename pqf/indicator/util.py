import polars as pl


def apply_expr_to_series(series: pl.Series, expr: pl.Expr) -> pl.Series:
    return series.to_frame().select(expr).to_series()
