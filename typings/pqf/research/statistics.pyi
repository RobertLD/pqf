from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    import polars as pl

@overload
def sharpe_ratio(returns: pl.Series, risk_free_rate: float) -> pl.Series: ...
@overload
def sharpe_ratio(returns: pl.Expr, risk_free_rate: float) -> pl.Expr: ...