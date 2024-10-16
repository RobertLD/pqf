from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    import polars as pl

@overload
def sharpe_ratio(returns: pl.Series, risk_free_rate: float) -> float | None: ...
@overload
def sharpe_ratio(returns: pl.Expr, risk_free_rate: float) -> pl.Expr: ...
def estimate_market_returns(
    market_constituent_returns: pl.LazyFrame | pl.DataFrame, date_column: str
) -> pl.LazyFrame | pl.DataFrame: ...
