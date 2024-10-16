from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    import polars as pl

@overload
def simple_slippage(prices: pl.Series, slippage_rate: float) -> pl.Series: ...
@overload
def simple_slippage(prices: pl.Expr, slippage_rate: float) -> pl.Expr: ...
