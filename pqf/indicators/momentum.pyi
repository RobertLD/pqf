from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    import polars as pl

@overload
def rsi(data: pl.Expr, period: int) -> pl.Expr: ...
@overload
def rsi(data: pl.Series, period: int) -> pl.Series: ...
@overload
def macd(
    data: pl.Series,
    slow_period: int = 26,
    fast_period: int = 12,
    signal_period: int = 9,
) -> pl.Series: ...
@overload
def macd(
    data: pl.Expr, slow_period: int = 26, fast_period: int = 12, signal_period: int = 9
) -> pl.Expr: ...
