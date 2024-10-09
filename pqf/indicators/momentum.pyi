from typing import overload

import polars as pl

@overload
def rsi(data: pl.Expr, period: int) -> pl.Expr: ...
@overload
def rsi(data: pl.Series, period: int) -> pl.Series: ...
