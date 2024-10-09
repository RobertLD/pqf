from typing import overload
import polars as pl


@overload
def simple_moving_average(prices: pl.Series, window_size: int) -> pl.Series: ...

@overload
def simple_moving_average(prices: pl.Expr, window_size: int) -> pl.Expr: ...

def simple_moving_average(prices: pl.Expr | pl.Series, window_size: int) -> pl.Expr | pl.Series:
  return prices.rolling_mean(window_size)