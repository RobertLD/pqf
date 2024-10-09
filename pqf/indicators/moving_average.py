from typing import Any
import polars as pl
import numpy as np
from numba import guvectorize


def simple_moving_average(
    prices: pl.Expr | pl.Series, window_size: int
) -> pl.Expr | pl.Series:
    return prices.rolling_mean(window_size)


@guvectorize(["void(float64[:], float64, float64[:])"], "(n),(),(n)", target="cpu")
def _vec_ewma(data, alpha, ema) -> Any:
    n = data.shape[0]
    if n == 0:
        return
    ema[0] = data[0]  # Set the first value of EMA to be the first data point

    for i in range(1, n):
        ema[i] = (data[i] * alpha) + (ema[i - 1] * (1 - alpha))


def _numpy_ewma(data: np.ndarray, window_size: int) -> np.ndarray:
    alpha = 2 / (window_size + 1.0)

    # Prepare an array for the result
    ema_result = np.empty_like(data)
    _vec_ewma(data, alpha, ema_result)
    return ema_result


def exponential_moving_average(prices: pl.Series, window_size: int) -> pl.Series:
    if isinstance(prices, pl.Series):
        data = prices.to_numpy()
        result = _numpy_ewma(data, window_size)
        return pl.Series(result)

    elif isinstance(prices, pl.Expr):
        return prices.apply(lambda s: _vec_ewma(s.to_numpy()), return_dtype=pl.Float64)
