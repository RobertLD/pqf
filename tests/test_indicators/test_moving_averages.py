import polars as pl
import polars.testing as plt
from pqf.indicators.moving_average import simple_moving_average


class TestSimpleMovingAverage:
    def test_simple_moving_avarage_nan_padding(self):
        prices = pl.Series([1, 2, 3, 4, 5])
        result = simple_moving_average(prices, 3)
        plt.assert_series_equal(result, pl.Series([None, None, 2.0, 3.0, 4.0]))

    def test_simple_moving_avarage_nan_padding_with_window_size_as_array_length(self):
        prices = pl.Series([1, 2, 3, 4, 5])
        result = simple_moving_average(prices, 5)
        plt.assert_series_equal(result, pl.Series([None, None, None, None, 3.0]))
