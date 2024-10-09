import polars as pl
import polars.testing as plt

from pqf.indicators.moving_average import (
    exponential_moving_average,
    simple_moving_average,
)


class TestMovingAverage:
    class TestSimpleMovingAverage:
        def test_simple_moving_avarage_nan_padding(self):
            prices = pl.Series([1, 2, 3, 4, 5])
            result = simple_moving_average(prices, 3)
            plt.assert_series_equal(result, pl.Series([None, None, 2.0, 3.0, 4.0]))

        def test_simple_moving_avarage_nan_padding_with_window_size_as_array_length(
            self,
        ):
            prices = pl.Series([1, 2, 3, 4, 5])
            result = simple_moving_average(prices, 5)
            plt.assert_series_equal(result, pl.Series([None, None, None, None, 3.0]))

        def test_simple_moving_avarage_correctly_handles_expr(self):
            prices = pl.Series([1, 2, 3, 4, 5], dtype=pl.Float64)
            expected_result = pl.Series([None, None, 2.0, 3.0, 4.0], dtype=pl.Float64)

            expression = simple_moving_average(pl.all(), 3)

            result = prices.to_frame().select(expression).to_series()

            plt.assert_series_equal(result, expected_result)

    class TestExponentialMovingAverage:
        def test_exp_moving_avarage_correctly_weights_vector(self):
            prices = pl.Series([1, 2, 3, 4, 5], dtype=pl.Float64)
            expected_result = pl.Series([1, 1.5, 2.25, 3.125, 4.0625], dtype=pl.Float64)

            result = exponential_moving_average(prices, 3)
            plt.assert_series_equal(result, expected_result)

        def test_exp_moving_avarage_correctly_weights_vector_as_expr(self):
            prices = pl.Series([1, 2, 3, 4, 5], dtype=pl.Float64)
            expected_result = pl.Series([1, 1.5, 2.25, 3.125, 4.0625], dtype=pl.Float64)
            expression = exponential_moving_average(pl.all(), 3)
            result = prices.to_frame().select(expression).to_series()

            plt.assert_series_equal(result, expected_result)
