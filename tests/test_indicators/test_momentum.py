import polars as pl
import polars.testing as plt

from pqf.indicators.momentum import rsi
from pqf.indicators.util import apply_expr_to_series


class TestRsi:
    def test_rsi_calculated_correctly_over_series(self):
        data_list = [
            228.87,
            228.20,
            226.47,
            227.37,
            226.37,
            227.52,
            227.79,
            233,
            226.21,
            226.78,
            225.67,
            226.80,
            221.69,
            225.77,
        ]
        data = pl.Series(data_list)
        rsi_data = rsi(data, 14)
        expected_data = pl.Series(
            [
                None,
                None,
                None,
                27.27,
                20.93,
                37.61,
                40.56,
                68.89,
                42.49,
                44.29,
                41.75,
                44.96,
                36.00,
                44.78,
            ]
        )
        plt.assert_series_equal(rsi_data, expected_data, atol=0.01)

    def test_rsi_calculated_correctly_over_expr(self):
        data_list = [
            228.87,
            228.20,
            226.47,
            227.37,
            226.37,
            227.52,
            227.79,
            233,
            226.21,
            226.78,
            225.67,
            226.80,
            221.69,
            225.77,
        ]
        data = pl.Series(data_list)
        rsi_expr = rsi(pl.col("*"), 14)

        rsi_data = apply_expr_to_series(data, rsi_expr)
        expected_data = pl.Series(
            [
                None,
                None,
                None,
                27.27,
                20.93,
                37.61,
                40.56,
                68.89,
                42.49,
                44.29,
                41.75,
                44.96,
                36.00,
                44.78,
            ]
        )
        plt.assert_series_equal(rsi_data, expected_data, atol=0.01)
