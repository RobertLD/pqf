import polars as pl
import polars.testing as plt
from pqf.indicator.util import apply_expr_to_series
from pqf.order.slippage import simple_slippage


class TestSlippage:
    def test_simple_slippage_calculated_correctly_as_series(self):
        prices = pl.Series([1, 2, 3, 4])
        result = simple_slippage(prices, slippage_rate=0.001)

        plt.assert_series_equal(result, prices / 1000)

    def test_simple_slippage_calculated_correctly_as_expr(self):
        prices = pl.Series([1, 2, 3, 4])
        expr = simple_slippage(pl.col("*"), slippage_rate=0.001)
        result = apply_expr_to_series(prices, expr)
        plt.assert_series_equal(result, prices / 1000)
