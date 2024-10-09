import polars as pl
import polars.testing as plt

from pqf.indicators.util import apply_expr_to_series


class TestUtil:
    def test_apply_expr_to_series_applies_expr_to_series(self):
        series = pl.Series([1, 2, 3])
        expr = pl.all().mul(2)
        result = apply_expr_to_series(series, expr)
        plt.assert_series_equal(result, pl.Series([2, 4, 6]))
