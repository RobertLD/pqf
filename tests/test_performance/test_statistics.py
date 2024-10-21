import polars as pl
import pytest

from pqf.research.statistics import annualized_return
from pqf.indicator.util import apply_expr_to_series


class TestAnnualizedReturn:
    def test_annualized_return_positive_returns(self):
        prices = pl.Series(
            [
                100.00,
                100.05,
                100.10,
                100.07,
                100.20,
                100.15,
                100.25,
                100.30,
                100.35,
                100.40,
                100.45,
            ]
        )
        returns = (prices / prices.shift(1)).log().alias("returns")
        result = annualized_return(returns)
        assert result is not None

        expected_result = 0.1787
        assert result[-1] == pytest.approx(expected_result, rel=1e-2)

        result_expr = annualized_return(pl.col("returns"))
        result_expr_applied = apply_expr_to_series(returns, result_expr)
        assert result_expr_applied[-1] == pytest.approx(expected_result, rel=1e-2)

    def test_annualized_return_monthly_data(self):
        prices = pl.Series(
            [
                100.00,
                100.05,
                100.10,
                100.07,
                100.20,
                100.15,
                100.25,
                100.30,
                100.35,
                100.40,
                100.45,
                100.50,
                100.55,
                100.60,
                100.65,
                100.70,
                100.75,
                100.80,
                100.85,
                100.90,
                100.95,
                101.00,
                101.05,
                101.10,
                101.15,
                101.20,
                101.25,
                101.30,
                101.35,
                101.40,
                101.45,
                101.50,
                101.55,
            ]
        )

        returns = (prices / prices.shift(1)).log().alias("returns")
        result = annualized_return(returns)
        assert result is not None

        expected_result = 0.191
        assert result[-1] == pytest.approx(expected_result, rel=1e-2)

        result_expr = annualized_return(pl.col("returns"))
        result_expr_applied = apply_expr_to_series(returns, result_expr)
        assert result_expr_applied[-1] == pytest.approx(expected_result, rel=1e-2)

    def test_annualized_return_negative_returns(self):
        prices = pl.Series(
            [
                100.00,
                99.50,
                99.00,
                98.75,
                98.50,
                98.25,
                97.50,
                97.00,
                96.75,
                96.50,
                96.00,
            ]
        )

        returns = (prices / prices.shift(1)).log().alias("returns")

        result = annualized_return(returns)
        assert result is not None

        expected_result = -0.7746
        assert result[-1] == pytest.approx(expected_result, rel=1e-2)

        result_expr = annualized_return(pl.col("returns"))
        result_expr_applied = apply_expr_to_series(returns, result_expr)
        assert result_expr_applied[-1] == pytest.approx(expected_result, rel=1e-2)

    def test_annualized_return_none(self):
        prices = pl.Series([100.00])
        returns = (prices / prices.shift(1)).log().alias("returns")

        result = annualized_return(returns)
        assert result is None

        empty_prices = pl.Series([], dtype=pl.Float64)  # Empty price series
        empty_returns = (
            (empty_prices / empty_prices.shift(1)).log().alias("returns")
        )  # This will also be all nulls

        result_empty = annualized_return(empty_returns)
        assert result_empty is None
