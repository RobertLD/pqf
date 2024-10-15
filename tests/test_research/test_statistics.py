import polars as pl
import pytest

from pqf.indicator.util import apply_expr_to_series
from pqf.research.statistics import sharpe_ratio, information_coef


class TestSharpeRatio:
    def test_sharpe_ratio_positive_risk_free_rate(self):
        returns = pl.Series([0.05, 0.1, 0.15, 0.2])
        risk_free_rate = 0.02
        result = sharpe_ratio(returns, risk_free_rate)
        expected_result = 1.62665300
        assert result == pytest.approx(expected_result)

        result_expr = sharpe_ratio(pl.col("*"), risk_free_rate)

        result = apply_expr_to_series(returns, result_expr).item()
        assert result == pytest.approx(expected_result)

    def test_sharpe_ratio_empty_returns(self):
        returns = pl.Series([])
        risk_free_rate = 0.02
        result = sharpe_ratio(returns, risk_free_rate)
        expected_result = None

        assert result == pytest.approx(expected_result)

        result_expr = sharpe_ratio(pl.col("*"), risk_free_rate)
        result = apply_expr_to_series(returns, result_expr).item()
        assert result == pytest.approx(expected_result)


class TestInformationCoef:
    def test_information_coef_with_highly_correlated_series(self):
        prices = pl.Series([1, 2, 3, 4])
        factor = pl.Series([2, 4, 8, 16])
        ic = information_coef(prices, factor)
        assert ic == pytest.approx(0.95916, rel=0.001)

    def test_information_coef_with_highly_inverse_correlated_series(self):
        prices = pl.Series([1, 2, 3, 4])
        factor = pl.Series([-2, -4, -8, -16])
        ic = information_coef(prices, factor)
        assert abs(ic) == pytest.approx(0.95916, rel=0.001)
