import polars as pl
import pytest

from pqf.indicator.util import apply_expr_to_series
from pqf.research.statistics import sharpe_ratio
from pqf.research.statistics import sortino_ratio


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


class TestSortinoRatio:
    def test_sortino_ratio_only_positive_returns(self):
        returns = pl.Series([0.05, 0.1, 0.15, 0.2])
        risk_free_rate = 0.02
        result = sortino_ratio(returns, risk_free_rate)
        expected_result = None
        assert result == pytest.approx(expected_result)

        result_expr = sortino_ratio(pl.col("*"), risk_free_rate)

        result = apply_expr_to_series(returns, result_expr).item()
        assert result == pytest.approx(expected_result)

    def test_sortino_ratio_postive_risk_free_rate(self):
        returns = pl.Series([-0.05, -0.1, 0.1, 0.15, -0.1, 0.20])
        risk_free_rate = 0.02
        result = sortino_ratio(returns, risk_free_rate)
        expected_result = 0.638
        assert result == pytest.approx(expected_result)

        result_expr = sortino_ratio(pl.col("*"), risk_free_rate)

        result = apply_expr_to_series(returns, result_expr).item()
        assert result == pytest.approx(expected_result)

    def test_sortino_ratio_negative_risk_free_rate(self):
        returns = pl.Series([-0.05, -0.1, 0.1, 0.15])
        risk_free_rate = -0.02
        result = sortino_ratio(returns, risk_free_rate)
        expected_result = 1.8
        assert result == pytest.approx(expected_result)

        result_expr = sortino_ratio(pl.col("*"), risk_free_rate)

        result = apply_expr_to_series(returns, result_expr).item()
        assert result == pytest.approx(expected_result)

    def test_sortino_ratio_empty(self):
        returns = pl.Series([])
        risk_free_rate = 0.02
        result = sortino_ratio(returns, risk_free_rate)
        expected_result = None

        assert result == pytest.approx(expected_result)

        result_expr = sortino_ratio(pl.col("*"), risk_free_rate)
        result = apply_expr_to_series(returns, result_expr).item()
        assert result == pytest.approx(expected_result)
