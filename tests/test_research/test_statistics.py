from datetime import datetime

import polars as pl
import polars.selectors as cs
import polars.testing as plt
import pytest

from pqf.indicator.util import apply_expr_to_series
from pqf.research.statistics import (
    estimate_market_returns,
    information_coefficient,
    sharpe_ratio,
)


class TestEstimatedMarketReturn:
    def test_estimated_market_returns_returns_correct_data(self):
        market_constituents = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 10, 16, 10, 0),
                    datetime(2024, 10, 16, 10, 5),
                    datetime(2024, 10, 16, 10, 10),
                    datetime(2024, 10, 16, 10, 15),
                    datetime(2024, 10, 16, 10, 20),
                ],
                "price_1": [100.5, 101.2, 102.3, 101.8, 100.9],
                "price_2": [200.1, 199.8, 201.0, 202.2, 200.5],
                "price_3": [300.7, 299.5, 301.1, 302.3, 301.0],
            }
        )

        market_constituents = market_constituents.with_columns(
            cs.contains("price").pct_change().name.keep()
        )
        market_returns = estimate_market_returns(market_constituents, "timestamp")
        expected = pl.DataFrame(
            [
                pl.Series(
                    "timestamp",
                    [
                        datetime(2024, 10, 16, 10, 0),
                        datetime(2024, 10, 16, 10, 5),
                        datetime(2024, 10, 16, 10, 10),
                        datetime(2024, 10, 16, 10, 15),
                        datetime(2024, 10, 16, 10, 20),
                    ],
                    dtype=pl.Datetime(time_unit="us", time_zone=None),
                ),
                pl.Series(
                    "market_return",
                    [
                        None,
                        0.0004917451202642899,
                        0.007405936095055629,
                        0.0016893168785435742,
                        -0.007182915208872126,
                    ],
                    dtype=pl.Float64,
                ),
            ]
        )
        plt.assert_frame_equal(expected, market_returns)


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
    def test_rolling_information_coef_calculated_correctly_as_perfectly_correlated(
        self,
    ):
        returns = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 10, 16, 10, 0),
                    datetime(2024, 10, 16, 10, 5),
                    datetime(2024, 10, 16, 10, 10),
                    datetime(2024, 10, 16, 10, 15),
                    datetime(2024, 10, 16, 10, 20),
                ],
                "returns": [100.5, 101.2, 102.3, 101.8, 100.9],
            }
        ).with_columns(pl.col("returns").pct_change())
        factor_returns = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 10, 16, 10, 0),
                    datetime(2024, 10, 16, 10, 5),
                    datetime(2024, 10, 16, 10, 10),
                    datetime(2024, 10, 16, 10, 15),
                    datetime(2024, 10, 16, 10, 20),
                ],
                "factor": [100.5, 101.2, 102.3, 101.8, 100.9],
            }
        ).with_columns(pl.col("factor").pct_change())

        ic: pl.DataFrame = (
            information_coefficient(
                "timestamp", "returns", returns, "factor", factor_returns, 2
            )
            .select("factor_IC")
            .drop_nulls()
            .to_series()
            .drop_nans()
        )
        assert all(x == 1 for x in list(ic))

    def test_rolling_information_coef_calculated_correctly_as_not_correlated(self):
        returns = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 10, 16, 10, 0),
                    datetime(2024, 10, 16, 10, 5),
                    datetime(2024, 10, 16, 10, 10),
                    datetime(2024, 10, 16, 10, 15),
                    datetime(2024, 10, 16, 10, 20),
                ],
                "returns": [100.5, 101.2, 102.3, 101.8, 98.9],
            }
        ).with_columns(pl.col("returns").pct_change())
        factor_returns = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 10, 16, 10, 0),
                    datetime(2024, 10, 16, 10, 5),
                    datetime(2024, 10, 16, 10, 10),
                    datetime(2024, 10, 16, 10, 15),
                    datetime(2024, 10, 16, 10, 20),
                ],
                "factor": [100.8, 102.2, 99.3, 111.8, 100.9],
            }
        ).with_columns(pl.col("factor").pct_change())

        ic: pl.DataFrame = (
            information_coefficient(
                "timestamp", "returns", returns, "factor", factor_returns, 3
            )
            .select("factor_IC")
            .drop_nulls()
            .to_series()
            .drop_nans()
        )
        plt.assert_series_equal(pl.Series("factor_IC", [-1.0, -0.999605, 0.40917]), ic)
