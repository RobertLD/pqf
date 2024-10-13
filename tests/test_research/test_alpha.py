from datetime import datetime

import polars as pl
import polars.testing as plt

from pqf.research.alpha import simple_factor_returns


class TestFactorReturns:
    def test_simple_factor_returns_calculated_correctly_with_single_asset_and_factor(
        self,
    ):
        factor = pl.DataFrame(
            [
                [
                    datetime(2021, 1, 1),
                    datetime(2021, 1, 2),
                    datetime(2021, 1, 3),
                    datetime(2021, 1, 4),
                    datetime(2021, 1, 5),
                    datetime(2021, 1, 6),
                ],
                [0.1, 0.1, -0.1, 0.5, -0.8, 0.3],
            ],
            schema=["timestamp", "factor"],
        ).lazy()
        returns = pl.DataFrame(
            [
                [
                    datetime(2021, 1, 2),
                    datetime(2021, 1, 3),
                    datetime(2021, 1, 4),
                    datetime(2021, 1, 5),
                    datetime(2021, 1, 6),
                    datetime(2021, 1, 7),
                ],
                [0.01, 0.01, -0.01, 0.05, -0.07, 0.032],
            ],
            schema=["timestamp", "returns"],
        ).lazy()
        factor_return_df = simple_factor_returns(factor, returns, "timestamp")
        factor_returns = factor_return_df.collect().select("factor_return").to_series()
        plt.assert_series_equal(
            factor_returns,
            pl.Series("factor_return", [0.0, 0.01, 0.05, 0.07, None]),
        )
