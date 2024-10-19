from datetime import datetime

import polars as pl
import polars.testing as plt

from pqf.research.factor import (
    mean_factor_returns_by_quantile,
    simple_factor_long_short_returns,
)


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
            schema=["timestamp", "BTC"],
        ).lazy()
        factor_return_df = simple_factor_long_short_returns(
            factor, returns, "timestamp"
        )
        factor_returns = (
            factor_return_df.collect().select("BTC_factor_return").to_series()
        )
        plt.assert_series_equal(
            factor_returns,
            pl.Series("BTC_factor_return", [0.0, 0.01, 0.05, 0.07, None]),
        )

    def test_simple_factor_returns_calculated_correctly_with_two_assets_and_one_factor(
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
                [0.1, 0.1, -0.1, 0.5, -0.8, 0.3],
            ],
            schema=["timestamp", "A", "B"],
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
                [0.01, 0.01, -0.01, 0.05, -0.07, 0.032],
            ],
            schema=["timestamp", "BTC", "ETH"],
        ).lazy()
        factor_return_df = simple_factor_long_short_returns(
            factor, returns, "timestamp"
        )
        factor_returns = factor_return_df.collect()

        expected_returns = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2021, 1, 2),
                    datetime(2021, 1, 3),
                    datetime(2021, 1, 4),
                    datetime(2021, 1, 5),
                    datetime(2021, 1, 6),
                ],
                "BTC_A_return": [0.0, 0.01, 0.05, 0.07, None],
                "ETH_A_return": [0.0, 0.01, 0.05, 0.07, None],
                "BTC_B_return": [0.0, 0.01, 0.05, 0.07, None],
                "ETH_B_return": [0.0, 0.01, 0.05, 0.07, None],
            }
        )
        plt.assert_frame_equal(factor_returns, expected_returns)

    def test_cumulative_simple_factor_returns_calculated_correctly_with_single_asset_and_factor(
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
            schema=["timestamp", "BTC"],
        ).lazy()
        factor_return_df = simple_factor_long_short_returns(
            factor, returns, "timestamp", cumulative=True
        )
        factor_returns = (
            factor_return_df.collect().select("BTC_factor_return").to_series()
        )
        plt.assert_series_equal(
            factor_returns,
            pl.Series("BTC_factor_return", [0.0, 0.01, 0.06, 0.13, None]),
        )

    def test_mean_factor_returns_calculated_correctly_with_two_assets_and_one_factor(
        self,
    ):
        with pl.StringCache():
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
                    [0.1, 0.1, -0.1, 0.5, -0.8, 0.3],
                ],
                schema=["timestamp", "A", "B"],
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
                    [0.01, 0.01, -0.01, 0.05, -0.07, 0.032],
                ],
                schema=["timestamp", "BTC", "ETH"],
            ).lazy()
            factor_return_df = mean_factor_returns_by_quantile(
                3, factor, returns, "timestamp"
            )
            quantile_returns = factor_return_df.collect()

            expected_quantile_returns = pl.DataFrame(
                [
                    pl.Series(
                        "A", ["1", "2", "0"], dtype=pl.Categorical(ordering="physical")
                    ),
                    pl.Series(
                        "B", ["1", "2", "0"], dtype=pl.Categorical(ordering="physical")
                    ),
                    pl.Series("BTC", [0.01, -0.04, 0.03000], dtype=pl.Float64),
                    pl.Series("ETH", [0.01, -0.04, 0.03000], dtype=pl.Float64),
                ]
            )
            plt.assert_frame_equal(
                quantile_returns, expected_quantile_returns, check_row_order=False
            )
