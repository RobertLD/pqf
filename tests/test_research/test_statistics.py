from datetime import datetime

import polars as pl
import polars.selectors as cs
import polars.testing as plt

from pqf.research.statistics import estimate_market_returns


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
