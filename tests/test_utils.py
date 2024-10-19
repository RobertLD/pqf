from datetime import datetime

import polars as pl
import polars.testing as plt

from pqf.utils import forward_returns


class TestReturns:
    def test_simple_forward_returns_return_correct_data(self):
        data = pl.DataFrame(
            [
                pl.Series(
                    "close",
                    [
                        37753.05,
                        37730.62,
                        37736.69,
                        37732.53,
                        37723.15,
                        37713.25,
                        37689.8,
                        37697.39,
                        37679.58,
                        37693.89,
                    ],
                    dtype=pl.Float64,
                ),
                pl.Series(
                    "timestamp",
                    [
                        datetime(2022, 5, 1, 0, 41),
                        datetime(2022, 5, 1, 0, 42),
                        datetime(2022, 5, 1, 0, 43),
                        datetime(2022, 5, 1, 0, 44),
                        datetime(2022, 5, 1, 0, 45),
                        datetime(2022, 5, 1, 0, 46),
                        datetime(2022, 5, 1, 0, 47),
                        datetime(2022, 5, 1, 0, 48),
                        datetime(2022, 5, 1, 0, 49),
                        datetime(2022, 5, 1, 0, 50),
                    ],
                    dtype=pl.Datetime(time_unit="ms", time_zone=None),
                ),
                pl.Series(
                    "BTC",
                    [
                        -0.0010687039867236063,
                        -0.0005941241833441348,
                        0.00016087729276645092,
                        -0.00011023754335643884,
                        -0.00024859186489740765,
                        -0.0002624383170546854,
                        -0.0006217973789052147,
                        0.0002013807449229369,
                        -0.00047244650093806686,
                        0.00037978130329472016,
                    ],
                    dtype=pl.Float64,
                ),
            ]
        )
        expected_result = pl.DataFrame(
            [
                pl.Series(
                    "timestamp",
                    [
                        datetime(2022, 5, 1, 0, 41),
                        datetime(2022, 5, 1, 0, 42),
                        datetime(2022, 5, 1, 0, 43),
                        datetime(2022, 5, 1, 0, 44),
                        datetime(2022, 5, 1, 0, 45),
                        datetime(2022, 5, 1, 0, 46),
                        datetime(2022, 5, 1, 0, 47),
                        datetime(2022, 5, 1, 0, 48),
                        datetime(2022, 5, 1, 0, 49),
                        datetime(2022, 5, 1, 0, 50),
                    ],
                    dtype=pl.Datetime(time_unit="ms", time_zone=None),
                ),
                pl.Series(
                    "BTC_1",
                    [
                        -0.0005941241833441348,
                        0.00016087729276645092,
                        -0.00011023754335643884,
                        -0.00024859186489740765,
                        -0.0002624383170546854,
                        -0.0006217973789052147,
                        0.0002013807449229369,
                        -0.00047244650093806686,
                        0.00037978130329472016,
                        None,
                    ],
                    dtype=pl.Float64,
                ),
                pl.Series(
                    "BTC_5",
                    [
                        -0.0002624383170546854,
                        -0.0006217973789052147,
                        0.0002013807449229369,
                        -0.00047244650093806686,
                        0.00037978130329472016,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                    dtype=pl.Float64,
                ),
                pl.Series(
                    "BTC_10",
                    [None, None, None, None, None, None, None, None, None, None],
                    dtype=pl.Float64,
                ),
            ]
        )
        data = data.select(pl.col("timestamp"), pl.col("close").alias("BTC"))
        result = forward_returns(data, "timestamp", ["BTC"], [1, 5, 10], log=False)
        plt.assert_frame_equal(result, expected_result)

    def test_log_forward_returns_return_correct_data(self):
        data = pl.DataFrame(
            [
                pl.Series(
                    "close",
                    [
                        37753.05,
                        37730.62,
                        37736.69,
                        37732.53,
                        37723.15,
                        37713.25,
                        37689.8,
                        37697.39,
                        37679.58,
                        37693.89,
                    ],
                    dtype=pl.Float64,
                ),
                pl.Series(
                    "timestamp",
                    [
                        datetime(2022, 5, 1, 0, 41),
                        datetime(2022, 5, 1, 0, 42),
                        datetime(2022, 5, 1, 0, 43),
                        datetime(2022, 5, 1, 0, 44),
                        datetime(2022, 5, 1, 0, 45),
                        datetime(2022, 5, 1, 0, 46),
                        datetime(2022, 5, 1, 0, 47),
                        datetime(2022, 5, 1, 0, 48),
                        datetime(2022, 5, 1, 0, 49),
                        datetime(2022, 5, 1, 0, 50),
                    ],
                    dtype=pl.Datetime(time_unit="ms", time_zone=None),
                ),
                pl.Series(
                    "BTC",
                    [
                        -0.0010687039867236063,
                        -0.0005941241833441348,
                        0.00016087729276645092,
                        -0.00011023754335643884,
                        -0.00024859186489740765,
                        -0.0002624383170546854,
                        -0.0006217973789052147,
                        0.0002013807449229369,
                        -0.00047244650093806686,
                        0.00037978130329472016,
                    ],
                    dtype=pl.Float64,
                ),
            ]
        )
        expected_result = pl.DataFrame(
            [
                pl.Series(
                    "timestamp",
                    [
                        datetime(2022, 5, 1, 0, 41),
                        datetime(2022, 5, 1, 0, 42),
                        datetime(2022, 5, 1, 0, 43),
                        datetime(2022, 5, 1, 0, 44),
                        datetime(2022, 5, 1, 0, 45),
                        datetime(2022, 5, 1, 0, 46),
                        datetime(2022, 5, 1, 0, 47),
                        datetime(2022, 5, 1, 0, 48),
                        datetime(2022, 5, 1, 0, 49),
                        datetime(2022, 5, 1, 0, 50),
                    ],
                    dtype=pl.Datetime(time_unit="ms", time_zone=None),
                ),
                pl.Series(
                    "BTC_1",
                    [
                        -0.0005943007450532889,
                        0.00016086435340234573,
                        -0.0001102436199609258,
                        -0.0002486227689768583,
                        -0.00026247276001534203,
                        -0.000621990775067971,
                        0.00020136047054108985,
                        -0.0004725581389486422,
                        0.00037970920462981894,
                        None,
                    ],
                    dtype=pl.Float64,
                ),
                pl.Series(
                    "BTC_5",
                    [
                        -0.00026247276001534203,
                        -0.000621990775067971,
                        0.00020136047054108985,
                        -0.0004725581389486422,
                        0.00037970920462981894,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                    dtype=pl.Float64,
                ),
                pl.Series(
                    "BTC_10",
                    [None, None, None, None, None, None, None, None, None, None],
                    dtype=pl.Float64,
                ),
            ]
        )
        data = data.select(pl.col("timestamp"), pl.col("close").alias("BTC"))
        result = forward_returns(data, "timestamp", ["BTC"], [1, 5, 10])
        plt.assert_frame_equal(result, expected_result)
