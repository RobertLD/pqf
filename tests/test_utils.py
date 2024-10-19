from datetime import datetime

import polars as pl
import polars.testing as plt

from pqf.utils import forward_returns


class TestReturns:
    def test_forward_returns_return_correct_data(self):
        data = pl.DataFrame(
            [
                pl.Series(
                    "close_time",
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
                    "symbol",
                    [
                        "BTCUSDT",
                        "BTCUSDT",
                        "BTCUSDT",
                        "BTCUSDT",
                        "BTCUSDT",
                        "BTCUSDT",
                        "BTCUSDT",
                        "BTCUSDT",
                        "BTCUSDT",
                        "BTCUSDT",
                    ],
                    dtype=pl.String,
                ),
                pl.Series(
                    "open",
                    [
                        37793.44,
                        37753.05,
                        37730.63,
                        37736.68,
                        37732.53,
                        37723.14,
                        37713.25,
                        37689.8,
                        37697.38,
                        37679.58,
                    ],
                    dtype=pl.Float64,
                ),
                pl.Series(
                    "high",
                    [
                        37806.79,
                        37776.96,
                        37746.31,
                        37747.26,
                        37735.74,
                        37727.07,
                        37713.25,
                        37713.12,
                        37710.0,
                        37701.92,
                    ],
                    dtype=pl.Float64,
                ),
                pl.Series(
                    "low",
                    [
                        37734.99,
                        37730.62,
                        37724.53,
                        37732.43,
                        37710.08,
                        37704.49,
                        37689.8,
                        37661.91,
                        37672.32,
                        37679.57,
                    ],
                    dtype=pl.Float64,
                ),
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
                    "volume",
                    [
                        38.08301,
                        39.41912,
                        21.87742,
                        8.05218,
                        10.6162,
                        12.83044,
                        9.69495,
                        40.82223,
                        10.51404,
                        12.33795,
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
                pl.Series(
                    "SMA25",
                    [
                        7.511431406522883e-05,
                        3.0512430892225738e-05,
                        3.0942655583896045e-05,
                        7.258123702260016e-05,
                        6.762893372631579e-05,
                        4.682877467100114e-05,
                        3.0568506804413556e-05,
                        4.064169162164622e-05,
                        2.7436876112282555e-05,
                        2.530223764309618e-05,
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
                    "SMA25",
                    [
                        7.511431406522883e-05,
                        3.0512430892225738e-05,
                        3.0942655583896045e-05,
                        7.258123702260016e-05,
                        6.762893372631579e-05,
                        4.682877467100114e-05,
                        3.0568506804413556e-05,
                        4.064169162164622e-05,
                        2.7436876112282555e-05,
                        2.530223764309618e-05,
                    ],
                    dtype=pl.Float64,
                ),
                pl.Series(
                    "BTC_1",
                    [
                        -0.001662828170067741,
                        -0.0015019508773012902,
                        -0.001612188420657729,
                        -0.0018607802855551366,
                        -0.002123218602609822,
                        -0.0027450159815150367,
                        -0.0025436352365921,
                        -0.0030160817375301666,
                        -0.0026363004342354465,
                        None,
                    ],
                    dtype=pl.Float64,
                ),
                pl.Series(
                    "BTC_5",
                    [
                        -0.002123218602609822,
                        -0.0027450159815150367,
                        -0.0025436352365921,
                        -0.0030160817375301666,
                        -0.0026363004342354465,
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

        result = forward_returns(data, "timestamp", ["SMA25"], ["BTC"], [1, 5, 10])
        plt.assert_frame_equal(result, expected_result)
