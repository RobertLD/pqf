import itertools
from typing import NamedTuple

import polars as pl
import polars.selectors as cs


class FactorAssetPair(NamedTuple):
    factor: str
    asset: str


def simple_factor_exposure(
    factors: pl.LazyFrame, factor_names: list[str]
) -> pl.LazyFrame:
    return factors.with_columns(
        *[
            pl.col(factor)
            .qcut([0.25, 0.75], labels=["-1", "0", "1"])
            .cast(pl.Int8)
            .name.keep()
            for factor in factor_names
        ]
    )


def get_factor_asset_permutations(
    factors: list[str], assets: list[str]
) -> list[FactorAssetPair]:
    return [
        FactorAssetPair(factor, asset)
        for factor, asset in itertools.product(factors, assets)
    ]


def simple_factor_returns(
    factors: pl.LazyFrame,
    returns: pl.LazyFrame,
    date_column: str,
) -> pl.LazyFrame:
    factor_columns = factors.select(cs.float()).columns

    factor_exposure = simple_factor_exposure(factors, factor_columns)
    factor_rank_df = returns.join(factor_exposure, on=date_column)

    asset_columns = returns.select(cs.float()).columns
    factor_returns_expressions = [
        pl.col(fs.factor)
        .mul(pl.col(fs.asset).shift(-1))
        .alias(f"{fs.asset}_{fs.factor}_return")
        for fs in get_factor_asset_permutations(factor_columns, asset_columns)
    ]
    factor_return_df = factor_rank_df.select(
        pl.col(date_column), *factor_returns_expressions
    )

    return factor_return_df
