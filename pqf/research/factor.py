import itertools
from typing import NamedTuple

import polars as pl
import polars.selectors as cs


class FactorAssetPair(NamedTuple):
    """Pair between a factor and an asset."""

    factor: str
    asset: str


def mean_factor_returns_by_quantile(
    quantiles: int,
    factors: pl.LazyFrame,
    returns: pl.LazyFrame,
    date_column: str,
    cumulative: bool = False,
) -> pl.LazyFrame:
    factor_columns = factors.select(cs.float()).columns
    factor_exposure = _simple_factor_quantiles(factors, factor_columns, quantiles)
    factor_rank_df = returns.join(factor_exposure, on=date_column)

    asset_columns = returns.select(cs.float()).columns

    factor_df = returns.join(factor_rank_df, on=[date_column] + asset_columns)
    factor_quantiles = (
        factor_df.group_by(factor_columns).mean().select(factor_columns + asset_columns)
    )

    if cumulative:
        factor_quantiles = factor_quantiles.select(cs.float().cum_sum())

    return factor_quantiles


def simple_factor_long_short_returns(
    factors: pl.LazyFrame,
    returns: pl.LazyFrame,
    date_column: str,
    cumulative: bool = False,
) -> pl.LazyFrame:
    """Calculates factor portfolio return based on factor quantiles and asset returns.

    Args:
        factors (pl.LazyFrame): DataFrame containing factor data.
        returns (pl.LazyFrame): DataFrame containing asset returns data.
        date_column (str): Name of the column representing dates.
        cumulative (bool, optional): Flag to calculate cumulative returns. Defaults to False.

    Returns:
        pl.LazyFrame: DataFrame with factor returns calculated.
    """
    factor_columns = factors.select(cs.float()).columns
    factor_exposure = _simple_factor_quantiles(
        factors, factor_columns, 3, ["-1", "0", "1"]
    )
    factor_rank_df = returns.join(factor_exposure, on=date_column)

    asset_columns = returns.select(cs.float()).columns
    factor_returns_expressions = [
        pl.col(fs.factor)
        .cast(pl.Int8)
        .mul(pl.col(fs.asset).shift(-1))
        .alias(f"{fs.asset}_{fs.factor}_return")
        for fs in _get_factor_asset_permutations(factor_columns, asset_columns)
    ]
    if cumulative:
        factor_returns_expressions = [f.cum_sum() for f in factor_returns_expressions]
    factor_return_df = factor_rank_df.select(
        pl.col(date_column), *factor_returns_expressions
    )

    return factor_return_df


def _get_factor_asset_permutations(
    factors: list[str], assets: list[str]
) -> list[FactorAssetPair]:
    return [
        FactorAssetPair(factor, asset)
        for factor, asset in itertools.product(factors, assets)
    ]


def _simple_factor_quantiles(
    factors: pl.LazyFrame,
    factor_names: list[str],
    quantiles: int = 3,
    labels: list[str] | None = None,
) -> pl.LazyFrame:
    if labels is None:
        labels = [str(i) for i in range(quantiles)]
    return factors.with_columns(
        *[
            pl.col(factor)
            .qcut(quantiles, allow_duplicates=True, labels=labels)
            .name.keep()
            for factor in factor_names
        ]
    )
