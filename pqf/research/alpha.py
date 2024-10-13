import polars as pl
import polars.selectors as cs


def simple_factor_exposure(factors: pl.LazyFrame) -> pl.LazyFrame:
    return factors.with_columns(
        cs.float()
        .qcut([0.25, 0.75], labels=["-1", "0", "1"])
        .cast(pl.Int8)
        .name.suffix("_exposure")
    )


def simple_factor_returns(
    factors: pl.LazyFrame, returns: pl.LazyFrame, date_column: str
) -> pl.LazyFrame:
    factor_exposure = simple_factor_exposure(factors)
    factor_rank_df = returns.join(factor_exposure, on=date_column)

    factor_return_df = factor_rank_df.select(
        pl.col(date_column),
        cs.contains("_exposure")
        .mul(pl.col("returns").shift(-1))
        .name.map(lambda col: col.replace("exposure", "return")),
    )

    return factor_return_df
