import polars as pl
import polars.selectors as cs


def simple_factor_returns(
    factor: pl.LazyFrame, returns: pl.LazyFrame, date_column: str
) -> pl.LazyFrame:
    factor_columns = factor.select(cs.float()).columns
    factor_portf_exprs = [
        pl.col(factor_col)
        .qcut([0.25, 0.75], labels=["short", "out", "long"])
        .alias(f"{factor_col}_exposure")
        for factor_col in factor_columns
    ]
    factor_rank_df = returns.join(
        factor.with_columns(*factor_portf_exprs), on=date_column
    )

    factor_return_df = factor_rank_df.with_columns(
        *[
            (
                pl.when(pl.col(f"{factor_col}_exposure") == "long")
                .then(pl.col("returns").shift(-1))  # Positive return for long
                .when(pl.col(f"{factor_col}_exposure") == "short")
                .then(pl.col("returns").shift(-1).mul(-1))  # Negative return for short
                .otherwise(0)  # No contribution for 'out'
            ).alias(f"{factor_col}_return")
            for factor_col in factor_columns
        ]
    ).select(pl.col(date_column), cs.contains("_return"))
    return factor_return_df
