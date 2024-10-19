import polars as pl


def forward_returns(
    data: pl.LazyFrame | pl.DataFrame,
    date_column: str,
    factors: list[str],
    assets: list[str],
    periods: list[int],
) -> pl.LazyFrame | pl.DataFrame:
    """Calculate the N period forward returns as percent change for each requested asset.

    Args:
        data (pl.LazyFrame): Underlying data containing (date_column, [FACTORS], [ASSETS])
        date_column (str): The time column by name
        factors (list[str]): A list of factor column names
        assets (list[str]): A list of asset names
        periods (list[int]): The periods to calculate forward returns for

    Returns:
        pl.LazyFrame | pl.DataFrame: Dataframe containing the factors and forward returns
    """
    forward_return_expressions = [
        pl.col(a).cum_sum().shift(-i).alias(f"{a}_{i}") for a in assets for i in periods
    ]
    return data.select(
        pl.col(date_column), *[pl.col(f) for f in factors], *forward_return_expressions
    )
