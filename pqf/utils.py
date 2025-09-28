import polars as pl


def forward_returns(
    prices: pl.LazyFrame | pl.DataFrame,
    date_column: str,
    assets: list[str],
    periods: list[int],
    log: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
    """Calculate the N period forward log returns as percent change for each requested asset.

    Input data is expected to be pricing data.

    Args:
        data (pl.LazyFrame): Underlying data containing (date_column, [ASSETS])
        date_column (str): The time column by name
        factors (list[str]): A list of factor column names
        assets (list[str]): A list of asset names
        periods (list[int]): The periods to calculate forward returns for
        log (bool): Return log returns instead of simple returns. Default: True

    Returns:
        pl.LazyFrame | pl.DataFrame: Dataframe containing the log forward returns
    """
    if log:
        forward_return_expressions = [
            pl.col(a).log().diff().shift(-i).name.suffix(f"_{i}")
            for a in assets
            for i in periods
        ]
    else:
        forward_return_expressions = [
            pl.col(a).pct_change().shift(-i).name.suffix(f"_{i}")
            for a in assets
            for i in periods
        ]
    return prices.select(pl.col(date_column), *forward_return_expressions)
