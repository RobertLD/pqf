import polars as pl
import polars.selectors as cs


def sharpe_ratio(
    returns: pl.Series | pl.Expr, risk_free_rate: float
) -> float | pl.Expr | None:
    """Calculate the Sharpe ratio.

    Args:
        returns (pl.Series | pl.Expr): Series or expression representing returns.
        risk_free_rate (float): The risk-free rate.

    Raises:
        TypeError: If standard deviation of returns cannot be calculated.

    Returns:
        float | pl.Expr | None: The calculated Sharpe ratio or None if not calculable.
    """
    if isinstance(returns, pl.Series):
        excess_returns = returns - risk_free_rate
        mean_return = excess_returns.mean()
        if mean_return is None:
            return None

        return_dist = excess_returns.std()
        if not isinstance(return_dist, float):
            raise TypeError("could not calculate std of returns")
        if return_dist is None:
            return None

        sharpe = float(mean_return / return_dist)  # type: ignore
    else:
        excess_returns = returns.sub(risk_free_rate)
        sharpe = (
            excess_returns.mean()
            .cast(pl.Float64)
            .truediv(excess_returns.std().cast(pl.Float64))
        )
    return sharpe


def estimate_market_returns(
    market_constituent_returns: pl.LazyFrame | pl.DataFrame, date_column: str
) -> pl.LazyFrame | pl.DataFrame:
    """Estimate the market returns based on the constituent returns.

    Args:
        market_constituent_returns (pl.LazyFrame | pl.DataFrame): DataFrame containing constituent returns.
        date_column (str): Name of the column containing dates.

    Returns:
        pl.LazyFrame | pl.DataFrame: DataFrame with selected date column and calculated market return.
    """
    return market_constituent_returns.select(
        pl.col(date_column), pl.mean_horizontal(cs.float()).alias("market_return")
    )


def information_coefficient(
    date_column: str,
    returns_column: str,
    returns: pl.LazyFrame | pl.DataFrame,
    factors: list[str],
    factor_returns: pl.LazyFrame | pl.DataFrame,
    window_size: int = 5,
) -> pl.LazyFrame | pl.DataFrame:
    """Calculate the rolling information coefficient for a set of factors against a benchmark return over a window size.

    Args:
        date_column (str): Datetime index column name
        returns_column (str): Column name containing benchmark returns
        returns (pl.LazyFrame): Lazyframe containing returns
        factors (list[str]): A list of factor column names
        factor_returns (pl.LazyFrame): Lazyframe containing factor returns
        window_size (int, optional): Rolling window size. Defaults to 5.

    Returns:
        pl.Lazyframe: Rolling IC for the set of factors against the benchmark
    """
    if isinstance(factors, str):
        factors = [factors]
    ic_exprs = [
        pl.rolling_corr(
            pl.col(f),
            pl.col(returns_column),
            window_size=window_size,
            min_periods=1,
        ).name.suffix("_IC")
        for f in factors
    ]
    ic = returns.join(factor_returns, on=pl.col(date_column)).select(
        pl.col(date_column), *ic_exprs
    )
    return ic
