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


def sortino_ratio(
    returns: pl.Series | pl.Expr, risk_free_rate: float
) -> float | pl.Expr | None:
    """Calculate the Sortino Ratio.

    Args:
        returns (pl.Series | pl.Expr): Series or expression representing returns.
        risk_free_rate (float): The risk-free rate.

    Raises:
        TypeError: If downside deviation cannot be calculated.

    Returns:
        float | pl.Expr | None: The calculated Sortino ratio or None if not calculable.
    """
    if isinstance(returns, pl.Series):
        if returns.mean() is None:
            return None
        mean_excess_return = returns.mean() - risk_free_rate

        negative_returns = returns.filter(returns < 0)
        if negative_returns.is_empty():
            return None

        downside_deviation = negative_returns.std()
        if downside_deviation is None or downside_deviation == 0:
            raise TypeError("Could not calculate downside deviation of returns.")

        sortino = mean_excess_return / downside_deviation

    else:
        mean_excess_return = returns.mean() - risk_free_rate

        negative_returns_expr = pl.when(returns < 0).then(returns).otherwise(None)

        downside_deviation_expr = negative_returns_expr.std().cast(pl.Float64)

        sortino = (
            pl.when(downside_deviation_expr != 0)
            .then(mean_excess_return / downside_deviation_expr)
            .otherwise(None)
        )

    return sortino


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


def annualized_returns(returns: pl.Series | pl.Expr) -> pl.Series | pl.Expr | None:
    """
    Calculate the annualized return from a series of daily log returns.

    The annualized return is calculated using the formula:
        e^(sum of log returns * 365 / period of days) - 1

    Args:
        returns (pl.Series | pl.Expr): Series or expression representing daily log returns.

    Returns:
        pl.Series | pl.Expr | None: A series of the cumulative annualized returns or None if not calculable.
    """

    if isinstance(returns, pl.Series):
        period = returns.count()
        if period <= 1:
            return None

        cumulative_log_returns = returns.cum_sum()

        annualization_factor = 365 / period
        annualized = (cumulative_log_returns * annualization_factor).exp() - 1

    else:
        period_expr = returns.count()
        annualized = (
            returns.cum_sum().mul(pl.lit(365).truediv(period_expr)).exp().sub(1)
        )

    return annualized
