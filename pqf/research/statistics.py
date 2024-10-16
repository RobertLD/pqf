import polars as pl
import polars.selectors as cs


def sharpe_ratio(
    returns: pl.Series | pl.Expr, risk_free_rate: float
) -> float | pl.Expr | None:
    """Calculate the Sharpe ratio for a given series of returns and a risk-free rate.

    :param returns: A polars Series or Expression representing the returns.
    :param risk_free_rate: The risk-free rate used in the calculation.

    :return: The Sharpe ratio value as a float, Expression, or None if the calculation is not possible.
    :rtype: float | pl.Expr | None
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
    return market_constituent_returns.select(
        pl.col(date_column), pl.mean_horizontal(cs.float()).alias("market_return")
    )
