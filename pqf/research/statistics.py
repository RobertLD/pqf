import polars as pl


def sharpe_ratio(
    returns: pl.Series | pl.Expr, risk_free_rate: float
) -> float | pl.Expr | None:
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
