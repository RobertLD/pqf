from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    import polars as pl

@overload
def zscore_normalize(v: pl.Series) -> pl.Series: ...
@overload
def zscore_normalize(v: pl.Expr) -> pl.Expr: ...
