import pandas as pd
import polars as pl

from pqf.pricing.utils import estimate_time_grain


class PricingData:
    """PricingData provides a structured interface for handling and aggregating pricing bar data using Polars DataFrames.

    Attributes:
        trade_date_col (str): Column name for trade date.
        timestamp_col (str): Column name for timestamp.
        open_col (str): Column name for open price.
        high_col (str): Column name for high price.
        low_col (str): Column name for low price.
        close_col (str): Column name for close price.
        volume_col (str): Column name for volume.
        financial_inst_id_col (str): Column name for financial instrument ID.
        output_schema (pl.Schema): Schema used to validate and cast the input data.
        data (pl.DataFrame | pl.LazyFrame): The processed pricing data.
        time_grain (str): Estimated time grain of the data.

    Methods:
        get_bars() -> pl.DataFrame | pl.LazyFrame:
            Returns the processed pricing bars data.
        get_aggregated_bars(freq: str) -> pl.DataFrame | pl.LazyFrame:
            Returns the pricing bars data aggregated to the specified frequency.
    Raises:
        Any exceptions raised by Polars during schema matching or data processing."""

    def __init__(
        self,
        data_source: pl.DataFrame | pl.LazyFrame,
        lazy: bool = True,
        trade_date_col: str = "trade_date",
        timestamp_col: str = "end_dtutc",
        financial_inst_id_col: str = "fid",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",

    ):

        self.trade_date_col = trade_date_col
        self.timestamp_col = timestamp_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.financial_inst_id_col = financial_inst_id_col

        self.output_schema = pl.Schema({
            self.trade_date_col: pl.Date,
            self.timestamp_col: pl.Datetime(time_unit="us"),
            self.financial_inst_id_col: pl.Int64,
            self.open_col: pl.Float64,
            self.high_col: pl.Float64,
            self.low_col: pl.Float64,
            self.close_col: pl.Float64,
            self.volume_col: pl.Float64,
        })

        if lazy:
            self.data = data_source.lazy().match_to_schema(self.output_schema)
        else:
            self.data = data_source.match_to_schema(self.output_schema)

        self.time_grain = estimate_time_grain(
            self.data.select(self.timestamp_col).lazy().collect().to_series())

    def get_bars(self) -> pl.DataFrame | pl.LazyFrame:
        """Get the pricing bars data.

        Returns:
            pl.DataFrame | pl.LazyFrame: The pricing bars data.
        """
        return self.data

    def get_aggregated_bars(
        self,
        freq: str
    ) -> pl.DataFrame | pl.LazyFrame:
        """Get aggregated pricing bars data.

        Args:
            freq (str): The frequency to aggregate to (e.g., '1h', '1d').


        Returns:
            pl.DataFrame | pl.LazyFrame: The aggregated pricing bars data.
        """

        if isinstance(self.data, pl.LazyFrame):
            df = self.data.collect()
        else:
            df = self.data
        df_out = df.sort(
            [self.trade_date_col, self.timestamp_col, self.financial_inst_id_col]).group_by_dynamic(
            index_column=self.timestamp_col,
            group_by=[self.financial_inst_id_col, self.trade_date_col],
            closed="left",
            label="left",
            every=freq,
        ).agg([
            pl.col(self.open_col).first().alias(self.open_col),
            pl.col(self.high_col).max().alias(self.high_col),
            pl.col(self.low_col).min().alias(self.low_col),
            pl.col(self.close_col).last().alias(self.close_col),
            pl.col(self.volume_col).sum().alias(self.volume_col)
        ])
        return df_out
