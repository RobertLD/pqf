import pandas as pd


class DataSourceTypes:
    PANDAS_DF = pd.DataFrame
    PARQUET_FILE = str


class PricingData:
    def __init__(
        self,
        data_source: DataSourceTypes,
        timestamp_col: str = "trade_Date",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        symbol_col: str = "symbol",
    ):
        self.data_source = data_source

        self.timestamp_col = timestamp_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.symbol_col = symbol_col
