from pqf.pricing.utils import estimate_time_grain
from pqf.pricing.core import PricingData
import polars as pl
import polars.testing as plt
from datetime import date, timedelta, datetime


class TestTimeGrainEstimation:
    def test_daily_accepts_time_series(self):
        dates = pl.Series("dates", [
            "2023-01-01 00:00:00",
            "2023-01-02 00:00:00",
            "2023-01-03 00:00:00",
        ]).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")

        # This should not throw an error
        time_grain = estimate_time_grain(dates)
        assert time_grain == timedelta(days=1)

    def test_hourly_accepts_time_series(self):
        dates = pl.Series("dates", [
            "2023-01-01 00:00:00",
            "2023-01-01 01:00:00",
            "2023-01-01 02:00:00",
        ]).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")

        # This should not throw an error
        time_grain = estimate_time_grain(dates)
        assert time_grain == timedelta(hours=1)

    def test_minutely_accepts_time_series(self):
        dates = pl.Series("dates", [
            "2023-01-01 00:00:00",
            "2023-01-01 00:01:00",
            "2023-01-01 00:02:00",
        ]).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")

        # This should not throw an error
        time_grain = estimate_time_grain(dates)
        assert time_grain == timedelta(minutes=1)

    def test_30min_accepts_time_series(self):
        dates = pl.Series("dates", [
            "2023-01-01 00:00:00",
            "2023-01-01 00:30:00",
            "2023-01-01 01:00:00",
        ]).str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")

        # This should not throw an error
        time_grain = estimate_time_grain(dates)
        assert time_grain == timedelta(minutes=30)


class TestPricingDataAggregation:
    def setup_method(self):
        """Set up test data for aggregation tests."""
        # Create minute-level OHLCV data matching the expected schema
        self.test_data = pl.DataFrame({
            "trade_date": [date(2025, 1, 1)] * 6,
            "end_dtutc": [
                datetime(2025, 1, 1, 0, 0, 0),
                datetime(2025, 1, 1, 0, 1, 0),
                datetime(2025, 1, 1, 0, 2, 0),
                datetime(2025, 1, 1, 0, 3, 0),
                datetime(2025, 1, 1, 0, 4, 0),
                datetime(2025, 1, 1, 0, 5, 0),
            ],
            "fid": [1] * 6,
            "open": [100.0, 101.0, 102.0, 101.5, 103.0, 102.0],
            "high": [101.5, 102.5, 103.0, 102.0, 104.0, 103.0],
            "low": [99.5, 100.5, 101.0, 100.5, 102.5, 101.0],
            "close": [101.0, 102.0, 101.5, 103.0, 102.0, 102.5],
            "volume": [1000.0, 1200.0, 800.0, 1500.0, 900.0, 1100.0],
        })

    def test_aggregated_bars_5min_frequency(self):
        """Test aggregation to 5-minute bars."""
        pricing_data = PricingData(self.test_data)
        aggregated = pricing_data.get_aggregated_bars("5m")

        # Should aggregate 6 minutes of data into 2 bars:
        # Bar 1: minutes 0-4 (first 5 minutes)
        # Bar 2: minute 5 (just the 6th minute)
        assert aggregated.height == 2

        # Check first aggregated bar (minutes 0-4)
        first_bar = aggregated.row(0, named=True)
        assert first_bar["open"] == 100.0  # First open
        # Max high across first 5 minutes
        assert first_bar["high"] == 104.0
        assert first_bar["low"] == 99.5    # Min low across first 5 minutes
        assert first_bar["close"] == 102.0  # Last close of first 5 minutes
        # Sum volume of first 5 minutes
        assert first_bar["volume"] == 5400.0

    def test_aggregated_bars_hourly_frequency(self):
        """Test aggregation to hourly bars."""
        pricing_data = PricingData(self.test_data)
        aggregated = pricing_data.get_aggregated_bars("1h")

        # All 6 minutes should aggregate into 1 hour bar
        assert aggregated.height == 1

        bar = aggregated.row(0, named=True)
        assert bar["open"] == 100.0   # First open
        assert bar["high"] == 104.0   # Max high
        assert bar["low"] == 99.5     # Min low
        assert bar["close"] == 102.5  # Last close
        assert bar["volume"] == 6500.0  # Sum of all volume

    def test_aggregated_bars_with_lazy_frame(self):
        """Test aggregation works with LazyFrame input."""
        pricing_data = PricingData(self.test_data.lazy(), lazy=True)
        aggregated = pricing_data.get_aggregated_bars("5m")

        # Should work the same as DataFrame input
        assert aggregated.height == 2

        # Verify the aggregation logic is correct
        first_bar = aggregated.row(0, named=True)
        assert first_bar["open"] == 100.0
        assert first_bar["volume"] == 5400.0

    def test_aggregated_bars_preserves_schema(self):
        """Test that aggregated bars preserve expected data types."""
        pricing_data = PricingData(self.test_data)
        aggregated = pricing_data.get_aggregated_bars("5m")

        # Check that schema matches expectations
        schema = aggregated.schema
        assert schema["trade_date"] == pl.Date
        assert schema["end_dtutc"] == pl.Datetime(time_unit="us")
        assert schema["fid"] == pl.Int64
        assert schema["open"] == pl.Float64
        assert schema["high"] == pl.Float64
        assert schema["low"] == pl.Float64
        assert schema["close"] == pl.Float64
        assert schema["volume"] == pl.Float64

    def test_aggregated_bars_multi_symbol(self):
        """Test aggregation with multiple fids (financial instruments)."""
        multi_fid_data = pl.DataFrame({
            "trade_date": [date(2025, 1, 1)] * 4,
            "end_dtutc": [
                datetime(2025, 1, 1, 0, 0, 0),
                datetime(2025, 1, 1, 0, 1, 0),
                datetime(2025, 1, 1, 0, 0, 0),
                datetime(2025, 1, 1, 0, 1, 0),
            ],
            "fid": [1, 1, 2, 2],  # Two different financial instruments
            "open": [100.0, 101.0, 2000.0, 2010.0],
            "high": [101.0, 102.0, 2020.0, 2030.0],
            "low": [99.0, 100.0, 1990.0, 2000.0],
            "close": [101.0, 102.0, 2010.0, 2020.0],
            "volume": [1000.0, 1200.0, 500.0, 600.0],
        })

        pricing_data = PricingData(multi_fid_data)
        aggregated = pricing_data.get_aggregated_bars("5m")

        # Should have one bar per fid (financial instrument)
        assert aggregated.height == 2

        # Check that both fids are represented
        fids = aggregated.select("fid").to_series().to_list()
        assert 1 in fids
        assert 2 in fids

    def test_forward_returns_with_one_period(self):
        """Test calculation of forward returns."""
        pricing_data = PricingData(self.test_data)
        returns = pricing_data.get_forward_returns(
            periods=[1], log=False).collect()

        # Should have same number of rows as input data
        assert returns.height == self.test_data.height

        expected_returns = self.test_data.select(
            pl.col('close').pct_change(n=1).shift(-1).alias('forward_return_1')).to_series()

        plt.assert_series_equal(returns['forward_return_1'], expected_returns)

    def test_forward_returns_with_two_periods(self):
        """Test calculation of forward returns."""
        pricing_data = PricingData(self.test_data)
        returns = pricing_data.get_forward_returns(
            periods=[1, 5], log=False).collect()

        # Should have same number of rows as input data
        assert returns.height == self.test_data.height

        expected_returns_1 = self.test_data.select(
            pl.col('close').pct_change(n=1).shift(-1).alias('forward_return_1')).to_series()
        expected_returns_5 = self.test_data.select(
            pl.col('close').pct_change(n=5).shift(-5).alias('forward_return_5')).to_series()

        plt.assert_series_equal(
            returns['forward_return_1'], expected_returns_1)
        plt.assert_series_equal(
            returns['forward_return_5'], expected_returns_5)
