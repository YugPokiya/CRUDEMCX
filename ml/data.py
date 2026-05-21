import os
import pandas as pd
# 1. Correct import name for the modern ml4t-data library
from ml4t.data.providers import YahooFinanceProvider  

# Configuration
start_date = "2022-05-17"
end_date = "2026-05-17"
symbol = "CL=F"  # Ticker for Continuous WTI Crude Oil Futures
csv_filename = "C:/Users/YUG POKIYA/Download/ml4t_crude_oil_4yr.csv"

# 2. Fetch using the correct provider API
provider = YahooFinanceProvider()
print(f"Downloading 4 years of crude oil data via {provider.__class__.__name__}...")

# Fetches standardized OHLCV data matrix natively
crude_oil_data = provider.fetch_ohlcv(symbol, start=start_date, end=end_date)

# 3. Convert Polars DataFrame to Pandas to store as local CSV
if not isinstance(crude_oil_data, pd.DataFrame):
    crude_oil_df = crude_oil_data.to_pandas()
else:
    crude_oil_df = crude_oil_data

# Ensure index handles date keys correctly
crude_oil_df.to_csv(csv_filename, index=True)
print(f"Data successfully stored to: {csv_filename}")

# 4. Retrieve data back from the CSV file
retrieved_df = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
print("\n--- Retrieved CSV Data Head ---")
print(retrieved_df[['close', 'volume']].head())
