import pandas as pd

# Create a date range
start_date = "2022-01-01"
end_date = "2022-01-31"
dates = pd.date_range(start=start_date, end=end_date, freq="D")

# Create a time series with random data
data = pd.Series(range(len(dates)), index=dates)

# Print the time series
print("Original Time Series:")
print(data)

# Accessing values in the time series
print("\nAccessing Values:")
print(data["2022-01-05"])  # Access a specific date
print(data["2022-01-10":"2022-01-15"])  # Access a range of dates

# Resampling the time series
print("\nResampling:")
weekly_data = data.resample("W").sum()  # Resample to weekly frequency
print(weekly_data)

# Rolling window calculations
print("\nRolling Window Calculations:")
# Calculate rolling mean with window size of 7
rolling_mean = data.rolling(window=7).mean()
print(rolling_mean)

# Time zone conversion
print("\nTime Zone Conversion:")
# Localize time series to UTC
data_utc = data.tz_localize("UTC")
# Convert to Eastern Time Zone
data_est = data_utc.tz_convert("US/Eastern")
print(data_est)
