import pandas as pd

# Create a sample time series data
data = {
    "date": pd.date_range(start="1/1/2022", periods=10, freq="D"),
    "value": [10, 15, 12, 18, 20, 17, 22, 25, 23, 28],
}
df = pd.DataFrame(data)

# Calculate rolling mean
df["rolling_mean"] = df["value"].rolling(window=3).mean()

# Calculate rolling standard deviation
df["rolling_std"] = df["value"].rolling(window=3).std()

# Calculate rolling sum
df["rolling_sum"] = df["value"].rolling(window=3).sum()

# Print the DataFrame
print(df)
