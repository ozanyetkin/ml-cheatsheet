import pandas as pd

# Create a sample DataFrame
data = {
    "Name": ["John", "Jane", "Mike", "Emily", "Tom"],
    "Age": [25, 30, None, 35, 40],
    "Salary": [50000, 60000, 70000, None, 80000],
}

df = pd.DataFrame(data)

# Check for missing values
print(df.isnull())

# Drop rows with missing values
df.dropna(inplace=True)

# Fill missing values with a specific value
df["Age"].fillna(0, inplace=True)

# Fill missing values with the mean of the column
df["Salary"].fillna(df["Salary"].mean(), inplace=True)

# Forward fill missing values
df.fillna(method="ffill", inplace=True)

# Backward fill missing values
df.fillna(method="bfill", inplace=True)

# Interpolate missing values
df.interpolate(inplace=True)

# Replace missing values with a specific value
df.replace(to_replace=None, value=0, inplace=True)

# Save the cleaned DataFrame to a new file
df.to_csv("/path/to/cleaned_data.csv", index=False)
