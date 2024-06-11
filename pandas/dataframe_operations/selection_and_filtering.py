import pandas as pd

# Create a sample DataFrame
data = {
    "Name": ["John", "Emma", "Peter", "Emily"],
    "Age": [25, 30, 35, 28],
    "City": ["New York", "London", "Paris", "Sydney"],
}
df = pd.DataFrame(data)

# Selecting a single column
name_column = df["Name"]
print(name_column)

# Selecting multiple columns
name_age_columns = df[["Name", "Age"]]
print(name_age_columns)

# Filtering rows based on a condition
filtered_df = df[df["Age"] > 30]
print(filtered_df)
