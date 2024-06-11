import pandas as pd

# Create a DataFrame from a dictionary
data = {
    "Name": ["John", "Emma", "Michael", "Sophia"],
    "Age": [25, 28, 32, 30],
    "City": ["New York", "London", "Paris", "Tokyo"],
}
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Accessing columns
print(df["Name"])  # Accessing a single column
print(df[["Name", "Age"]])  # Accessing multiple columns

# Accessing rows
print(df.loc[0])  # Accessing a single row by label
print(df.iloc[2])  # Accessing a single row by index

# Filtering rows
filtered_df = df[df["Age"] > 28]
print(filtered_df)

# Adding a new column
df["Gender"] = ["Male", "Female", "Male", "Female"]
print(df)

# Dropping columns
df = df.drop("City", axis=1)
print(df)

# Updating values
df.loc[1, "Age"] = 29
print(df)
