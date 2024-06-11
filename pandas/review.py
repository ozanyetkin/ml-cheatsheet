import pandas as pd

# Create a DataFrame
data = {
    "Name": ["John", "Emma", "Peter", "Olivia"],
    "Age": [25, 28, 32, 30],
    "City": ["New York", "London", "Paris", "Tokyo"],
}
df = pd.DataFrame(data)

# Display the DataFrame
print("Original DataFrame:")
print(df)

# Selecting data
print("\nSelecting data:")
print(df["Name"])  # Select a single column
print(df[["Name", "Age"]])  # Select multiple columns
print(df.loc[1])  # Select a row by label
print(df.iloc[2])  # Select a row by index

# Filtering data
print("\nFiltering data:")
print(df[df["Age"] > 28])  # Filter rows based on condition

# Sorting data
print("\nSorting data:")
print(df.sort_values("Age"))  # Sort DataFrame by column

# Adding and removing columns
print("\nAdding and removing columns:")
df["Gender"] = ["Male", "Female", "Male", "Female"]  # Add a new column
print(df)
df = df.drop("City", axis=1)  # Remove a column
print(df)

# Aggregating data
print("\nAggregating data:")
print(df["Age"].mean())  # Calculate mean of a column
print(df.groupby("Gender")["Age"].mean())  # Group by column and calculate mean

# Handling missing data
print("\nHandling missing data:")
df.loc[2, "Age"] = None  # Set a value as missing
print(df)
print(df.dropna())  # Drop rows with missing values
print(df.fillna(0))  # Fill missing values with a specific value

# Saving and loading data
print("\nSaving and loading data:")
df.to_csv("data.csv", index=False)  # Save DataFrame to CSV file
df = pd.read_csv("data.csv")  # Load DataFrame from CSV file
print(df)
