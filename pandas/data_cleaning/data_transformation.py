import pandas as pd

# Create a sample DataFrame
data = {
    "Name": ["John", "Emma", "Peter", "Olivia"],
    "Age": [25, 30, 35, 40],
    "Salary": [50000, 60000, 70000, 80000],
}
df = pd.DataFrame(data)

# Apply data transformation methods
df["Age"] = df["Age"] + 5  # Increase age by 5 years
df["Salary"] = df["Salary"] * 1.1  # Increase salary by 10%

# Rename columns
df = df.rename(columns={"Name": "Full Name", "Salary": "Annual Salary"})

# Drop a column
df = df.drop("Age", axis=1)

# Sort the DataFrame by a column
df = df.sort_values("Full Name")

# Reset the index
df = df.reset_index(drop=True)

# Display the transformed DataFrame
print(df)
