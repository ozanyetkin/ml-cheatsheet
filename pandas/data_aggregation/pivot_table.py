import pandas as pd

# Create a sample dataframe
data = {
    "Name": ["John", "Jane", "John", "Jane", "John", "Jane"],
    "Subject": ["Math", "Math", "Science", "Science", "English", "English"],
    "Score": [90, 85, 95, 92, 88, 90],
}

df = pd.DataFrame(data)

# Create a pivot table
pivot_table = pd.pivot_table(df, values="Score", index="Name", columns="Subject")

print(pivot_table)
