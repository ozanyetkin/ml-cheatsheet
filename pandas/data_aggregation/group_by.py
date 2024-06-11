import pandas as pd

# Create a sample DataFrame
data = {
    "Name": ["John", "Emily", "John", "Emily", "John"],
    "Subject": ["Math", "Math", "Science", "Science", "Science"],
    "Score": [90, 85, 92, 88, 95],
}
df = pd.DataFrame(data)

# Group the DataFrame by 'Name' and 'Subject'
grouped = df.groupby(["Name", "Subject"])

# Calculate the mean score for each group
mean_scores = grouped["Score"].mean()

# Calculate the maximum score for each group
max_scores = grouped["Score"].max()

# Calculate the minimum score for each group
min_scores = grouped["Score"].min()

# Print the results
print("Mean Scores:")
print(mean_scores)
print("\nMax Scores:")
print(max_scores)
print("\nMin Scores:")
print(min_scores)
