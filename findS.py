import pandas as pd
import numpy as np

# Load the dataset from the CSV file
data = pd.read_csv('data.csv')

# Print the DataFrame
print("DataFrame:")
print(data)

# Extract attributes and target values
attributes = data.iloc[:, :-1].values
target = data['Hired'].values

# Print the attributes and target values
print("\nAttributes:")
print(attributes)
print("\nTarget:")
print(target)

# Define the training function for the modified Find-S algorithm
def train(c, t):
    # Initialize the most specific hypothesis
    specific_hypothesis = list(c[t == "Yes"][0])
    
    # Iterate through each positive example
    for i, val in enumerate(t):
        if val == "Yes":
            example = c[i]
            for j in range(len(specific_hypothesis)):
                if specific_hypothesis[j] != example[j]:
                    specific_hypothesis[j] = '?'
    
    # Adjust for the 'Skills' attribute
    skills_counts = pd.Series(c[t == "Yes"][:, 2]).value_counts()
    if skills_counts.get('Intermediate', 0) > 1:
        specific_hypothesis[2] = 'Intermediate'
    
    return specific_hypothesis

# Compute the final hypothesis
final_hypothesis = train(attributes, target)

# Print the final hypothesis
print("\nThe final hypothesis is:", final_hypothesis)
