import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate a synthetic dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Create a DataFrame for easier visualization
df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
df['Target'] = y

# Display the first few rows of the dataset
print(df.head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the k-NN model with k=3
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# Fit the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Visualizing the dataset and decision boundaries
plt.figure(figsize=(10, 6))
plt.scatter(df['Feature1'], df['Feature2'], c=df['Target'], cmap='coolwarm', edgecolor='k', s=50)
plt.title('Data Points with Class Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()

# Highlight the test points
plt.scatter(X_test[:, 0], X_test[:, 1], c='yellow', edgecolor='k', s=100, label='Test Points')

plt.legend()
plt.show()

# Describing the graph
print("\nGraph Interpretation:")
print("The scatter plot shows data points color-coded by their class labels (0 in red and 1 in blue).")
print("Yellow points indicate the test instances.")
print("Observing the separation between the colors gives insight into how well the classes are distinguished.")
print("If the yellow points are surrounded mostly by one color, it suggests correct predictions.")
print("Conversely, if yellow points are surrounded by the other color, it indicates potential misclassifications.")
print("The accuracy score indicates the proportion of correctly predicted instances in the test set.")
