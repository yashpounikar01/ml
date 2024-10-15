# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels (target)

# Step 1: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA for dimensionality reduction (reduce to 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 3: Visualize the result
plt.figure(figsize=(8, 6))
for i, target_name in zip([0, 1, 2], iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset (2 components)')
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio for each component:", explained_variance)

# Step 5: Describe the plot in the terminal
total_variance = np.sum(explained_variance) * 100
print(f"\nGraph Description:")
print(f"The scatter plot shows the Iris dataset reduced to 2 principal components.")
print(f"Each point represents a sample from the Iris dataset.")
print(f"Three species are represented: Setosa (purple), Versicolor (yellow), and Virginica (green).")
print(f"Principal Component 1 explains {explained_variance[0]*100:.2f}% of the variance, while Principal Component 2 explains {explained_variance[1]*100:.2f}%.")
print(f"Together, these two components capture {total_variance:.2f}% of the total variance in the dataset.")
print(f"\nYou should see distinct clusters for Setosa, with some overlap between Versicolor and Virginica.")
