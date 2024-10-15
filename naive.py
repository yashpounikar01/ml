import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Generating synthetic weather data, with balanced rain cases
data = {
    'Temperature': [30, 22, 25, 28, 35, 18, 15, 40, 32, 29],
    'Humidity': [70, 65, 80, 85, 90, 75, 60, 95, 88, 72],
    'Wind Speed': [10, 5, 7, 15, 20, 18, 12, 22, 14, 9],
    'Rain': [0, 1, 1, 1, 1, 1, 0, 1, 1, 0]  # Balanced rain cases
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['Temperature', 'Humidity', 'Wind Speed']]
y = df['Rain']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a Naive Bayes classifier
model = GaussianNB()

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probability of predicting rain

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Example: New weather data for prediction
new_weather = pd.DataFrame({
    'Temperature': [33, 20, 27],
    'Humidity': [78, 85, 90],  # Higher humidity to favor rain prediction
    'Wind Speed': [12, 20, 18]
})

# Getting probabilities for rain prediction
new_weather_proba = model.predict_proba(new_weather)[:, 1]  # Probability of rain (1)

# Adding predicted probabilities to the new weather DataFrame
new_weather['Predicted Rain Probability'] = new_weather_proba

# Plotting predicted rain probabilities
plt.figure(figsize=(8, 4))
new_weather.plot(kind='bar', x='Temperature', y='Predicted Rain Probability', color='blue', legend=False)
plt.title('Rain Probability Predictions Based on Weather Conditions')
plt.xlabel('Temperature')
plt.ylabel('Predicted Rain Probability')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()

print("\nNew Weather Predictions with Probabilities:")
print(new_weather)
