import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from email.csv
df = pd.read_csv('email.csv')

# Features and target variable
X = df['Email']  # Email content
y = df['Label']  # 1 = Spam, 0 = Non-Spam

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converting text data to numerical features using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.85)
X_train_vec = vectorizer.fit_transform(X_train)  # Training data in numerical form
X_test_vec = vectorizer.transform(X_test)        # Test data in numerical form

# Creating a Naive Bayes classifier
model = MultinomialNB()

# Training the model
model.fit(X_train_vec, y_train)

# Making predictions for the test data
y_pred = model.predict(X_test_vec)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Making predictions for the entire dataset
X_all_vec = vectorizer.transform(X)  # Transform the entire dataset (all emails)
y_all_pred = model.predict(X_all_vec)  # Predictions for all emails

# Adding predictions to the DataFrame
df['Predicted Label'] = y_all_pred

# Renaming labels for better readability (optional)
df['Actual Label'] = df['Label'].apply(lambda x: 'Spam' if x == 1 else 'Non-Spam')
df['Predicted Label'] = df['Predicted Label'].apply(lambda x: 'Spam' if x == 1 else 'Non-Spam')

# Displaying the results in tabular form using pandas' DataFrame
print("\nPredictions for the dataset (Tabular form):\n")
print(df[['Email', 'Predicted Label']].to_string(index=False))

# Saving the predictions to a new CSV file
df.to_csv('email_predictions.csv', index=False)

print("\nPredictions saved to 'email_predictions.csv'.")
