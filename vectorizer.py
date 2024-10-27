import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
import joblib

# Load the true news dataset
true_df = pd.read_csv("true_tokenized.csv")
true_df.columns = true_df.columns.str.replace(r"[\[\]']", "", regex=True).str.strip()
true_df['label'] = 0  # Assign label 0 for true news

# Load the fake news dataset
fake_df = pd.read_csv("fake_tokenized.csv")  # Adjust this path as necessary
fake_df.columns = fake_df.columns.str.replace(r"[\[\]']", "", regex=True).str.strip()
fake_df['label'] = 1  # Assign label 1 for fake news

# Combine both datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Define target variable
y = df['label']  # Use the new 'label' column for target labels

# Initialize vectorizers for each column, limiting features to reduce memory usage
title_vectorizer = TfidfVectorizer(max_features=300000)  # Adjust max_features based on memory availability
text_vectorizer = TfidfVectorizer(max_features=300000)

# Fit vectorizers on the combined data
title_vectorizer.fit(df['title'])
text_vectorizer.fit(df['text'])

# Transform the 'title' and 'text' columns separately
X_title = title_vectorizer.transform(df['title'])
X_text = text_vectorizer.transform(df['text'])

# Combine the title and text features as a sparse matrix
X = hstack([X_title, X_text])

# Split the combined features and labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model and vectorizers
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(title_vectorizer, 'title_vectorizer.pkl')
joblib.dump(text_vectorizer, 'text_vectorizer.pkl')

# Display the first few features without converting to dense format
print("Title Features (sample):\n", X_title[:5])
print("Text Features (sample):\n", X_text[:5])
