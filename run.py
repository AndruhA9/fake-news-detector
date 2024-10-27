import pandas as pd
import joblib
from scipy.sparse import hstack  # Import hstack

# Load the saved model and vectorizers
model = joblib.load('fake_news_model.pkl')
title_vectorizer = joblib.load('title_vectorizer.pkl')
text_vectorizer = joblib.load('text_vectorizer.pkl')

# Function to predict if a news article is real or fake
def predict_news(article):
    # Split the input into title and text if needed (or handle as a single input)
    # For now, we can assume the entire article is treated as the text
    X_title = title_vectorizer.transform([article])  # Treat the article as title for simplicity
    X_text = text_vectorizer.transform([article])    # You can also split this if needed

    # Combine features
    X = hstack([X_title, X_text])

    # Make prediction
    prediction = model.predict(X)

    # Map the prediction back to a label
    if prediction[0] == 0:
        return "Real"
    else:
        return "Fake"

# Get user input
user_input = input("Enter the news article text: ")
result = predict_news(user_input)
print(f"The news article is likely: {result}")
