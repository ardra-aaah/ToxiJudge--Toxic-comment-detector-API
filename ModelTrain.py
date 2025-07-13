# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv('train.csv')  # From Jigsaw Toxic dataset
df['comment_text'] = df['comment_text'].fillna('')  # Fill missing

# For simplicity: toxic (1) if any toxic label is 1, else not toxic (0)
df['toxic_flag'] = df[['toxic', 'severe_toxic', 'obscene', 
                       'threat', 'insult', 'identity_hate']].max(axis=1)

# Split
X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['toxic_flag'], test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('clf', LogisticRegression(solver='liblinear'))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save
joblib.dump(pipeline, 'toxic_comment_model.pkl')
print("Model saved as toxic_comment_model.pkl")
