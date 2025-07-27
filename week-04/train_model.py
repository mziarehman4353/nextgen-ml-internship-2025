# train_model.py
import pandas as pd
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv('career.csv')

# Preprocess text
def clean_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

df['clean_question'] = df['question'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_question'])
y = df['role']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'intent_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
