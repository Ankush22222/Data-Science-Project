import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("Reviews.csv")

# Select required columns
df = df[["Text", "Score"]].dropna()

# Remove neutral reviews (Score = 3)
df = df[df["Score"] != 3]

# Reduce dataset (for faster training)
df = df.sample(50000, random_state=42)

# Convert Score → Sentiment
df["sentiment"] = df["Score"].apply(lambda x: 1 if x > 3 else 0)

# TEXT PREPROCESSING (IMPORTANT AS PER PDF)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)          # remove HTML
    text = re.sub(r'http\S+|www\S+', ' ', text) # remove URLs
    text = re.sub(r'[^a-zA-Z]', ' ', text)      # keep only alphabets
    return text

df["Text"] = df["Text"].apply(clean_text)

# FEATURE EXTRACTION (TF-IDF)
vectorizer = TfidfVectorizer(
    stop_words=None, 
    max_df=0.9,
    min_df=2,
    ngram_range=(1, 2)   # unigram + bigram
)

X = vectorizer.fit_transform(df["Text"])
y = df["sentiment"]


# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# NAIVE BAYES MODEL
model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)


# EVALUATION (AS REQUIRED IN PDF)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# SAVE MODEL
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel trained and saved successfully!")