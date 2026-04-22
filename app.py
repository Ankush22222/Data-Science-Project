from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# Extract keywords
def extract_keywords(text):
    words = text.split()
    return ", ".join(words[:5])

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]

    cleaned = clean_text(review)
    words = cleaned.split()

    positive_words = ["good", "excellent", "amazing", "nice", "liked"]
    negative_words = ["not", "bad", "poor", "worst", "disappointing", "useless"]

    pos_count = sum(word in words for word in positive_words)
    neg_count = sum(word in words for word in negative_words)

    # 🔥 STEP 1: NEGATION FIX
    negation_phrases = ["not good", "not nice", "not worth", "not recommended", "not satisfied"]

    if any(phrase in cleaned for phrase in negation_phrases):
        sentiment = "Negative 😞"
        decision = "Not Recommended ❌"
        trust = "High Confidence (Negation Detected)"
        confidence = 85

    # 🔥 STEP 2: MIXED DETECTION
    elif pos_count > 0 and neg_count > 0:
        sentiment = "Mixed 🤔"
        decision = "Needs Review ⚠️"
        trust = "Balanced Sentiment"
        confidence = 70

    else:
        # 🔥 STEP 3: BUT LOGIC
        important_text = cleaned
        if "but" in cleaned:
            important_text = cleaned.split("but")[-1]

        vec = vectorizer.transform([important_text])
        result = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        confidence = round(max(proba) * 100, 2)

        # 🔥 STEP 4: NEGATIVE OVERRIDE
        if neg_count >= 1:
            sentiment = "Negative 😞"
            decision = "Not Recommended ❌"
            trust = "High Confidence (Rule Based)"
        else:
            if result == 1:
                sentiment = "Positive 😊"
                decision = "Recommended ✅"
            else:
                sentiment = "Negative 😞"
                decision = "Not Recommended ❌"

            trust = "High Confidence" if confidence > 85 else "Moderate Confidence"

    # Final outputs
    keywords = extract_keywords(cleaned)
    length = len(review.split())

    return render_template("index.html",
                           sentiment=sentiment,
                           decision=decision,
                           confidence=confidence,
                           length=length,
                           keywords=keywords,
                           trust=trust)

# Run app
if __name__ == "__main__":
    app.run(debug=True)