# data_preprocessing.py

import pandas as pd
import numpy as np
import re
import nltk

from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------------------
# 1. Download necessary NLTK resources (only needed on first run)
# -------------------------------------------------------------------
try:
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("punkt_tab")   # fixes your error
    nltk.download("wordnet")
except:
    print("⚠️ Warning: Could not download some NLTK resources.")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------------------------------------------------
# 2. Text Cleaning Function
# -------------------------------------------------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()                        # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"<.*?>", "", text)               # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)           # keep only letters

    try:
        tokens = nltk.word_tokenize(text)           # tokenize
    except LookupError:
        tokens = text.split()                       # fallback if tokenizer missing

    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# -------------------------------------------------------------------
# 3. Sentiment Labeling Function
# -------------------------------------------------------------------
def map_sentiment(r):
    try:
        r = int(r)
    except:
        return np.nan
    if r >= 4:
        return "Positive"
    elif r <= 2:
        return "Negative"
    elif r == 3:
        return "Neutral"
    return np.nan

# -------------------------------------------------------------------
# 4. Preprocessing Pipeline
# -------------------------------------------------------------------
def preprocess_dataset(file_path, output_path="preprocessed_reviews.csv"):
    # Load dataset
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)

    # Handle missing values
    df = df.fillna({
        "title": "",
        "review": "",
        "platform": "Unknown",
        "language": "Unknown",
        "location": "Unknown",
        "version": "Unknown",
        "verified_purchase": "No"
    })

    # Clean review text
    df["review_clean"] = df["review"].apply(clean_text)

    # Add sentiment labels
    df["sentiment"] = df["rating"].apply(map_sentiment)

    # Encode categorical features
    enc_platform = LabelEncoder()
    enc_verified = LabelEncoder()
    enc_version = LabelEncoder()

    df["platform_encoded"] = enc_platform.fit_transform(df["platform"].astype(str))
    df["verified_encoded"] = enc_verified.fit_transform(df["verified_purchase"].astype(str))
    df["version_encoded"] = enc_version.fit_transform(df["version"].astype(str))

    # Save preprocessed dataset
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessing complete. Saved to {output_path}")
    print("\nSentiment distribution:\n", df["sentiment"].value_counts())
    return df

# -------------------------------------------------------------------
# 5. Run if executed directly
# -------------------------------------------------------------------
if __name__ == "__main__":
    preprocess_dataset("processed_reviews.csv")
