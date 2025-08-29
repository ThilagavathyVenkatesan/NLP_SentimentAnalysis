# step1_setup_data_understanding.py

import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv(
    "C:/Users/Modern/Downloads/chatgpt_style_reviews_dataset.xlsx - Sheet1.csv",
    low_memory=False
)

print("✅ Dataset loaded:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Standardize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# 3. Check for missing values
missing_summary = df.isna().sum()
print("\n🔎 Missing Values per Column:\n", missing_summary)

# 4. Check for duplicates
duplicates = df.duplicated().sum()
print("\n📌 Duplicate Rows:", duplicates)

# 5. Sentiment labeling (based on rating)
def map_sentiment(r):
    try:
        r = float(r)
    except:
        return np.nan
    if r >= 4:
        return "Positive"
    elif r <= 2:
        return "Negative"
    elif r == 3:
        return "Neutral"
    return np.nan

df["sentiment"] = df["rating"].apply(map_sentiment)

# 6. Check class balance
print("\n⚖️ Rating Distribution:\n", df["rating"].value_counts().sort_index())
print("\n⚖️ Sentiment Distribution:\n", df["sentiment"].value_counts())

# 7. Other useful summaries
print("\n🌍 Platform Counts:\n", df["platform"].value_counts())
print("\n👤 Verified Purchase Counts:\n", df["verified_purchase"].value_counts())

# 8. Save a processed copy
df.to_csv("processed_reviews.csv", index=False)
print("\n💾 Processed dataset saved as processed_reviews.csv")
