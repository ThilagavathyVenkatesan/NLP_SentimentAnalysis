import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pickle

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
@st.cache_resource
def load_model_and_vectorizer():
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# ----------------------------
# Load Data (must be SAME preprocessing as training!)
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("preprocessed_reviews.csv")

df = load_data()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["EDA Insights", "Predict Sentiment"])


# =======================
# PART 1: EDA Insights
# =======================
if menu == "EDA Insights":
    st.header("Exploratory Insights on User Reviews")

    # Q1: Overall Sentiment
    st.subheader("1️⃣ Overall Sentiment of Reviews")
    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    st.bar_chart(sentiment_counts)

    # Q2: Sentiment vs Rating
    st.subheader("2️⃣ Sentiment vs Rating")
    fig, ax = plt.subplots()
    sns.countplot(x="rating", hue="sentiment", data=df, ax=ax)
    st.pyplot(fig)

    # Q3: Keywords per Sentiment
    st.subheader("3️⃣ Keywords Associated with Each Sentiment")
    sent_choice = st.selectbox("Choose Sentiment", df['sentiment'].unique())
    text = " ".join(df[df['sentiment'] == sent_choice]['review_clean'])
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    st.image(wc.to_array())

    # Q4: Sentiment Trend Over Time
    st.subheader("4️⃣ Sentiment Over Time")
    df['date'] = pd.to_datetime(df['date'], errors="coerce")
    trend = df.groupby(df['date'].dt.to_period("M"))['sentiment'].value_counts(normalize=True).unstack().fillna(0)
    st.line_chart(trend)

    # Q5: Verified vs Non-Verified
    st.subheader("5️⃣ Verified vs Non-Verified Users")
    fig, ax = plt.subplots()
    sns.countplot(x="verified_purchase", hue="sentiment", data=df, ax=ax)
    st.pyplot(fig)

    # Q6: Review Length vs Sentiment
    st.subheader("6️⃣ Review Length vs Sentiment")
    df['review_length'] = df['review'].str.split().str.len()
    fig, ax = plt.subplots()
    sns.boxplot(x="sentiment", y="review_length", data=df, ax=ax)
    st.pyplot(fig)

    # Q7: Location-based Sentiment
    st.subheader("7️⃣ Location-based Sentiment")
    loc_sent = df.groupby("location")['sentiment'].value_counts(normalize=True).unstack().fillna(0)
    st.bar_chart(loc_sent)

    # Q8: Platform-based Sentiment
    st.subheader("8️⃣ Platform-based Sentiment")
    plat_sent = df.groupby("platform")['sentiment'].value_counts(normalize=True).unstack().fillna(0)
    st.bar_chart(plat_sent)

    # Q9: Version-based Sentiment
    st.subheader("9️⃣ Sentiment by ChatGPT Version")
    ver_sent = df.groupby("version")['sentiment'].value_counts(normalize=True).unstack().fillna(0)
    st.bar_chart(ver_sent)

    # Q10: Negative Feedback Themes
    st.subheader("🔟 Common Negative Feedback Themes")
    neg_text = " ".join(df[df['sentiment'] == "Negative"]['review_clean'])
    wc_neg = WordCloud(width=800, height=400, background_color="black", colormap="Reds").generate(neg_text)
    st.image(wc_neg.to_array())

# ====================
# PART 2: Prediction
# ====================
elif menu == "Predict Sentiment":
    st.title("🔮 Predict Sentiment from New Review")
    user_input = st.text_area("✍️ Write your review here:")

    if st.button("🔍 Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a review")
        else:
            # Transform input
            X_input = vectorizer.transform([user_input])  

            # Predict numeric label
            pred_num = model.predict(X_input)[0]

            # Load label encoder
            with open("label_encoder.pkl", "rb") as f:
                le = pickle.load(f)

            # Convert back to original sentiment
            pred_label = le.inverse_transform([pred_num])[0]

            # Show in words with icons
            if pred_label == "Positive":
                st.success(f"✅ Predicted Sentiment: **{pred_label}** 🙂")
            elif pred_label == "Negative":
                st.error(f"❌ Predicted Sentiment: **{pred_label}** 😡")
            else:
                st.info(f"ℹ️ Predicted Sentiment: **{pred_label}** 😐")

