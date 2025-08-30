# NLP_SentimentAnalysis
This project analyzes user reviews and predicts sentiment (Positive, Neutral, Negative)using Machine Learning (XGBoost, Logistic Regression, Random Forest, Naive Bayes) and Deep Learning (LSTM, GRU, BERT, DistilBERT) models..It also provides Exploratory Data Analysis (EDA) Insights like trends, word clouds, and sentiment comparisons.The goal is to gain insights into customer satisfaction, identify common concerns, and enhance the application's user experience.
Built with ❤️ using Python, scikit-learn, TensorFlow/Keras, Hugging Face Transformers, and Streamlit.

## 🚀 Features

✅ Real-time Sentiment Prediction (Positive, Neutral, Negative)                                                                                                                            
✅ EDA Dashboard with 10 key insights                                                                                                                                                      
✅ 📈 Visualizations using Matplotlib & Seaborn                                                                                                                                           
✅ 🌍 WordClouds for each sentiment                                                                                                                                                       
✅ 🔮 Model trained with TF-IDF + XGBoost                                                                                                                                                 
✅ ⚡ Easy-to-use Streamlit App                                                                                                                                                           

## 🛠️ Tech Stack

- **Programming Language**: Python (Pandas, NumPy, Matplotlib, Seaborn)  
- **Natural Language Processing (NLP)**: NLTK, Scikit-learn (TF-IDF, preprocessing)  
- **Machine Learning Models**: Logistic Regression, Naive Bayes, Random Forest, XGBoost  
- **Deep Learning Models**: LSTM, GRU, Transformer-based (BERT / DistilBERT)  
- **Visualization & Dashboard**: Streamlit (Interactive Dashboard), WordCloud  
- **Deployment**: Streamlit App (Local/Cloud), GitHub

## 📊 Sentiment Analysis Insights  

### 1. Overall Sentiment Distribution  
- Positive, Neutral, Negative proportions  

### 2. Sentiment vs Ratings  
- Detect mismatches between star ratings & review text  

### 3. Keyword & Phrase Associations  
- Word clouds & frequent terms per sentiment  

### 4. Sentiment Trend Over Time  
- Monthly/weekly changes in satisfaction or dissatisfaction  

### 5. Verified vs Non-Verified Users  
- Impact of purchase verification on sentiment  

### 6. Review Length vs Sentiment  
- Do longer reviews lean more negative or positive?  

### 7. Location-based Sentiment  
- Regional analysis of user experiences  

### 8. Platform-wise Sentiment  
- Comparison between Web vs Mobile reviews  

### 9. ChatGPT Version Sentiment  
- Effect of different versions on user satisfaction  

### 10. Negative Feedback Themes  
- Topic modeling & recurring pain points  

## 📂 Project Structure

├── 📁 data/  
│   ├── chatgpt_style_reviews_dataset.xlsx - Sheet1.csv              # Raw input dataset  
│   ├── preprocessed_reviews.csv                                     # Cleaned + processed dataset  

├── 📁 notebooks/  
│   ├── NLP_eda.ipynb                # Exploratory Data Analysis (EDA)  
│   ├── sentiment_models.ipynb       # Machine Learning model training + LSTM / GRU / BERT experiments
  
├── 📁 models/  
│   ├── sentiment_model.pkl          # Saved ML/DL model  
│   ├── tfidf_vectorizer.pkl         # TF-IDF vectorizer  
│   ├── label_encoder.pkl            # Encoded labels  

├── 📁 app/  
│   ├── NLPapp.py                    # Streamlit dashboard app  

├── 📁 visuals/  
│   ├── sentiment_wordcloud.png      # Wordcloud sample  
│   ├── confusion_matrix.png         # Model confusion matrix  
│   ├── sentiment_trend.png          # Sentiment trend over time  

├── requirements.txt                 # Python dependencies  
├── README.md                        # Project documentation  

### 📸 Screenshots
<img width="1366" height="768" alt="Screenshot (64)" src="https://github.com/user-attachments/assets/4b51eda0-8b74-47c6-88a7-af87da05ab1b" />
<img width="1366" height="768" alt="Screenshot (65)" src="https://github.com/user-attachments/assets/6e09e7a7-dc10-45f5-ac7a-67ce249050d3" />
<img width="1366" height="768" alt="Screenshot (66)" src="https://github.com/user-attachments/assets/2ae68d7b-524a-409b-b4a9-f7c2929c6c01" />
<img width="1366" height="768" alt="Screenshot (67)" src="https://github.com/user-attachments/assets/2f03b2ec-04b5-42ef-a25b-ed0d5f35e9ff" />
<img width="1366" height="768" alt="Screenshot (68)" src="https://github.com/user-attachments/assets/ce419e70-666f-4e0e-9f29-97ed63de77fb" />
<img width="1366" height="768" alt="Screenshot (69)" src="https://github.com/user-attachments/assets/c868deb7-c896-4de6-a851-89a9f3cf5806" />
<img width="1366" height="768" alt="Screenshot (70)" src="https://github.com/user-attachments/assets/887a52fa-52b7-439c-bd60-f700719b9a9f" />


## 🚀 How to Run

Follow these steps to set up and run the project locally in VS Code with Streamlit:

### 1️⃣ Clone the Repository
git clone https://github.com/ThilagavathyVenkatesan/NLP_SentimentAnalysis.git                                                                                                             
cd NLP_SentimentAnalysis

### 2️⃣ Create a Virtual Environment
## Windows
python -m venv venv
venv\Scripts\activate

## macOS/Linux
python3 -m venv venv
source venv/bin/activate

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Prepare Data & Models

1. **Place datasets inside the `/data` folder**
   - `raw_reviews.csv`
   - `preprocessed_reviews.csv`

2. **Place trained models and vectorizers inside the `/models` folder**
   - `sentiment_model.pkl`
   - `tfidf_vectorizer.pkl`
   - `label_encoder.pkl`

### 5️⃣ Run the Streamlit App
```bash
streamlit run NLPapp.py

# Open in Browser
By default, the app runs at:

👉 [http://localhost:8501](http://localhost:8501)

Use the sidebar to switch between:

- 📊 **EDA Insights**  
- 🔮 **Sentiment Prediction**
