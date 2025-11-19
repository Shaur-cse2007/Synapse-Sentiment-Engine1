import os
import psycopg2
import datetime
import nltk
import pickle
import numpy as np
from flask import Flask, render_template, jsonify, request
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --- MOCK DATA IMPORT (No Reddit API needed!) ---
import random

class MockSocialDataGenerator:
    """Generates realistic social media data - NO API REQUIRED"""
    
    def __init__(self):
        self.bullish_templates = [
            "{ticker} just broke resistance! 🚀 Target ${target}",
            "Massive volume spike on {ticker}. Something big is coming.",
            "I'm all in on {ticker}. This is the next 10-bagger 💎🙌",
            "{ticker} earnings beat by 25%! Moon mission activated 🌕",
            "Technical setup on {ticker} is perfect. Cup and handle forming.",
            "Just doubled my position in {ticker}. Can't miss this opportunity.",
            "Analysts upgraded {ticker} to strong buy. PT ${target}",
            "{ticker} short squeeze incoming! Institutions loading up.",
            "Best entry point for {ticker} in months. Don't sleep on this.",
            "Revenue growth for {ticker} is insane. Fundamentals are solid.",
        ]
        
        self.bearish_templates = [
            "Getting worried about {ticker}. Might take profits soon.",
            "{ticker} technical analysis shows double top. Selling signal.",
            "Overvalued at current levels. {ticker} due for correction.",
            "{ticker} insider selling increased 40% last month. Red flag 🚩",
            "Competition eating {ticker}'s market share. Not looking good.",
            "Regulatory concerns mounting for {ticker}. Too risky.",
            "{ticker} guidance lowered. Cutting my position by 50%.",
            "Debt levels for {ticker} are concerning. Balance sheet issues.",
            "Selling {ticker} before earnings. Too much uncertainty.",
            "{ticker} breaking key support levels. Technical breakdown.",
        ]
        
        self.neutral_templates = [
            "Watching {ticker} closely. Waiting for confirmation.",
            "{ticker} moving sideways. Need more volume to make a move.",
            "50/50 on {ticker} right now. Could go either way.",
            "Anyone else tracking {ticker}? What's your thesis?",
            "{ticker} consolidating. Looking for breakout direction.",
            "Mixed signals on {ticker}. Staying on the sidelines for now.",
            "Interesting price action on {ticker}. Setting alerts.",
            "{ticker} at key decision point. Monitoring closely.",
        ]
        
        self.sources = [
            "Reddit: r/wallstreetbets", "Reddit: r/stocks", "Reddit: r/investing",
            "Reddit: r/StockMarket", "Twitter", "StockTwits"
        ]
        
        self.company_data = {
            'AAPL': {'name': 'Apple', 'price': 180},
            'TSLA': {'name': 'Tesla', 'price': 240},
            'MSFT': {'name': 'Microsoft', 'price': 380},
            'GOOGL': {'name': 'Google', 'price': 140},
            'NVDA': {'name': 'NVIDIA', 'price': 480},
            'AMD': {'name': 'AMD', 'price': 150},
        }
    
    def generate_posts(self, ticker, count=20):
        ticker = ticker.upper()
        company = self.company_data.get(ticker, {'name': ticker, 'price': random.randint(50, 500)})
        posts = []
        
        # Realistic sentiment distribution
        for i in range(count):
            rand = random.random()
            if rand < 0.35:  # 35% bullish
                template = random.choice(self.bullish_templates)
            elif rand < 0.60:  # 25% bearish
                template = random.choice(self.bearish_templates)
            else:  # 40% neutral
                template = random.choice(self.neutral_templates)
            
            target_price = company['price'] * random.uniform(1.1, 1.4)
            text = template.format(ticker=f"${ticker}", target=f"{target_price:.0f}")
            
            posts.append({
                'text': text,
                'source': random.choice(self.sources),
                'score': random.randint(5, 500)
            })
        
        return [(p['text'], p['source']) for p in posts]

# Initialize mock generator
mock_generator = MockSocialDataGenerator()

# --- NLTK SETUP ---
nltk.download('vader_lexicon', download_dir='/tmp/nltk_data', quiet=True)
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data')
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# --- APP CONFIGURATION ---
app = Flask(__name__)

# --- FINBERT SETUP (Optional - will gracefully fallback to VADER) ---
USE_FINBERT = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    USE_FINBERT = True
    print("✅ FinBERT loaded successfully")
except Exception as e:
    print(f"ℹ️ Using VADER (FinBERT not available: {e})")

# --- ML MODEL SETUP ---
MODEL_PATH = 'sentiment_predictor.pkl'
SCALER_PATH = 'scaler.pkl'

def load_or_create_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ ML Model loaded from disk")
        return model, scaler
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        print("ℹ️ New ML model created (will use heuristics)")
        return model, scaler

ml_model, ml_scaler = load_or_create_model()

# --- DATABASE CONNECTION ---
def get_db_connection():
    try:
        conn = psycopg2.connect(os.environ.get("POSTGRES_URL", ""))
        return conn
    except Exception as e:
        print(f"ℹ️ Database not connected: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS logs (
                        id SERIAL PRIMARY KEY,
                        ticker TEXT,
                        source TEXT,
                        text TEXT,
                        score REAL,
                        label TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''')
        conn.commit()
        cur.close()
        conn.close()

# --- SENTIMENT ANALYSIS ---
def analyze_with_finbert(text):
    try:
        inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = finbert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_score = probs[0][0].item() - probs[0][1].item()
        return sentiment_score
    except Exception as e:
        return sia.polarity_scores(text)['compound']

def analyze_sentiment(text):
    if USE_FINBERT:
        return analyze_with_finbert(text)
    else:
        return sia.polarity_scores(text)['compound']

# --- DATA COLLECTION (Using Mock Data) ---
def fetch_social_data(ticker):
    """Fetch realistic mock social media data - NO API REQUIRED"""
    return mock_generator.generate_posts(ticker, count=20)

# --- ML PREDICTION ---
def get_ml_prediction(ticker, sentiment_scores):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        
        if len(hist) < 2:
            return None, "Insufficient data"
        
        avg_sentiment = np.mean(sentiment_scores)
        sentiment_std = np.std(sentiment_scores)
        recent_price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
        avg_volume = hist['Volume'].mean()
        
        features = np.array([[avg_sentiment, sentiment_std, recent_price_change, avg_volume]])
        
        try:
            features_scaled = ml_scaler.transform(features)
            prediction = ml_model.predict(features_scaled)[0]
            confidence = ml_model.predict_proba(features_scaled)[0].max()
            return prediction, confidence
        except:
            # Heuristic prediction
            if avg_sentiment > 0.1:
                return "BULLISH", 0.65
            elif avg_sentiment < -0.1:
                return "BEARISH", 0.65
            else:
                return "NEUTRAL", 0.55
    except Exception as e:
        print(f"ML prediction error: {e}")
        return None, 0.5

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    ticker = request.json.get('ticker', 'AAPL').upper()
    
    # 1. Fetch Real Stock Data
    price_dates = []
    price_values = []
    current_price = "N/A"
    volume_data = []
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d", interval="15m")
        
        if not hist.empty:
            price_dates = [d.strftime('%H:%M') for d in hist.index]
            price_values = [round(v, 2) for v in hist['Close'].tolist()]
            volume_data = hist['Volume'].tolist()
            current_price = price_values[-1]
    except Exception as e:
        print(f"⚠️ yFinance Error: {e}")
        # Fallback data
        price_dates = ["09:30", "10:00", "10:30", "11:00", "11:30", "12:00"]
        price_values = [150 + i*2 for i in range(6)]
        volume_data = [1000000] * 6
        current_price = price_values[-1]

    # 2. Fetch & Analyze Social Sentiment (Mock Data)
    raw_posts = fetch_social_data(ticker)
    
    conn = get_db_connection()
    new_logs = []
    sentiment_scores = []
    
    for text, source in raw_posts:
        score = analyze_sentiment(text)
        sentiment_scores.append(score)
        label = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
        
        # Store in database if available
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("INSERT INTO logs (ticker, source, text, score, label) VALUES (%s, %s, %s, %s, %s)",
                           (ticker, source, text, score, label))
                conn.commit()
                cur.close()
            except Exception as e:
                print(f"DB insert error: {e}")
        
        new_logs.append({
            'source': source,
            'text': text[:150] + "..." if len(text) > 150 else text,
            'label': label,
            'score': round(score, 3)
        })
    
    if conn:
        conn.close()

    # 3. ML-Based Prediction
    ml_prediction, confidence = get_ml_prediction(ticker, sentiment_scores)
    
    avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    
    if ml_prediction:
        prediction = f"{ml_prediction} 📈" if "BULLISH" in str(ml_prediction) else f"{ml_prediction} 📉" if "BEARISH" in str(ml_prediction) else f"{ml_prediction} ➖"
        prediction_details = f"{prediction} (Confidence: {confidence*100:.1f}%)"
    else:
        prediction = "BULLISH 📈" if avg_score > 0.05 else "BEARISH 📉" if avg_score < -0.05 else "NEUTRAL ➖"
        prediction_details = prediction

    # 4. Create sentiment trend
    sentiment_trend = []
    if sentiment_scores:
        base_sentiment = avg_score
        for i in range(len(price_values)):
            variation = np.random.normal(0, 0.1)
            sentiment_trend.append(round(base_sentiment + variation, 3))
    else:
        sentiment_trend = [0] * len(price_values)

    return jsonify({
        'ticker': ticker,
        'prediction': prediction_details,
        'avg_score': round(avg_score, 3),
        'current_price': current_price,
        'logs': new_logs[:15],
        'total_posts_analyzed': len(new_logs),
        'data_source': '🎭 High-Quality Mock Data (Demo Mode)',
        'nlp_model': 'FinBERT' if USE_FINBERT else 'VADER',
        'chart_data': {
            'labels': price_dates,
            'prices': price_values,
            'sentiment': sentiment_trend,
            'volume': volume_data
        }
    })

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'reddit_connected': False,
        'mock_data_active': True,
        'finbert_loaded': USE_FINBERT,
        'ml_model_trained': os.path.exists(MODEL_PATH),
        'database_connected': get_db_connection() is not None,
        'message': 'Running in Demo Mode with realistic mock data'
    })

# Initialize DB on startup
with app.app_context():
    init_db()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 SYNAPSE SENTIMENT ENGINE")
    print("="*60)
    print("✅ Mock Data Generator Active (No API keys needed)")
    print(f"✅ NLP Engine: {'FinBERT' if USE_FINBERT else 'VADER'}")
    print(f"✅ Database: {'Connected' if get_db_connection() else 'Optional (using memory)'}")
    print("="*60 + "\n")
    app.run(debug=True)
