import os
import psycopg2
import datetime
import nltk
import pickle
import random
from flask import Flask, render_template, jsonify, request
import yfinance as yf

# --- NLTK SETUP ---
try:
    nltk.download('vader_lexicon', download_dir='/tmp/nltk_data', quiet=True)
except:
    pass

NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data')
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# --- MOCK DATA GENERATOR ---
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
            'AMZN': {'name': 'Amazon', 'price': 170},
            'META': {'name': 'Meta', 'price': 350},
            'NFLX': {'name': 'Netflix', 'price': 450},
            'DIS': {'name': 'Disney', 'price': 95},
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

# --- APP CONFIGURATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'synapse-demo-2024')

# --- REDDIT API SETUP (Optional) ---
USE_REDDIT = False
try:
    import praw
    reddit_client_id = os.environ.get("REDDIT_CLIENT_ID", "")
    reddit_client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    
    if reddit_client_id and reddit_client_secret:
        reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent="SynapseSentimentEngine/1.0"
        )
        USE_REDDIT = True
        print("✅ Reddit API connected")
except Exception as e:
    print(f"ℹ️ Using mock data (Reddit API not available: {e})")

# --- ML MODEL SETUP (OPTIONAL) ---
USE_ML_MODEL = False
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    MODEL_PATH = 'sentiment_predictor.pkl'
    SCALER_PATH = 'scaler.pkl'
    
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            ml_model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            ml_scaler = pickle.load(f)
        USE_ML_MODEL = True
        print("✅ ML Model loaded")
    else:
        print("ℹ️ ML model not found, using heuristic predictions")
except Exception as e:
    print(f"ℹ️ ML libraries not available: {e}")

# --- DATABASE CONNECTION ---
def get_db_connection():
    try:
        postgres_url = os.environ.get("POSTGRES_URL", "")
        if postgres_url and postgres_url.strip():
            conn = psycopg2.connect(postgres_url, connect_timeout=10)
            return conn
    except Exception as e:
        print(f"ℹ️ Database not connected: {e}")
    return None

def init_db():
    conn = get_db_connection()
    if conn:
        try:
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
            print("✅ Database initialized")
        except Exception as e:
            print(f"⚠️ Database init error: {e}")

# --- SENTIMENT ANALYSIS ---
def analyze_sentiment(text):
    """Analyze sentiment using VADER"""
    return sia.polarity_scores(text)['compound']

# --- DATA COLLECTION ---
def fetch_reddit_posts(ticker, limit=20):
    """Fetch real posts from Reddit"""
    if not USE_REDDIT:
        return None
    
    posts = []
    try:
        subreddits = ['wallstreetbets', 'stocks', 'investing']
        for subreddit_name in subreddits[:2]:  # Limit to 2 for speed
            subreddit = reddit.subreddit(subreddit_name)
            for submission in subreddit.search(f"${ticker}", limit=limit//2, time_filter='day'):
                posts.append({
                    'text': f"{submission.title}. {submission.selftext[:200]}",
                    'source': f'Reddit: r/{subreddit_name}'
                })
        return [(p['text'], p['source']) for p in posts] if posts else None
    except Exception as e:
        print(f"Reddit fetch error: {e}")
        return None

def fetch_social_data(ticker):
    """Fetch social media data - tries Reddit, falls back to mock"""
    if USE_REDDIT:
        reddit_posts = fetch_reddit_posts(ticker)
        if reddit_posts:
            return reddit_posts
    
    # Fallback to mock data
    return mock_generator.generate_posts(ticker, count=20)

# --- ML PREDICTION ---
def get_ml_prediction(ticker, sentiment_scores):
    """Predict stock movement using ML or heuristic"""
    if not sentiment_scores:
        return "NEUTRAL", 0.5
    
    try:
        # Calculate average sentiment
        if USE_ML_MODEL:
            import numpy as np
            # Get historical data for features
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            
            if len(hist) >= 2:
                avg_sentiment = np.mean(sentiment_scores)
                sentiment_std = np.std(sentiment_scores)
                recent_price_change = (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]
                avg_volume = hist['Volume'].mean()
                
                features = np.array([[avg_sentiment, sentiment_std, recent_price_change, avg_volume]])
                features_scaled = ml_scaler.transform(features)
                prediction = ml_model.predict(features_scaled)[0]
                confidence = ml_model.predict_proba(features_scaled)[0].max()
                return prediction, confidence
    except Exception as e:
        print(f"ML prediction error: {e}")
    
    # Heuristic fallback
    avg_score = sum(sentiment_scores) / len(sentiment_scores)
    if avg_score > 0.1:
        return "BULLISH", 0.65
    elif avg_score < -0.1:
        return "BEARISH", 0.65
    else:
        return "NEUTRAL", 0.55

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    ticker = request.json.get('ticker', 'AAPL').upper()
    
    # 1. Fetch Stock Data
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
        else:
            # Fallback for closed market
            hist = stock.history(period="5d", interval="1h")
            if not hist.empty:
                hist = hist.tail(7)
                price_dates = [d.strftime('%H:%M') for d in hist.index]
                price_values = [round(v, 2) for v in hist['Close'].tolist()]
                volume_data = hist['Volume'].tolist()
                current_price = price_values[-1]
    except Exception as e:
        print(f"⚠️ yFinance Error: {e}")
        # Emergency fallback
        price_dates = ["09:30", "10:00", "10:30", "11:00", "11:30", "12:00"]
        price_values = [150 + i*2 for i in range(6)]
        volume_data = [1000000] * 6
        current_price = price_values[-1]

    # 2. Fetch & Analyze Social Sentiment
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
        if "BULLISH" in str(ml_prediction):
            prediction = f"{ml_prediction} 📈 (Confidence: {confidence*100:.1f}%)"
        elif "BEARISH" in str(ml_prediction):
            prediction = f"{ml_prediction} 📉 (Confidence: {confidence*100:.1f}%)"
        else:
            prediction = f"{ml_prediction} ➖ (Confidence: {confidence*100:.1f}%)"
    else:
        prediction = "BULLISH 📈" if avg_score > 0.05 else "BEARISH 📉" if avg_score < -0.05 else "NEUTRAL ➖"

    # 4. Create sentiment trend
    sentiment_trend = []
    if sentiment_scores:
        base_sentiment = avg_score
        for i in range(len(price_values)):
            variation = random.uniform(-0.1, 0.1)
            sentiment_trend.append(round(base_sentiment + variation, 3))
    else:
        sentiment_trend = [0] * len(price_values)

    return jsonify({
        'ticker': ticker,
        'prediction': prediction,
        'avg_score': round(avg_score, 3),
        'current_price': current_price,
        'logs': new_logs[:15],
        'total_posts_analyzed': len(new_logs),
        'data_source': 'Real Reddit Data' if USE_REDDIT else '🎭 High-Quality Mock Data (Demo Mode)',
        'nlp_model': 'VADER',
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
        'reddit_connected': USE_REDDIT,
        'mock_data_active': not USE_REDDIT,
        'finbert_loaded': False,
        'ml_model_trained': USE_ML_MODEL,
        'database_connected': get_db_connection() is not None,
        'message': 'Running on Vercel with VADER sentiment analysis'
    })

# Initialize DB on startup
with app.app_context():
    init_db()

# Vercel entry point
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 SYNAPSE SENTIMENT ENGINE - VERCEL OPTIMIZED")
    print("="*60)
    print(f"✅ Mock Data Generator: {'Standby' if USE_REDDIT else 'Active'}")
    print(f"✅ NLP Engine: VADER")
    print(f"✅ ML Model: {'Loaded' if USE_ML_MODEL else 'Heuristic Mode'}")
    print(f"✅ Database: {'Connected' if get_db_connection() else 'Optional (memory)'}")
    print("="*60 + "\n")
    app.run(debug=True)
