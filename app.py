import os
import psycopg2
import datetime
import random
import nltk
from flask import Flask, render_template, jsonify, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf

# --- APP CONFIGURATION ---
app = Flask(__name__)

# NLTK SETUP: Handle Serverless Environment
# Vercel is read-only, so we download NLTK data to /tmp (writable) if missing
nltk_data_path = '/tmp/nltk_data'
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    # Download only if not found to save time
    nltk.download('vader_lexicon', download_dir=nltk_data_path)

sia = SentimentIntensityAnalyzer()

# --- DATABASE CONNECTION ---
def get_db_connection():
    """Connects to Vercel Postgres using the environment variable."""
    try:
        # POSTGRES_URL is automatically set by Vercel when you link the database
        conn = psycopg2.connect(os.environ["POSTGRES_URL"])
        return conn
    except Exception as e:
        print(f"❌ Database Connection Failed: {e}")
        return None

def init_db():
    """Creates the table if it doesn't exist."""
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        # We use TEXT for flexibility and TIMESTAMP for logs
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

# --- CORE INTELLIGENCE ---
def fetch_social_data(ticker):
    """
    Simulates social media scraping. 
    In a real hackathon, replace this with 'praw' (Reddit) or 'tweepy' (Twitter).
    """
    # Weighted templates to make the demo look realistic
    templates = [
        (f"${ticker} is breaking out! 🚀 Huge volume incoming.", "Twitter"),
        (f"I'm worried about ${ticker}'s upcoming earnings report.", "Reddit"),
        (f"Just loaded up on more ${ticker} shares.", "Reddit"),
        (f"Market is red, but ${ticker} is holding strong.", "Twitter"),
        (f"Technical analysis shows a double top for ${ticker}. Selling.", "Reddit"),
        (f"${ticker} to the moon! 🌕", "Twitter")
    ]
    return [random.choice(templates) for _ in range(5)]

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    ticker = request.json.get('ticker', 'AAPL').upper()
    
    # 1. Fetch Real Stock Data (The "Reality" Check)
    price_dates = []
    price_values = []
    current_price = "N/A"
    
    try:
        stock = yf.Ticker(ticker)
        # Get 1 day of data, 15-minute intervals for granularity
        hist = stock.history(period="1d", interval="15m")
        
        if not hist.empty:
            # Format dates to be readable (e.g., "10:30")
            price_dates = [d.strftime('%H:%M') for d in hist.index]
            price_values = [round(v, 2) for v in hist['Close'].tolist()]
            current_price = price_values[-1]
    except Exception as e:
        print(f"⚠️ yFinance Error: {e}")
        # Fallback data so the app doesn't break during demo
        price_dates = ["10:00", "11:00", "12:00", "13:00", "14:00"]
        price_values = [150, 152, 151, 153, 155]

    # 2. Fetch & Analyze Social Sentiment (The "Hype" Check)
    raw_posts = fetch_social_data(ticker)
    
    conn = get_db_connection()
    new_logs = []
    
    if conn:
        cur = conn.cursor()
        for text, source in raw_posts:
            score = sia.polarity_scores(text)['compound']
            label = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
            
            # Store in Postgres
            cur.execute("INSERT INTO logs (ticker, source, text, score, label) VALUES (%s, %s, %s, %s, %s)",
                        (ticker, source, text, score, label))
            new_logs.append({'source': source, 'text': text, 'label': label, 'score': score})
        conn.commit()
        cur.close()
        conn.close()
    else:
        # Fallback if DB fails (keeps app running)
        for text, source in raw_posts:
            score = sia.polarity_scores(text)['compound']
            label = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
            new_logs.append({'source': source, 'text': text, 'label': label, 'score': score})

    # 3. Calculate Prediction
    avg_score = sum([l['score'] for l in new_logs]) / len(new_logs) if new_logs else 0
    prediction = "BULLISH 📈" if avg_score > 0.05 else "BEARISH 📉" if avg_score < -0.05 else "NEUTRAL ➖"

    # 4. Align Sentiment Data with Price Data for the Chart
    # (Create a dummy sentiment trend that matches the length of price history for plotting)
    sentiment_trend = [avg_score + random.uniform(-0.2, 0.2) for _ in range(len(price_values))]

    return jsonify({
        'ticker': ticker,
        'prediction': prediction,
        'avg_score': round(avg_score, 2),
        'current_price': current_price,
        'logs': new_logs,
        'chart_data': {
            'labels': price_dates,
            'prices': price_values,
            'sentiment': sentiment_trend
        }
    })

# Initialize DB on Vercel startup (one-time check)
with app.app_context():
    init_db()

# Standard boilerplate
if __name__ == '__main__':
    app.run(debug=True)
