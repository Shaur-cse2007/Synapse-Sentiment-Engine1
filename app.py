from flask import Flask, render_template, jsonify, request
import sqlite3
import random
import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf

# --- SETUP ---
app = Flask(__name__)
DB_NAME = "synapse.db"

# Download VADER lexicon (first run only)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

# --- DATABASE ---
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT, source TEXT, text TEXT,
                        score REAL, label TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

# --- CORE ENGINE ---
def fetch_social_data(ticker):
    """Simulates fetching data. Replace this with PRAW/Tweepy for real data."""
    # Mock data templates for demo purposes
    templates = [
        (f"${ticker} is looking strong! 🚀", "Twitter"),
        (f"Selling my ${ticker} shares, market looks bad.", "Reddit"),
        (f"Just bought the dip on ${ticker}.", "Reddit"),
        (f"${ticker} earnings are going to be huge.", "Twitter"),
        (f"Not sure about ${ticker}, waiting for news.", "Twitter")
    ]
    return [random.choice(templates) for _ in range(5)]

@app.route('/api/analyze', methods=['POST'])
def analyze():
    ticker = request.json.get('ticker', 'AAPL').upper()
    
    # --- NEW: Fetch Real Stock Price Data ---
    try:
        stock = yf.Ticker(ticker)
        # Get last 1 day of data at 15-minute intervals
        hist = stock.history(period="1d", interval="15m")
        
        # Extract relevant data for the chart
        price_dates = [str(date.strftime('%H:%M')) for date in hist.index]
        price_values = [round(val, 2) for val in hist['Close'].tolist()]
        current_price = price_values[-1] if price_values else "N/A"
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        price_dates, price_values, current_price = [], [], "Error"

    # --- EXISTING: Sentiment Analysis Logic ---
    raw_posts = fetch_social_data(ticker) # (Your existing function)
    
    # ... (Keep your existing SQL logic here for inserting logs) ...
    # For the response, we just need to calculate the average score again
    
    # Mocking the score calculation for this snippet (use your DB logic here)
    avg_score = 0.15 
    prediction = "BULLISH 📈" if avg_score > 0.05 else "BEARISH 📉"

    return jsonify({
        'ticker': ticker,
        'prediction': prediction,
        'avg_score': avg_score,
        'current_price': current_price,
        'chart_data': {
            'labels': price_dates,       # X-Axis (Time)
            'prices': price_values,      # Y-Axis 1 (Price)
            # Create mock sentiment data matching the length of price data for the demo chart
            'sentiment': [avg_score + random.uniform(-0.1, 0.1) for _ in range(len(price_values))] 
        }
    })

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    ticker = request.json.get('ticker', 'AAPL').upper()
    
    # 1. Fetch & Analyze
    raw_posts = fetch_social_data(ticker)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    new_logs = []
    for text, source in raw_posts:
        score, label = analyze_sentiment(text)
        c.execute("INSERT INTO logs (ticker, source, text, score, label) VALUES (?,?,?,?,?)",
                  (ticker, source, text, score, label))
        new_logs.append({'source': source, 'text': text, 'label': label, 'score': score})
    
    conn.commit()
    
    # 2. Calculate Prediction (based on last 20 posts)
    c.execute("SELECT score FROM logs WHERE ticker=? ORDER BY id DESC LIMIT 20", (ticker,))
    scores = [r[0] for r in c.fetchall()]
    conn.close()
    
    avg_score = sum(scores) / len(scores) if scores else 0
    prediction = "BULLISH 📈" if avg_score > 0.05 else "BEARISH 📉" if avg_score < -0.05 else "NEUTRAL ➖"

    return jsonify({
        'ticker': ticker,
        'prediction': prediction,
        'avg_score': round(avg_score, 2),
        'logs': new_logs
    })

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)
