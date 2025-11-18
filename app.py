from flask import Flask, render_template, jsonify, request
import sqlite3
import random
import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

def analyze_sentiment(text):
    """Uses NLTK VADER to score text."""
    score = sia.polarity_scores(text)['compound']
    if score > 0.05: return score, "Positive"
    if score < -0.05: return score, "Negative"
    return score, "Neutral"

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
