import os
import random
from flask import Flask, render_template, jsonify, request

# --- MINIMAL IMPORTS ONLY ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'synapse-demo-2024')

# --- MOCK DATA GENERATOR ---
class MockSocialDataGenerator:
    def __init__(self):
        self.bullish_templates = [
            "{ticker} just broke resistance! 🚀 Target ${target}",
            "Massive volume spike on {ticker}. Something big is coming.",
            "I'm all in on {ticker}. This is the next 10-bagger 💎🙌",
            "Technical setup on {ticker} is perfect. Cup and handle forming.",
            "Just doubled my position in {ticker}. Can't miss this opportunity.",
        ]
        
        self.bearish_templates = [
            "Getting worried about {ticker}. Might take profits soon.",
            "{ticker} technical analysis shows double top. Selling signal.",
            "Overvalued at current levels. {ticker} due for correction.",
            "{ticker} insider selling increased. Red flag 🚩",
            "Competition eating {ticker}'s market share. Not looking good.",
        ]
        
        self.neutral_templates = [
            "Watching {ticker} closely. Waiting for confirmation.",
            "{ticker} moving sideways. Need more volume to make a move.",
            "50/50 on {ticker} right now. Could go either way.",
            "Anyone else tracking {ticker}? What's your thesis?",
        ]
        
        self.sources = ["Reddit: r/wallstreetbets", "Reddit: r/stocks", "Twitter", "StockTwits"]
        
        self.company_data = {
            'AAPL': 180, 'TSLA': 240, 'MSFT': 380, 'GOOGL': 140,
            'NVDA': 480, 'AMD': 150, 'AMZN': 170, 'META': 350
        }
    
    def generate_posts(self, ticker, count=15):
        ticker = ticker.upper()
        price = self.company_data.get(ticker, random.randint(50, 500))
        posts = []
        
        for i in range(count):
            rand = random.random()
            if rand < 0.35:
                template = random.choice(self.bullish_templates)
            elif rand < 0.60:
                template = random.choice(self.bearish_templates)
            else:
                template = random.choice(self.neutral_templates)
            
            target_price = price * random.uniform(1.1, 1.4)
            text = template.format(ticker=f"${ticker}", target=f"{target_price:.0f}")
            
            posts.append((text, random.choice(self.sources)))
        
        return posts

mock_generator = MockSocialDataGenerator()

# --- SIMPLE SENTIMENT ANALYSIS ---
def analyze_sentiment_simple(text):
    """Ultra-simple sentiment without NLTK"""
    text_lower = text.lower()
    
    # Positive words
    positive = ['moon', '🚀', 'bullish', 'buy', 'long', 'breakout', 'bullish', 
                'upgrade', 'beat', 'growth', 'strong', 'doubled', 'squeeze']
    
    # Negative words
    negative = ['worried', 'sell', 'crash', 'bearish', 'drop', 'correction',
                'overvalued', 'loss', 'risk', 'flag', '📉', 'debt']
    
    score = 0
    for word in positive:
        if word in text_lower:
            score += 0.3
    for word in negative:
        if word in text_lower:
            score -= 0.3
    
    # Clamp between -1 and 1
    return max(-1, min(1, score))

# --- SIMPLE STOCK DATA ---
def get_stock_data(ticker):
    """Generate simple mock stock data"""
    base_price = mock_generator.company_data.get(ticker.upper(), 150)
    
    times = ["09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30"]
    prices = []
    current = base_price
    
    for i in range(len(times)):
        change = random.uniform(-0.02, 0.02)  # ±2% change
        current = current * (1 + change)
        prices.append(round(current, 2))
    
    return {
        'times': times,
        'prices': prices,
        'current_price': prices[-1]
    }

# --- ROUTES ---
@app.route('/')
def index():
    try:
        return render_template('dashboard.html')
    except Exception as e:
        return f"Error loading template: {e}", 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        # Get ticker
        data = request.get_json()
        ticker = data.get('ticker', 'AAPL').upper()
        
        # Get stock data
        stock_data = get_stock_data(ticker)
        
        # Get social posts
        raw_posts = mock_generator.generate_posts(ticker, count=15)
        
        # Analyze sentiment
        new_logs = []
        sentiment_scores = []
        
        for text, source in raw_posts:
            score = analyze_sentiment_simple(text)
            sentiment_scores.append(score)
            label = "Positive" if score > 0.1 else "Negative" if score < -0.1 else "Neutral"
            
            new_logs.append({
                'source': source,
                'text': text[:150],
                'label': label,
                'score': round(score, 3)
            })
        
        # Calculate prediction
        avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        if avg_score > 0.1:
            prediction = f"BULLISH 📈 (Confidence: 65.0%)"
        elif avg_score < -0.1:
            prediction = f"BEARISH 📉 (Confidence: 65.0%)"
        else:
            prediction = f"NEUTRAL ➖ (Confidence: 55.0%)"
        
        # Create sentiment trend
        sentiment_trend = [round(avg_score + random.uniform(-0.1, 0.1), 3) 
                          for _ in range(len(stock_data['times']))]
        
        return jsonify({
            'ticker': ticker,
            'prediction': prediction,
            'avg_score': round(avg_score, 3),
            'current_price': stock_data['current_price'],
            'logs': new_logs[:15],
            'total_posts_analyzed': len(new_logs),
            'data_source': '🎭 Demo Mode (Optimized)',
            'nlp_model': 'Pattern-Based',
            'chart_data': {
                'labels': stock_data['times'],
                'prices': stock_data['prices'],
                'sentiment': sentiment_trend,
                'volume': [1000000 + random.randint(-200000, 200000) 
                          for _ in range(len(stock_data['times']))]
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'reddit_connected': False,
        'mock_data_active': True,
        'finbert_loaded': False,
        'ml_model_trained': False,
        'database_connected': False,
        'message': 'Demo mode - All systems operational'
    })

if __name__ == '__main__':
    app.run(debug=True)
