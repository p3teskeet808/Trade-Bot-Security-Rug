# Complete Project Code Documentation

## Table of Contents
1. Sentiment Analysis System
2. Web Server Implementation
3. Whale Tracking System
4. Gamification System
5. Configuration Files

## 1. Sentiment Analysis System (bot/sentiment_analyzer.py)
```python
"""Sentiment analyzer for crypto market analysis"""
import asyncio
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os

# Third-party imports
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import tweepy

from utils.logger import setup_logger
from utils.rate_limiter import RateLimiter
from utils.config_loader import load_config

logger = setup_logger()

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with proper error handling"""
        self.config = load_config()
        self.initialization_error = None
        self.initialized = False

        try:
            self.vader = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Failed to initialize VADER: {str(e)}")
            self.initialization_error = "Failed to initialize sentiment analysis"
            self.vader = None

        self.news_api = None
        self.twitter_api = None

        # Rate limiters
        self.news_limiter = RateLimiter(calls=100, period=86400)  # 100 calls per day
        self.twitter_limiter = RateLimiter(calls=180, period=900)  # 180 calls per 15 minutes

        # Historical sentiment data
        self.historical_sentiment = []
        self.max_history_points = 50
        self.last_update = 0

    async def setup(self):
        """Async setup for APIs and initial data"""
        if self.initialized:
            return

        try:
            # Check API keys
            news_api_key = os.getenv('NEWS_API_KEY')
            twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')

            if not news_api_key or not twitter_bearer_token:
                self.initialization_error = "Missing API keys. Please check environment variables."
                logger.error(self.initialization_error)
                return

            # Initialize APIs
            try:
                self.news_api = NewsApiClient(api_key=news_api_key)
                self.twitter_api = tweepy.Client(
                    bearer_token=twitter_bearer_token,
                    wait_on_rate_limit=True
                )
            except Exception as e:
                self.initialization_error = f"Failed to initialize APIs: {str(e)}"
                logger.error(self.initialization_error)
                return

            # Load initial sentiment data
            try:
                sentiment_data = await self.get_combined_sentiment("SOL")
                if not sentiment_data.get('error'):
                    self.historical_sentiment.append(sentiment_data)
                    self.initialized = True
                    logger.info("Sentiment analyzer initialized successfully")
                else:
                    self.initialization_error = f"Failed to get initial sentiment: {sentiment_data['error']}"
                    logger.error(self.initialization_error)
            except Exception as e:
                self.initialization_error = f"Failed to load initial sentiment data: {str(e)}"
                logger.error(self.initialization_error)

        except Exception as e:
            self.initialization_error = f"Initialization failed: {str(e)}"
            logger.error(self.initialization_error)

    def get_current_sentiment(self) -> Dict:
        """Get current sentiment data for the widget"""
        if not self.initialized:
            return {
                'error': self.initialization_error or 'Sentiment analyzer not initialized',
                'overall_score': 0,
                'news_sentiment': 0,
                'social_sentiment': 0,
                'historical_sentiment': [],
                'historical_timestamps': [],
                'updated_at': datetime.now().isoformat()
            }

        try:
            if self.historical_sentiment:
                current_sentiment = self.historical_sentiment[-1]
                return {
                    'overall_score': current_sentiment.get('sentiment_score', 0),
                    'news_sentiment': current_sentiment.get('news_sentiment', {}).get('score', 0),
                    'social_sentiment': current_sentiment.get('social_sentiment', {}).get('score', 0),
                    'historical_sentiment': [h.get('sentiment_score', 0) for h in self.historical_sentiment],
                    'historical_timestamps': [h.get('timestamp') for h in self.historical_sentiment],
                    'updated_at': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error getting current sentiment: {str(e)}")
            return {
                'error': f'Failed to process sentiment data: {str(e)}',
                'overall_score': 0,
                'news_sentiment': 0,
                'social_sentiment': 0,
                'historical_sentiment': [],
                'historical_timestamps': [],
                'updated_at': datetime.now().isoformat()
            }
```

## 2. Web Server Implementation (bot/web_server.py)
```python
"""Flask web server for trading bot UI components"""
import os
import sys
from datetime import datetime
import asyncio
from flask import Flask, render_template, jsonify
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.debug(f"Added {project_root} to Python path")

try:
    from utils.logger import setup_logger
    from utils.config_loader import load_config
    from bot.sentiment_analyzer import SentimentAnalyzer
    logger.debug("Successfully imported required modules")
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    sys.exit(1)

app = Flask(__name__, 
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
CORS(app)
sentiment_analyzer = None

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}")
        return "Error loading dashboard", 500

@app.route('/api/sentiment')
def get_sentiment():
    """Get current market sentiment data"""
    try:
        if sentiment_analyzer is None:
            logger.warning("Sentiment analyzer not initialized")
            return jsonify({
                'error': 'Sentiment analyzer not initialized',
                'overall_score': 0,
                'news_sentiment': 0,
                'social_sentiment': 0,
                'historical_sentiment': [],
                'historical_timestamps': [],
                'updated_at': datetime.now().isoformat()
            }), 503

        sentiment_data = sentiment_analyzer.get_current_sentiment()
        return jsonify(sentiment_data)
    except Exception as e:
        logger.error(f"Error fetching sentiment data: {str(e)}")
        return jsonify({
            'error': str(e),
            'overall_score': 0,
            'news_sentiment': 0,
            'social_sentiment': 0,
            'historical_sentiment': [],
            'historical_timestamps': [],
            'updated_at': datetime.now().isoformat()
        }), 500
```

## 3. Whale Tracking System (bot/whale_tracker.py)
```python
"""Whale tracker for monitoring large trades"""
import asyncio
import time
from utils.logger import setup_logger
from utils.config_loader import load_config
from utils.rate_limiter import RateLimiter
from bot.rpc_manager import RPCManager 
from decimal import Decimal
import numpy as np

logger = setup_logger()

class WhaleTracker:
    def __init__(self):
        self.config = load_config()
        self.rate_limiter = RateLimiter(
            calls=self.config["rate_limits"]["solscan_api"],
            period=60
        )
        self.rpc_manager = RPCManager()
        # Enhanced whale categorization with profit patterns
        self.whale_wallets = {
            "FZbPXmL5PtbusvXE4UJ5xMVeqYoJRa3MCKGhS8KcZJB2": {
                "name": "Alpha Trader",
                "type": "alpha_trader",
                "min_trade_size": 25,
                "success_rate": 0.85,
                "avg_profit": 35.2,
                "quick_flip_rate": 0.75
            },
            "6vH9ByZVQwQ2xeJf3EhzBhBqrqXYwkzHf8teFPvqCGdA": {
                "name": "Pattern Trader",
                "type": "pattern_trader",
                "min_trade_size": 15,
                "success_rate": 0.82,
                "avg_profit": 28.5,
                "pattern_accuracy": 0.80
            }
        }
        self.transaction_cache = {}
        self.last_check = {}
        self.success_history = {}
        self.pattern_cache = {}
        self.price_impact_history = {}
        self.monitoring_start_time = time.time()
        self.profit_patterns = []

    async def analyze_price_impact(self, token_address, transfer_amount, supply):
        """Analyze real-time price impact and potential profit"""
        try:
            impact = (transfer_amount / supply) * 100
            timestamp = time.time()

            if token_address not in self.price_impact_history:
                self.price_impact_history[token_address] = []

            self.price_impact_history[token_address].append({
                'impact': impact,
                'timestamp': timestamp
            })

            # Keep only recent history
            recent_impacts = [x for x in self.price_impact_history[token_address] 
                            if timestamp - x['timestamp'] < 3600]
            self.price_impact_history[token_address] = recent_impacts

            # Calculate potential profit based on impact
            avg_impact = sum(x['impact'] for x in recent_impacts) / len(recent_impacts) if recent_impacts else impact
            profit_potential = min(impact * 2, avg_impact * 3)  # Conservative estimate

            return {
                'current_impact': impact,
                'avg_impact': avg_impact,
                'profit_potential': profit_potential,
                'confidence': len(recent_impacts) / 10  # Max confidence at 10 data points
            }

        except Exception as e:
            logger.error(f"Error analyzing price impact: {str(e)}")
            return None
```

## 4. Gamification System (bot/gamification/learning_path.py)
```python
"""
Gamified Learning Path for Crypto Trading Skills
Implements a progression system with achievements and educational content
"""
from enum import Enum
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
from utils.logger import setup_logger

logger = setup_logger()

class SkillLevel(Enum):
    NOVICE = "Novice"
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"

@dataclass
class Achievement:
    id: str
    name: str
    description: str
    points: int
    requirements: Dict[str, float]
    completed: bool = False
    completed_at: Optional[float] = None

class LearningPath:
    def __init__(self):
        self.current_level = SkillLevel.NOVICE
        self.total_points = 0
        self.achievements = self._initialize_achievements()
        self.completed_lessons = set()
        
    def _initialize_achievements(self) -> List[Achievement]:
        """Initialize the achievement system"""
        return [
            Achievement(
                id="first_trade",
                name="First Steps",
                description="Complete your first automated trade",
                points=10,
                requirements={"trades_completed": 1}
            ),
            Achievement(
                id="profit_master",
                name="Profit Master",
                description="Achieve 10% profit on a single trade",
                points=20,
                requirements={"trade_profit_percentage": 10.0}
            ),
            Achievement(
                id="whale_watcher",
                name="Whale Watcher",
                description="Successfully follow 5 whale trades",
                points=30,
                requirements={"whale_trades_followed": 5}
            ),
            Achievement(
                id="diamond_hands",
                name="Diamond Hands",
                description="Hold a profitable position for over 24 hours",
                points=25,
                requirements={"hold_time_hours": 24}
            )
        ]
```

## 5. Configuration Files

### config.yaml
```yaml
wallet:
  solana_address: "7rh424MTkyhZdY2UyYLthwv7wQmEefY76ninD3a4h61F"
  solana_private_key: "5pHxng4UxWUCsS1MP9Spi6yr6zoEKADnvNHu1P2fQvUbM5yjGavJZrFGvcQj6cMWMjDf2mwVmMfusKioP97CeQ43"  
  trade_amount_sol: 0.000001  # Minimum possible trade amount
  slippage: 0.5  # Reduced for tighter spreads

telegram:
  bot_token: "7037595555:AAEpm5FgVbtbb7yjEGUU6_hvLrKqhySrKRY"
  chat_id: "1993844835"  

apis:
  news_api_key: ""  # Will be populated from environment secrets
  twitter_bearer_token: ""  # Will be populated from environment secrets

filters:
  min_liquidity: 5000  # Reduced from 25000
  min_market_cap: 10000  # Reduced from 50000
  max_fake_volume: 5  # More strict
  min_pump_percent: 5  # Lower threshold for faster entry
  auto_sell_percent: 15  # Quicker profit taking
  stop_loss_percent: 10  # Tighter stop loss
```

### pyproject.toml
```toml
[project]
name = "repl-nix-workspace"
version = "0.1.0"
description = "Add your description here"
requires-python = ">=3.11"
dependencies = [
    "aiohttp>=3.11.12",
    "anchorpy>=0.20.1",
    "base58>=2.1.1",
    "cachetools>=5.5.1",
    "construct>=2.10.68",
    "construct-typing>=0.5.6",
    "flask-cors>=5.0.0",
    "flask>=3.1.0",
    "flask-login>=0.6.3",
    "flask-wtf>=1.2.2",
    "joblib>=1.4.2",
    "newsapi-python>=0.2.7",
    "numpy>=2.2.3",
    "oauthlib>=3.2.2",
    "pandas>=2.2.3",
    "pytest>=8.3.4",
    "python-dotenv>=1.0.1",
    "pyyaml>=6.0.2",
    "requests>=2.32.3",
    "scikit-learn>=1.6.1",
    "sendgrid>=6.11.0",
    "slack-sdk>=3.34.0",
    "solana>=0.35.1",
    "solders>=0.21.0",
    "telepot>=12.7",
    "textblob>=0.19.0",
    "tweepy>=4.15.0",
    "twilio>=9.4.5",
    "typing-extensions>=4.12.2",
    "vadersentiment>=3.3.2",
]
```
