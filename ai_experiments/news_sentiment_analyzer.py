#!/usr/bin/env python3
"""
News Sentiment Analyzer Module

This module provides classes and functions for fetching and analyzing news sentiment 
from various sources including Alpha Vantage, NewsAPI, Reddit, and Google News.
"""

import os
import json
import logging
from datetime import datetime, timedelta
import re
import statistics
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd # type: ignore
import numpy as np # type: ignore
import requests
from bs4 import BeautifulSoup # type: ignore
from newsapi import NewsApiClient # type: ignore
import praw # type: ignore
from gnews import GNews  # type: ignore # Using gnews instead of pygooglenews
from ai_experiments.alpha_vantage_pipeline import AlphaVantageAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('news_sentiment_analyzer')

# Load environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "FinanceAI/1.0")

class NewsSentimentAnalyzer:
    """Analyzes news sentiment for financial markets."""
    
    def __init__(self):
        """Initialize the news sentiment analyzer."""
        self.initialized = True
    
    def get_market_sentiment(self):
        """Get market sentiment analysis."""
        # Mock data for news sentiment
        return {
            "overall": 0.35,
            "topSources": [
                {"name": "Bloomberg", "sentiment": 0.42},
                {"name": "CNBC", "sentiment": 0.38},
                {"name": "Reuters", "sentiment": 0.25},
                {"name": "Wall Street Journal", "sentiment": 0.15},
                {"name": "Financial Times", "sentiment": -0.12}
            ],
            "recentArticles": [
                {
                    "title": "Fed signals potential rate cuts later this year",
                    "source": "Bloomberg",
                    "date": "2023-06-15",
                    "sentiment": 0.58,
                    "url": "#"
                },
                {
                    "title": "Tech stocks rally as inflation concerns ease",
                    "source": "CNBC",
                    "date": "2023-06-14",
                    "sentiment": 0.65,
                    "url": "#"
                },
                {
                    "title": "Market volatility increases amid geopolitical tensions",
                    "source": "Financial Times",
                    "date": "2023-06-13",
                    "sentiment": -0.32,
                    "url": "#"
                },
                {
                    "title": "Treasury yields climb after latest economic data",
                    "source": "Wall Street Journal",
                    "date": "2023-06-12",
                    "sentiment": -0.18,
                    "url": "#"
                }
            ]
        }


class AlphaVantageSentiment(NewsSentimentAnalyzer):
    """Class for analyzing news sentiment from Alpha Vantage News API."""
    
    def __init__(self, api: AlphaVantageAPI):
        """
        Initialize with AlphaVantageAPI instance.
        
        Args:
            api (AlphaVantageAPI): Instance of AlphaVantageAPI
        """
        super().__init__()
        self.api = api
    
    def get_sentiment_for_symbol(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Get sentiment analysis for a specific symbol.
        
        Args:
            symbol (str): Stock symbol
            days_back (int): Number of days to look back
            
        Returns:
            Dict: Sentiment analysis results
        """
        try:
            # Get news data from Alpha Vantage
            logger.info(f"Fetching news sentiment for {symbol}")
            news_data = self.api.get_market_news(tickers=symbol, limit=50)
            
            if not news_data or 'feed' not in news_data or not news_data['feed']:
                logger.warning(f"No news data returned for {symbol}")
                return {
                    'average_sentiment': 0,
                    'sentiment_volume': 0,
                    'sentiment_trend': 'stable',
                    'source': 'Alpha Vantage (no data)',
                    'last_updated': datetime.now().isoformat()
                }
            
            # Extract sentiment scores
            sentiment_scores = []
            timestamps = []
            
            for article in news_data.get('feed', []):
                # Use Alpha Vantage's sentiment score if available
                if 'overall_sentiment_score' in article:
                    score = float(article['overall_sentiment_score'])
                    sentiment_scores.append(score)
                    timestamps.append(article.get('time_published', ''))
                # Otherwise calculate our own
                elif 'title' in article:
                    score = self.basic_sentiment_analysis(article['title'] + ' ' + article.get('summary', ''))
                    sentiment_scores.append(score)
                    timestamps.append(article.get('time_published', ''))
            
            if not sentiment_scores:
                logger.warning(f"No sentiment scores calculated for {symbol}")
                return {
                    'average_sentiment': 0,
                    'sentiment_volume': 0,
                    'sentiment_trend': 'stable',
                    'source': 'Alpha Vantage (no scores)',
                    'last_updated': datetime.now().isoformat()
                }
            
            # Calculate metrics
            average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            sentiment_trend = self._calculate_trend(sentiment_scores)
            
            # Return sentiment analysis
            return {
                'average_sentiment': round(average_sentiment, 2),
                'sentiment_volume': len(sentiment_scores),
                'sentiment_trend': sentiment_trend,
                'source': 'Alpha Vantage',
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
            return {
                'average_sentiment': 0,
                'sentiment_volume': 0,
                'sentiment_trend': 'stable',
                'source': f'Error: {str(e)}',
                'last_updated': datetime.now().isoformat()
            }


class NewsAPISentiment(NewsSentimentAnalyzer):
    """Class for analyzing news sentiment from NewsAPI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize with NewsAPI key.
        
        Args:
            api_key (str, optional): NewsAPI key, defaults to environment variable
        """
        super().__init__()
        self.api_key = api_key or NEWS_API_KEY
        if not self.api_key:
            logger.warning("NewsAPI key not provided, this source will be unavailable")
        else:
            self.client = NewsApiClient(api_key=self.api_key)
    
    def get_sentiment_for_symbol(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Get sentiment analysis for a specific symbol from NewsAPI.
        
        Args:
            symbol (str): Stock symbol
            days_back (int): Number of days to look back
            
        Returns:
            Dict: Sentiment analysis results
        """
        if not self.api_key:
            return {
                'average_sentiment': 0,
                'sentiment_volume': 0,
                'sentiment_trend': 'stable',
                'source': 'NewsAPI (no API key)',
                'last_updated': datetime.now().isoformat()
            }
            
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Format dates for NewsAPI
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            # Search for news about the symbol
            company_search = self._get_company_name(symbol)
            query = f"{symbol} OR {company_search}"
            
            articles = self.client.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='publishedAt',
                page_size=100
            )
            
            if 'articles' not in articles or not articles['articles']:
                logger.warning(f"No news articles found for {symbol} from NewsAPI")
                return {
                    'average_sentiment': 0,
                    'sentiment_volume': 0,
                    'sentiment_trend': 'stable',
                    'source': 'NewsAPI (no articles)',
                    'last_updated': datetime.now().isoformat()
                }
            
            # Calculate sentiment for each article
            sentiment_scores = []
            timestamps = []
            
            for article in articles['articles']:
                title = article.get('title', '')
                description = article.get('description', '')
                content = article.get('content', '')
                
                text = f"{title} {description} {content}"
                score = self.basic_sentiment_analysis(text)
                
                sentiment_scores.append(score)
                timestamps.append(article.get('publishedAt', ''))
            
            # Calculate metrics
            average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            sentiment_trend = self._calculate_trend(sentiment_scores)
            
            # Return sentiment analysis
            return {
                'average_sentiment': round(average_sentiment, 2),
                'sentiment_volume': len(sentiment_scores),
                'sentiment_trend': sentiment_trend,
                'source': 'NewsAPI',
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment from NewsAPI for {symbol}: {str(e)}")
            return {
                'average_sentiment': 0,
                'sentiment_volume': 0,
                'sentiment_trend': 'stable',
                'source': f'NewsAPI Error: {str(e)}',
                'last_updated': datetime.now().isoformat()
            }
    
    def _get_company_name(self, symbol: str) -> str:
        """Map stock symbol to company name for better search results."""
        # Common stock symbols to company names
        symbol_to_name = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google OR Alphabet',
            'AMZN': 'Amazon',
            'META': 'Facebook OR Meta',
            'TSLA': 'Tesla',
            'NVDA': 'NVIDIA',
            'JPM': 'JPMorgan Chase',
            'V': 'Visa Inc',
            'JNJ': 'Johnson & Johnson',
        }
        
        return symbol_to_name.get(symbol, symbol)


class RedditSentiment(NewsSentimentAnalyzer):
    """Class for analyzing sentiment from Reddit."""
    
    def __init__(
        self, 
        client_id: Optional[str] = None, 
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Initialize with Reddit API credentials.
        
        Args:
            client_id (str, optional): Reddit API client ID
            client_secret (str, optional): Reddit API client secret
            user_agent (str, optional): Reddit API user agent
        """
        super().__init__()
        self.client_id = client_id or REDDIT_CLIENT_ID
        self.client_secret = client_secret or REDDIT_CLIENT_SECRET
        self.user_agent = user_agent or REDDIT_USER_AGENT
        
        if not all([self.client_id, self.client_secret, self.user_agent]):
            logger.warning("Reddit API credentials not complete, this source will be unavailable")
            self.reddit = None
        else:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
            except Exception as e:
                logger.error(f"Error initializing Reddit client: {str(e)}")
                self.reddit = None
        
        # Finance-related subreddits to monitor
        self.finance_subreddits = ['wallstreetbets', 'stocks', 'investing', 'finance', 'options']
    
    def get_sentiment_for_symbol(self, symbol: str, days_back: int = 7, limit: int = 100) -> Dict[str, Any]:
        """
        Get sentiment analysis for a specific symbol from Reddit.
        
        Args:
            symbol (str): Stock symbol
            days_back (int): Number of days to look back
            limit (int): Maximum number of posts to analyze per subreddit
            
        Returns:
            Dict: Sentiment analysis results
        """
        if not self.reddit:
            return {
                'average_sentiment': 0,
                'sentiment_volume': 0,
                'sentiment_trend': 'stable',
                'source': 'Reddit (no credentials)',
                'last_updated': datetime.now().isoformat()
            }
            
        try:
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            sentiment_scores = []
            timestamps = []
            
            # Search across finance subreddits
            for subreddit_name in self.finance_subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for posts matching the symbol
                    search_query = f"{symbol}"
                    
                    for post in subreddit.search(search_query, sort='new', limit=limit):
                        # Skip posts older than cutoff
                        post_time = datetime.fromtimestamp(post.created_utc)
                        if post_time < cutoff_time:
                            continue
                            
                        # Analyze post title and body
                        post_text = f"{post.title} {post.selftext}"
                        
                        # Check if post actually mentions the symbol
                        if re.search(r'\b{}\b'.format(symbol), post_text):
                            score = self.basic_sentiment_analysis(post_text)
                            sentiment_scores.append(score)
                            timestamps.append(post_time.isoformat())
                            
                            # Also analyze top comments
                            post.comments.replace_more(limit=0)
                            for comment in list(post.comments)[:5]:  # Top 5 comments
                                comment_score = self.basic_sentiment_analysis(comment.body)
                                sentiment_scores.append(comment_score)
                                timestamps.append(datetime.fromtimestamp(comment.created_utc).isoformat())
                
                except Exception as e:
                    logger.warning(f"Error analyzing {subreddit_name}: {str(e)}")
                    continue
            
            if not sentiment_scores:
                logger.warning(f"No Reddit posts found for {symbol}")
                return {
                    'average_sentiment': 0,
                    'sentiment_volume': 0,
                    'sentiment_trend': 'stable',
                    'source': 'Reddit (no posts)',
                    'last_updated': datetime.now().isoformat()
                }
            
            # Calculate metrics
            average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            sentiment_trend = self._calculate_trend(sentiment_scores)
            
            # Return sentiment analysis
            return {
                'average_sentiment': round(average_sentiment, 2),
                'sentiment_volume': len(sentiment_scores),
                'sentiment_trend': sentiment_trend,
                'source': 'Reddit',
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment from Reddit for {symbol}: {str(e)}")
            return {
                'average_sentiment': 0,
                'sentiment_volume': 0,
                'sentiment_trend': 'stable',
                'source': f'Reddit Error: {str(e)}',
                'last_updated': datetime.now().isoformat()
            }


class SentimentManager:
    """
    Manager class that aggregates sentiment from multiple sources.
    """
    
    def __init__(self, alpha_vantage_api: Optional[AlphaVantageAPI] = None):
        """
        Initialize with API instances.
        
        Args:
            alpha_vantage_api (AlphaVantageAPI, optional): Alpha Vantage API instance
        """
        self.sources = []
        
        # Initialize Alpha Vantage sentiment source if API provided
        if alpha_vantage_api:
            self.sources.append(('alpha_vantage', AlphaVantageSentiment(alpha_vantage_api), 1.0))
        
        # Initialize NewsAPI sentiment source
        try:
            news_api_sentiment = NewsAPISentiment()
            if news_api_sentiment.api_key:
                self.sources.append(('news_api', news_api_sentiment, 0.8))
        except Exception as e:
            logger.error(f"Error initializing NewsAPI sentiment: {str(e)}")
        
        # Initialize Reddit sentiment source
        try:
            reddit_sentiment = RedditSentiment()
            if reddit_sentiment.reddit:
                self.sources.append(('reddit', reddit_sentiment, 0.7))
        except Exception as e:
            logger.error(f"Error initializing Reddit sentiment: {str(e)}")
    
    def get_aggregated_sentiment(self, symbol: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Get aggregated sentiment from all available sources.
        
        Args:
            symbol (str): Stock symbol
            days_back (int): Number of days to look back
            
        Returns:
            Dict: Aggregated sentiment analysis
        """
        if not self.sources:
            logger.warning("No sentiment sources available")
            return {
                'symbol': symbol,
                'average_sentiment': 0,
                'sentiment_volume': 0,
                'sentiment_trend': 'stable',
                'sources': ['None available'],
                'source_sentiments': {},
                'last_updated': datetime.now().isoformat()
            }
        
        try:
            # Collect sentiment from each source
            sentiments = {}
            sources_used = []
            
            for source_name, source, weight in self.sources:
                logger.info(f"Getting sentiment from {source_name} for {symbol}")
                sentiment = source.get_sentiment_for_symbol(symbol, days_back)
                
                if sentiment.get('sentiment_volume', 0) > 0:
                    sentiments[source_name] = sentiment
                    sources_used.append(source_name)
            
            if not sentiments:
                logger.warning(f"No sentiment data available for {symbol} from any source")
                return {
                    'symbol': symbol,
                    'average_sentiment': 0,
                    'sentiment_volume': 0,
                    'sentiment_trend': 'stable',
                    'sources': ['No data available'],
                    'source_sentiments': {},
                    'last_updated': datetime.now().isoformat()
                }
            
            # Extract scores and weights
            scores = []
            weights = []
            volumes = []
            trends = []
            
            for source_name, source, source_weight in self.sources:
                if source_name in sentiments:
                    scores.append(sentiments[source_name].get('average_sentiment', 0))
                    volumes.append(sentiments[source_name].get('sentiment_volume', 0))
                    trends.append(sentiments[source_name].get('sentiment_trend', 'stable'))
                    weights.append(source_weight * sentiments[source_name].get('sentiment_volume', 0))
            
            # Calculate weighted metrics
            avg_sentiment = 0
            total_volume = sum(volumes)
            
            if scores and weights and sum(weights) > 0:
                # Use volume-weighted average
                avg_sentiment = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            
            # Determine overall trend
            trend_counts = {'improving': 0, 'stable': 0, 'worsening': 0}
            for t in trends:
                if t in trend_counts:
                    trend_counts[t] += 1
            
            overall_trend = max(trend_counts.items(), key=lambda x: x[1])[0]
            
            # Return aggregated sentiment
            return {
                'symbol': symbol,
                'average_sentiment': round(avg_sentiment, 2),
                'sentiment_volume': total_volume,
                'sentiment_trend': overall_trend,
                'sources': sources_used,
                'source_sentiments': {name: sentiments[name] for name in sources_used},
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error aggregating sentiment for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'average_sentiment': 0,
                'sentiment_volume': 0,
                'sentiment_trend': 'stable',
                'sources': [f'Error: {str(e)}'],
                'source_sentiments': {},
                'last_updated': datetime.now().isoformat()
            }


# Utility functions
def extract_topics_from_news(news_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract common topics and their sentiment from news data.
    
    Args:
        news_data (Dict): News data from Alpha Vantage or other source
        
    Returns:
        Dict: Topic names and their average sentiment scores
    """
    topics = {}
    
    # Process Alpha Vantage news feed
    if 'feed' in news_data:
        for article in news_data['feed']:
            # Extract topics from categories
            if 'topics' in article:
                for topic in article['topics']:
                    topic_name = topic.get('topic', '').lower()
                    if topic_name and len(topic_name) > 3:  # Skip very short topics
                        sentiment = float(topic.get('relevance_score', 0.5))
                        
                        if topic_name in topics:
                            topics[topic_name].append(sentiment)
                        else:
                            topics[topic_name] = [sentiment]
    
    # Calculate average sentiment for each topic
    return {topic: sum(scores) / len(scores) for topic, scores in topics.items() if scores}


if __name__ == "__main__":
    # Example usage
    from ai_experiments.alpha_vantage_pipeline import AlphaVantageAPI
    
    api = AlphaVantageAPI()
    sentiment_manager = SentimentManager(api)
    
    symbol = "AAPL"
    sentiment = sentiment_manager.get_aggregated_sentiment(symbol)
    
    print(f"Sentiment for {symbol}:")
    print(f"Average sentiment: {sentiment['average_sentiment']}")
    print(f"Volume: {sentiment['sentiment_volume']}")
    print(f"Trend: {sentiment['sentiment_trend']}")
    print(f"Sources: {sentiment['sources']}")
    print("Source details:")
    for source, data in sentiment['source_sentiments'].items():
        print(f"  {source}: {data['average_sentiment']} ({data['sentiment_volume']} articles)") 