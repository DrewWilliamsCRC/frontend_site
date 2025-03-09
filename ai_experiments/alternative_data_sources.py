#!/usr/bin/env python3
"""
Alternative Data Sources Module

This module provides functions for collecting and analyzing alternative data sources
including web scraping, social media, and satellite imagery to enhance market predictions.
"""

import os
import re
import json
import logging
import time
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
from threading import Thread
from queue import Queue

import pandas as pd # type: ignore
import numpy as np # type: ignore
import requests
from bs4 import BeautifulSoup # type: ignore
import nltk # type: ignore
from nltk.sentiment.vader import SentimentIntensityAnalyzer # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
from nltk.corpus import stopwords # type: ignore
import praw # type: ignore # type: ignore # type: ignore
from collections import Counter
from sklearn.preprocessing import StandardScaler # type: ignore
from PIL import Image # type: ignore
import io
from urllib.request import urlopen
import base64

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('alternative_data_sources')

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(os.path.join(DATA_DIR, "alternative_data"), exist_ok=True)

# Financial news sources to scrape
FINANCIAL_NEWS_SOURCES = [
    {
        "name": "Yahoo Finance",
        "url": "https://finance.yahoo.com/news",
        "article_selector": "div.Ov\(h\) ul li",
        "title_selector": "h3",
        "link_selector": "a",
        "link_prefix": "https://finance.yahoo.com",
        "content_selector": "div.caas-body"
    },
    {
        "name": "CNBC",
        "url": "https://www.cnbc.com/finance/",
        "article_selector": "div.Card-standardBreakerCard",
        "title_selector": "a.Card-title",
        "link_selector": "a.Card-title",
        "link_prefix": "",
        "content_selector": "div.group"
    },
    {
        "name": "Seeking Alpha",
        "url": "https://seekingalpha.com/market-news",
        "article_selector": "div[data-test-id='post-list-item']",
        "title_selector": "h3",
        "link_selector": "a",
        "link_prefix": "https://seekingalpha.com",
        "content_selector": "div.sa-art"
    }
]

# List of financial entities to track (companies, indexes, etc.)
FINANCIAL_ENTITIES = {
    "AAPL": ["Apple", "AAPL", "Tim Cook"],
    "MSFT": ["Microsoft", "MSFT", "Satya Nadella"],
    "GOOGL": ["Google", "Alphabet", "GOOGL", "Sundar Pichai"],
    "AMZN": ["Amazon", "AMZN", "Andy Jassy"],
    "META": ["Meta", "Facebook", "META", "Mark Zuckerberg"],
    "TSLA": ["Tesla", "TSLA", "Elon Musk"],
    "SPY": ["S&P 500", "SPY", "S&P", "Standard & Poor's"],
    "QQQ": ["Nasdaq", "QQQ", "NASDAQ"],
    "BTC": ["Bitcoin", "BTC", "cryptocurrency", "crypto"],
    "ETH": ["Ethereum", "ETH"]
}

# Cache for downloaded articles to avoid re-scraping
ARTICLE_CACHE = {}

class WebScrapingPipeline:
    """Pipeline for scraping and analyzing financial news."""
    
    def __init__(self, sources: List[Dict] = None, use_cache: bool = True, cache_expiry: int = 3600):
        """
        Initialize the web scraping pipeline.
        
        Args:
            sources (List[Dict]): List of news sources to scrape (defaults to FINANCIAL_NEWS_SOURCES)
            use_cache (bool): Whether to use caching for downloaded articles
            cache_expiry (int): Cache expiry time in seconds (default: 1 hour)
        """
        self.sources = sources or FINANCIAL_NEWS_SOURCES
        self.use_cache = use_cache
        self.cache_expiry = cache_expiry
        self.cache_file = os.path.join(DATA_DIR, "alternative_data", "article_cache.json")
        
        # Load cache if it exists
        self._load_cache()
        
        # Set up NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try: 
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Load custom financial sentiment lexicon
        self._load_financial_lexicon()
    
    def _load_financial_lexicon(self):
        """Load custom financial sentiment lexicon to enhance VADER."""
        # Financial domain-specific sentiment words
        financial_lexicon = {
            'rally': 3.0,
            'surge': 2.5,
            'jump': 2.0,
            'gain': 1.5,
            'profit': 2.0,
            'growth': 1.5, 
            'bullish': 3.0,
            'upgrade': 2.0,
            'beat': 1.5,
            'exceed': 1.5,
            'positive': 1.0,
            'outperform': 2.0,
            'upside': 1.5,
            
            'plunge': -3.0,
            'crash': -3.5,
            'tumble': -2.5,
            'drop': -1.5,
            'fall': -1.5,
            'loss': -2.0,
            'bearish': -3.0,
            'downgrade': -2.0,
            'miss': -1.5,
            'negative': -1.0,
            'underperform': -2.0,
            'downside': -1.5,
            'recession': -3.0,
            'bankruptcy': -3.5
        }
        
        # Update VADER lexicon with financial terms
        for word, score in financial_lexicon.items():
            self.sentiment_analyzer.lexicon[word] = score
            
        logger.info("Financial sentiment lexicon loaded")
    
    def _load_cache(self):
        """Load article cache from disk."""
        global ARTICLE_CACHE
        if self.use_cache and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Filter out expired entries
                    now = datetime.now().timestamp()
                    ARTICLE_CACHE = {
                        k: v for k, v in cache_data.items() 
                        if now - v.get('timestamp', 0) < self.cache_expiry
                    }
                logger.info(f"Loaded {len(ARTICLE_CACHE)} articles from cache")
            except Exception as e:
                logger.error(f"Error loading cache: {str(e)}")
                ARTICLE_CACHE = {}
    
    def _save_cache(self):
        """Save article cache to disk."""
        if self.use_cache:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(ARTICLE_CACHE, f)
                logger.info(f"Saved {len(ARTICLE_CACHE)} articles to cache")
            except Exception as e:
                logger.error(f"Error saving cache: {str(e)}")
    
    def scrape_news(self, max_articles_per_source: int = 10) -> List[Dict]:
        """
        Scrape financial news from all sources.
        
        Args:
            max_articles_per_source (int): Maximum number of articles to scrape per source
            
        Returns:
            List[Dict]: List of article data dictionaries
        """
        articles = []
        
        for source in self.sources:
            try:
                source_articles = self._scrape_source(source, max_articles_per_source)
                articles.extend(source_articles)
                logger.info(f"Scraped {len(source_articles)} articles from {source['name']}")
            except Exception as e:
                logger.error(f"Error scraping {source['name']}: {str(e)}")
        
        # Save cache after scraping
        self._save_cache()
        
        return articles
    
    def _scrape_source(self, source: Dict, max_articles: int) -> List[Dict]:
        """
        Scrape a single news source.
        
        Args:
            source (Dict): Source configuration dictionary
            max_articles (int): Maximum number of articles to scrape
            
        Returns:
            List[Dict]: List of article data dictionaries
        """
        articles = []
        
        try:
            # Fetch and parse the main page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(source['url'], headers=headers, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {source['name']}: Status code {response.status_code}")
                return articles
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all article elements
            article_elements = soup.select(source['article_selector'])
            
            # Process articles
            for idx, article_elem in enumerate(article_elements[:max_articles]):
                try:
                    # Extract title
                    title_elem = article_elem.select_one(source['title_selector'])
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text().strip()
                    
                    # Extract link
                    link_elem = article_elem.select_one(source['link_selector'])
                    if not link_elem:
                        continue
                    
                    link = link_elem.get('href', '')
                    if link.startswith('/'):
                        link = source['link_prefix'] + link
                    
                    # Skip if we don't have a valid link
                    if not link:
                        continue
                    
                    # Create article object
                    article = {
                        'source': source['name'],
                        'title': title,
                        'url': link,
                        'published_at': datetime.now().isoformat(),  # Fallback
                        'entities': self._extract_entities(title),
                        'content': None,
                        'sentiment': None
                    }
                    
                    # Extract full article content if not in cache
                    if link not in ARTICLE_CACHE:
                        article['content'] = self._fetch_article_content(link, source['content_selector'])
                        # Skip articles with no content
                        if not article['content']:
                            continue
                    else:
                        article['content'] = ARTICLE_CACHE[link]['content']
                    
                    # Add to articles list
                    articles.append(article)
                    
                    # Sleep briefly to avoid rate limiting
                    time.sleep(random.uniform(0.5, 1.5))
                    
                except Exception as e:
                    logger.error(f"Error processing article: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Error scraping source {source['name']}: {str(e)}")
            
        return articles
    
    def _fetch_article_content(self, url: str, content_selector: str) -> Optional[str]:
        """
        Fetch and extract the content of an article.
        
        Args:
            url (str): URL of the article
            content_selector (str): CSS selector for content element
            
        Returns:
            Optional[str]: Article content or None if error
        """
        global ARTICLE_CACHE
        
        # Check cache first
        if self.use_cache and url in ARTICLE_CACHE:
            cache_entry = ARTICLE_CACHE[url]
            # Check if cache entry is still valid
            if datetime.now().timestamp() - cache_entry.get('timestamp', 0) < self.cache_expiry:
                return cache_entry['content']
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch article {url}: Status code {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract article content
            content_elem = soup.select_one(content_selector)
            if not content_elem:
                logger.warning(f"No content found for {url}")
                return None
            
            # Get all paragraph text
            paragraphs = content_elem.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs])
            
            # Store in cache
            if self.use_cache:
                ARTICLE_CACHE[url] = {
                    'content': content,
                    'timestamp': datetime.now().timestamp()
                }
            
            return content
            
        except Exception as e:
            logger.error(f"Error fetching article content for {url}: {str(e)}")
            return None
    
    def _extract_entities(self, text: str) -> Dict[str, int]:
        """
        Extract financial entities mentioned in text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, int]: Dictionary of entity symbols and occurrence count
        """
        entities = {}
        text = text.lower()
        
        for symbol, keywords in FINANCIAL_ENTITIES.items():
            count = 0
            for keyword in keywords:
                count += len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text))
            
            if count > 0:
                entities[symbol] = count
        
        return entities
    
    def analyze_sentiment(self, articles: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for a list of articles.
        
        Args:
            articles (List[Dict]): List of article dictionaries with content
            
        Returns:
            List[Dict]: Articles with sentiment scores added
        """
        for article in articles:
            if not article.get('content'):
                continue
                
            # Extract sentiment using VADER
            sentiment_scores = self.sentiment_analyzer.polarity_scores(article['content'])
            
            # Add sentiment to article
            article['sentiment'] = {
                'compound': sentiment_scores['compound'],
                'positive': sentiment_scores['pos'],
                'negative': sentiment_scores['neg'],
                'neutral': sentiment_scores['neu'],
                'classification': self._classify_sentiment(sentiment_scores['compound'])
            }
        
        return articles
    
    def _classify_sentiment(self, compound_score: float) -> str:
        """
        Classify sentiment based on compound score.
        
        Args:
            compound_score (float): VADER compound sentiment score
            
        Returns:
            str: Sentiment classification
        """
        if compound_score >= 0.5:
            return 'very_positive'
        elif compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.5:
            return 'very_negative'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def generate_entity_sentiment(self, articles: List[Dict]) -> Dict[str, Dict]:
        """
        Generate aggregated sentiment for each financial entity.
        
        Args:
            articles (List[Dict]): List of articles with sentiment and entities
            
        Returns:
            Dict[str, Dict]: Entity sentiment dictionary
        """
        entity_sentiment = {}
        
        # Initialize entity sentiment
        for symbol in FINANCIAL_ENTITIES.keys():
            entity_sentiment[symbol] = {
                'article_count': 0,
                'avg_sentiment': 0,
                'sentiment_sum': 0,
                'sentiment_count': 0,
                'very_positive': 0,
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'very_negative': 0,
                'recent_headlines': []
            }
        
        # Aggregate sentiment for each entity
        for article in articles:
            if not article.get('entities') or not article.get('sentiment'):
                continue
                
            for symbol, count in article['entities'].items():
                if symbol not in entity_sentiment:
                    continue
                    
                entity_sentiment[symbol]['article_count'] += 1
                entity_sentiment[symbol]['sentiment_sum'] += article['sentiment']['compound']
                entity_sentiment[symbol]['sentiment_count'] += 1
                entity_sentiment[symbol][article['sentiment']['classification']] += 1
                
                # Add headline to recent headlines
                if len(entity_sentiment[symbol]['recent_headlines']) < 5:
                    entity_sentiment[symbol]['recent_headlines'].append({
                        'title': article['title'],
                        'source': article['source'],
                        'url': article['url'],
                        'sentiment': article['sentiment']['compound']
                    })
        
        # Calculate average sentiment
        for symbol, data in entity_sentiment.items():
            if data['sentiment_count'] > 0:
                data['avg_sentiment'] = data['sentiment_sum'] / data['sentiment_count']
        
        return entity_sentiment
    
    def save_entity_sentiment(self, entity_sentiment: Dict[str, Dict]) -> str:
        """
        Save entity sentiment data to disk.
        
        Args:
            entity_sentiment (Dict[str, Dict]): Entity sentiment data
            
        Returns:
            str: Path to saved file
        """
        # Add timestamp
        data = {
            'generated_at': datetime.now().isoformat(),
            'entities': entity_sentiment
        }
        
        # Save to file
        output_file = os.path.join(DATA_DIR, "alternative_data", "entity_sentiment.json")
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved entity sentiment data to {output_file}")
        
        return output_file
    
    def run_pipeline(self, max_articles_per_source: int = 10) -> Dict[str, Dict]:
        """
        Run the complete web scraping pipeline.
        
        Args:
            max_articles_per_source (int): Maximum articles to scrape per source
            
        Returns:
            Dict[str, Dict]: Entity sentiment dictionary
        """
        logger.info("Running web scraping pipeline...")
        
        # Scrape news
        articles = self.scrape_news(max_articles_per_source)
        
        # Analyze sentiment
        articles_with_sentiment = self.analyze_sentiment(articles)
        
        # Generate entity sentiment
        entity_sentiment = self.generate_entity_sentiment(articles_with_sentiment)
        
        # Save entity sentiment
        self.save_entity_sentiment(entity_sentiment)
        
        logger.info(f"Completed web scraping pipeline, analyzed {len(articles)} articles")
        
        return entity_sentiment

# Social Media Analysis class (placeholder for now)
class SocialMediaAnalysis:
    """Analysis of social media data for market sentiment."""
    
    def __init__(self, 
                 reddit_client_id: Optional[str] = None,
                 reddit_client_secret: Optional[str] = None, 
                 reddit_user_agent: Optional[str] = None):
        """
        Initialize social media analysis.
        
        Args:
            reddit_client_id (Optional[str]): Reddit API client ID
            reddit_client_secret (Optional[str]): Reddit API client secret
            reddit_user_agent (Optional[str]): Reddit API user agent
        """
        self.reddit = None
        self.cache_file = os.path.join(DATA_DIR, "alternative_data", "reddit_cache.json")
        self.output_file = os.path.join(DATA_DIR, "alternative_data", "reddit_sentiment.json")
        
        # Try to initialize Reddit API if credentials provided
        if reddit_client_id and reddit_client_secret and reddit_user_agent:
            try:
                self.reddit = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent=reddit_user_agent
                )
                logger.info("Reddit API initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Reddit API: {str(e)}")
        else:
            logger.warning("Reddit API credentials not provided")
        
        # Initialize sentiment analyzer
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Load financial lexicon
        self._load_financial_lexicon()
        
    def _load_financial_lexicon(self):
        """Load custom financial sentiment lexicon to enhance VADER."""
        # Financial domain-specific sentiment words
        financial_lexicon = {
            'rally': 3.0,
            'surge': 2.5,
            'jump': 2.0,
            'gain': 1.5,
            'profit': 2.0,
            'growth': 1.5, 
            'bullish': 3.0,
            'upgrade': 2.0,
            'beat': 1.5,
            'exceed': 1.5,
            'positive': 1.0,
            'outperform': 2.0,
            'upside': 1.5,
            'moon': 3.5,    # Reddit slang
            'tendies': 2.5, # Reddit slang
            'rocket': 3.0,  # 🚀 emoji reference
            'diamond': 2.0, # 💎🙌 reference
            'hold': 1.5,    # HODL reference
            
            'plunge': -3.0,
            'crash': -3.5,
            'tumble': -2.5,
            'drop': -1.5,
            'fall': -1.5,
            'loss': -2.0,
            'bearish': -3.0,
            'downgrade': -2.0,
            'miss': -1.5,
            'negative': -1.0,
            'underperform': -2.0,
            'downside': -1.5,
            'recession': -3.0,
            'bankruptcy': -3.5,
            'bagholder': -2.5, # Reddit slang
            'puts': -1.5,      # Options reference
            'short': -2.0,     # Short selling reference
            'dump': -3.0       # Reddit slang
        }
        
        # Add these terms to the sentiment analyzer's lexicon
        for word, score in financial_lexicon.items():
            self.sentiment_analyzer.lexicon[word] = score
            
        logger.info("Financial sentiment lexicon loaded for social media analysis")
    
    def analyze_reddit(self, 
                      subreddits: List[str] = None, 
                      limit: int = 100, 
                      time_filter: str = 'day', 
                      submission_limit: int = 20) -> Dict[str, Any]:
        """
        Analyze Reddit posts and comments for market sentiment.
        
        Args:
            subreddits (List[str]): List of subreddits to analyze
            limit (int): Maximum number of submissions to retrieve
            time_filter (str): Time filter for submissions ('day', 'week', 'month', etc.)
            submission_limit (int): Maximum number of comments to analyze per submission
            
        Returns:
            Dict[str, Any]: Reddit sentiment data
        """
        if self.reddit is None:
            logger.error("Reddit API not initialized")
            return self._load_cached_reddit_data()
        
        if subreddits is None:
            subreddits = ['wallstreetbets', 'investing', 'stocks', 'options', 'cryptocurrency']
        
        # Initialize results structure
        results = {
            'generated_at': datetime.now().isoformat(),
            'analyzed_posts': 0,
            'analyzed_comments': 0,
            'subreddits': {},
            'entities': {}
        }
        
        # Initialize entity sentiment
        for symbol in FINANCIAL_ENTITIES.keys():
            results['entities'][symbol] = {
                'mentions': 0,
                'sentiment_sum': 0.0,
                'avg_sentiment': 0.0,
                'posts': [],
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        # Process each subreddit
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Initialize subreddit data
                results['subreddits'][subreddit_name] = {
                    'submission_count': 0,
                    'comment_count': 0,
                    'avg_sentiment': 0.0,
                    'sentiment_sum': 0.0,
                    'top_entities': [],
                    'top_posts': []
                }
                
                # Get hot submissions
                for submission in subreddit.top(time_filter=time_filter, limit=limit):
                    try:
                        # Skip stickied posts (usually mod posts)
                        if submission.stickied:
                            continue
                        
                        # Process submission
                        submission_data = self._process_reddit_submission(submission, submission_limit)
                        
                        # Update subreddit stats
                        results['subreddits'][subreddit_name]['submission_count'] += 1
                        results['subreddits'][subreddit_name]['comment_count'] += submission_data['comment_count']
                        results['subreddits'][subreddit_name]['sentiment_sum'] += submission_data['sentiment']['compound']
                        
                        # Update entity mentions and sentiment
                        for symbol, data in submission_data['entities'].items():
                            if symbol in results['entities']:
                                results['entities'][symbol]['mentions'] += data['count']
                                results['entities'][symbol]['sentiment_sum'] += data['sentiment_sum']
                                
                                # Add to entity posts if significant
                                if data['count'] > 0 and len(results['entities'][symbol]['posts']) < 10:
                                    results['entities'][symbol]['posts'].append({
                                        'title': submission.title,
                                        'url': f"https://www.reddit.com{submission.permalink}",
                                        'score': submission.score,
                                        'sentiment': data['avg_sentiment'],
                                        'subreddit': subreddit_name
                                    })
                                
                                # Update sentiment counts
                                sentiment = data['avg_sentiment']
                                if sentiment >= 0.05:
                                    results['entities'][symbol]['positive_count'] += 1
                                elif sentiment <= -0.05:
                                    results['entities'][symbol]['negative_count'] += 1
                                else:
                                    results['entities'][symbol]['neutral_count'] += 1
                        
                        # Add to top posts if score is high
                        if len(results['subreddits'][subreddit_name]['top_posts']) < 5 or submission.score > min(p['score'] for p in results['subreddits'][subreddit_name]['top_posts']):
                            post_data = {
                                'title': submission.title,
                                'url': f"https://www.reddit.com{submission.permalink}",
                                'score': submission.score,
                                'comment_count': submission_data['comment_count'],
                                'sentiment': submission_data['sentiment']['compound'],
                                'entities': [s for s, d in submission_data['entities'].items() if d['count'] > 0]
                            }
                            
                            # Add to top posts and keep only top 5
                            results['subreddits'][subreddit_name]['top_posts'].append(post_data)
                            results['subreddits'][subreddit_name]['top_posts'] = sorted(
                                results['subreddits'][subreddit_name]['top_posts'], 
                                key=lambda x: x['score'], 
                                reverse=True
                            )[:5]
                        
                        results['analyzed_posts'] += 1
                        results['analyzed_comments'] += submission_data['comment_count']
                    
                    except Exception as e:
                        logger.error(f"Error processing Reddit submission: {str(e)}")
                        continue
                
                # Calculate average sentiment for subreddit
                if results['subreddits'][subreddit_name]['submission_count'] > 0:
                    results['subreddits'][subreddit_name]['avg_sentiment'] = results['subreddits'][subreddit_name]['sentiment_sum'] / results['subreddits'][subreddit_name]['submission_count']
                
                # Calculate top entities for subreddit
                entity_counts = {symbol: data['mentions'] for symbol, data in results['entities'].items() if data['mentions'] > 0}
                results['subreddits'][subreddit_name]['top_entities'] = sorted(
                    entity_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
                
            except Exception as e:
                logger.error(f"Error analyzing Reddit subreddit {subreddit_name}: {str(e)}")
                continue
        
        # Calculate average sentiment for each entity
        for symbol, data in results['entities'].items():
            if data['mentions'] > 0:
                data['avg_sentiment'] = data['sentiment_sum'] / data['mentions']
        
        # Save results to file
        try:
            with open(self.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved Reddit sentiment data to {self.output_file}")
            
            # Update cache
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().timestamp(),
                    'data': results
                }, f)
        except Exception as e:
            logger.error(f"Error saving Reddit sentiment data: {str(e)}")
        
        return results
    
    def _process_reddit_submission(self, submission, comment_limit: int = 20) -> Dict[str, Any]:
        """
        Process a Reddit submission and its comments.
        
        Args:
            submission: PRAW submission object
            comment_limit (int): Maximum number of comments to process
            
        Returns:
            Dict[str, Any]: Processed submission data
        """
        # Initialize submission data
        submission_data = {
            'title': submission.title,
            'score': submission.score,
            'created_utc': submission.created_utc,
            'comment_count': 0,
            'entities': {symbol: {'count': 0, 'sentiment_sum': 0.0, 'avg_sentiment': 0.0} for symbol in FINANCIAL_ENTITIES},
            'sentiment': {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        }
        
        # Analyze submission title and text
        combined_text = submission.title
        if submission.selftext and len(submission.selftext) > 0:
            combined_text += " " + submission.selftext
        
        # Extract entities from title and text
        entities = self._extract_entities(combined_text)
        for symbol, count in entities.items():
            submission_data['entities'][symbol]['count'] += count
        
        # Analyze sentiment of title and text
        if combined_text:
            sentiment = self.sentiment_analyzer.polarity_scores(combined_text)
            submission_data['sentiment'] = sentiment
            
            # Update entity sentiment
            for symbol, count in entities.items():
                if count > 0:
                    submission_data['entities'][symbol]['sentiment_sum'] += sentiment['compound']
        
        # Expand comment forest and sort by score
        submission.comments.replace_more(limit=0)
        comments = list(submission.comments)
        comments.sort(key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
        
        # Process top comments
        for comment in comments[:comment_limit]:
            try:
                # Skip removed or deleted comments
                if not comment.body or comment.body in ['[removed]', '[deleted]']:
                    continue
                
                # Extract entities from comment
                comment_entities = self._extract_entities(comment.body)
                
                # Update entity counts
                for symbol, count in comment_entities.items():
                    submission_data['entities'][symbol]['count'] += count
                
                # Analyze comment sentiment
                comment_sentiment = self.sentiment_analyzer.polarity_scores(comment.body)
                
                # Update entity sentiment
                for symbol, count in comment_entities.items():
                    if count > 0:
                        submission_data['entities'][symbol]['sentiment_sum'] += comment_sentiment['compound']
                
                submission_data['comment_count'] += 1
                
            except Exception as e:
                logger.error(f"Error processing Reddit comment: {str(e)}")
                continue
        
        # Calculate average sentiment for each entity
        for symbol, data in submission_data['entities'].items():
            if data['count'] > 0:
                data['avg_sentiment'] = data['sentiment_sum'] / data['count']
        
        return submission_data
    
    def _extract_entities(self, text: str) -> Dict[str, int]:
        """
        Extract financial entities mentioned in text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, int]: Dictionary of entity symbols and occurrence count
        """
        entities = {}
        
        if not text:
            return entities
            
        text = text.lower()
        
        # Special handling for emoji and slang used on Reddit for stocks
        text = text.replace('🚀', ' rocket ')
        text = text.replace('💎', ' diamond ')
        text = text.replace('🙌', ' hands ')
        text = text.replace('🦍', ' ape ')
        text = text.replace('📈', ' up ')
        text = text.replace('📉', ' down ')
        
        for symbol, keywords in FINANCIAL_ENTITIES.items():
            count = 0
            for keyword in keywords:
                # For tickers, match them as standalone words to avoid false positives
                if keyword.isupper() and len(keyword) <= 5:
                    count += len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text))
                    # Also match with $ prefix (common on Reddit)
                    count += len(re.findall(r'\$' + re.escape(keyword.lower()) + r'\b', text))
                else:
                    count += len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text))
            
            if count > 0:
                entities[symbol] = count
        
        return entities
    
    def _load_cached_reddit_data(self) -> Dict[str, Any]:
        """
        Load cached Reddit data if available.
        
        Returns:
            Dict[str, Any]: Cached Reddit data or empty result
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                
                # Check if cache is recent (less than 6 hours old)
                if datetime.now().timestamp() - cache.get('timestamp', 0) < 6 * 3600:
                    logger.info("Using cached Reddit data")
                    return cache['data']
            except Exception as e:
                logger.error(f"Error loading cached Reddit data: {str(e)}")
        
        # Return empty result if no cache
        return {
            'generated_at': datetime.now().isoformat(),
            'analyzed_posts': 0,
            'analyzed_comments': 0,
            'subreddits': {},
            'entities': {symbol: {'mentions': 0, 'avg_sentiment': 0.0} for symbol in FINANCIAL_ENTITIES}
        }
    
    def get_reddit_sentiment(self, max_age_hours: int = 6) -> Dict[str, Any]:
        """
        Get Reddit sentiment data, either from cache or by running analysis.
        
        Args:
            max_age_hours (int): Maximum age of cached data in hours
            
        Returns:
            Dict[str, Any]: Reddit sentiment data
        """
        # Check if cached data is available and recent
        if os.path.exists(self.output_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.output_file))
            if file_age < timedelta(hours=max_age_hours):
                try:
                    with open(self.output_file, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading Reddit sentiment data: {str(e)}")
        
        # If no Reddit API access, return cached data regardless of age
        if self.reddit is None:
            return self._load_cached_reddit_data()
        
        # Run Reddit analysis
        return self.analyze_reddit()

# Function to get Reddit sentiment
def get_reddit_sentiment(
    reddit_client_id: Optional[str] = None,
    reddit_client_secret: Optional[str] = None,
    reddit_user_agent: Optional[str] = None,
    max_age_hours: int = 6
) -> Dict[str, Any]:
    """
    Get Reddit sentiment data.
    
    Args:
        reddit_client_id (Optional[str]): Reddit API client ID
        reddit_client_secret (Optional[str]): Reddit API client secret
        reddit_user_agent (Optional[str]): Reddit API user agent
        max_age_hours (int): Maximum age of cached data in hours
        
    Returns:
        Dict[str, Any]: Reddit sentiment data
    """
    analyzer = SocialMediaAnalysis(
        reddit_client_id=reddit_client_id,
        reddit_client_secret=reddit_client_secret,
        reddit_user_agent=reddit_user_agent
    )
    return analyzer.get_reddit_sentiment(max_age_hours=max_age_hours)

# Main function to run the scraping pipeline
def run_web_scraping():
    """Run the web scraping pipeline and return entity sentiment."""
    pipeline = WebScrapingPipeline()
    entity_sentiment = pipeline.run_pipeline()
    return entity_sentiment

# Function to get entity sentiment (cached if available)
def get_entity_sentiment(max_age_hours: int = 12) -> Dict[str, Dict]:
    """
    Get entity sentiment data.
    
    Args:
        max_age_hours (int): Maximum age of cached data in hours
        
    Returns:
        Dict[str, Dict]: Entity sentiment dictionary
    """
    sentiment_file = os.path.join(DATA_DIR, "alternative_data", "entity_sentiment.json")
    
    # Check if file exists and is recent
    if os.path.exists(sentiment_file):
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(sentiment_file))
        if file_age < timedelta(hours=max_age_hours):
            try:
                with open(sentiment_file, 'r') as f:
                    data = json.load(f)
                    return data['entities']
            except Exception as e:
                logger.error(f"Error loading entity sentiment data: {str(e)}")
    
    # Run pipeline if no recent data
    return run_web_scraping()

# Satellite Imagery Analysis class
class SatelliteImageryAnalysis:
    """Analysis of satellite imagery for economic activity."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize satellite imagery analysis.
        
        Args:
            api_key (Optional[str]): API key for satellite imagery provider
        """
        self.api_key = api_key
        self.cache_dir = os.path.join(DATA_DIR, "alternative_data", "satellite_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.output_file = os.path.join(DATA_DIR, "alternative_data", "satellite_analysis.json")
        logger.info("Satellite Imagery Analysis initialized")
    
    def analyze_retail_parking_lots(self, 
                                   locations: List[Dict[str, Any]] = None,
                                   use_mock: bool = True) -> Dict[str, Any]:
        """
        Analyze retail parking lot satellite imagery to estimate store traffic.
        
        Args:
            locations (List[Dict]): List of retail location dictionaries
            use_mock (bool): Whether to use mock data instead of real API
            
        Returns:
            Dict[str, Any]: Retail traffic analysis results
        """
        if locations is None:
            # Default to major retailers
            locations = [
                {"name": "Walmart Supercenter", "lat": 37.7749, "lon": -122.4194, "ticker": "WMT"},
                {"name": "Target", "lat": 40.7128, "lon": -74.0060, "ticker": "TGT"},
                {"name": "Home Depot", "lat": 33.7490, "lon": -84.3880, "ticker": "HD"},
                {"name": "Costco Wholesale", "lat": 47.6062, "lon": -122.3321, "ticker": "COST"},
                {"name": "Best Buy", "lat": 44.9778, "lon": -93.2650, "ticker": "BBY"}
            ]
        
        # Use mock data if requested or if no API key
        if use_mock or not self.api_key:
            return self._generate_mock_retail_data(locations)
        
        # Initialize results
        results = {
            "generated_at": datetime.now().isoformat(),
            "locations": {}
        }
        
        # Process each location
        for location in locations:
            try:
                location_id = f"{location['name']}_{location['lat']}_{location['lon']}".replace(" ", "_")
                
                # Check cache first
                cache_file = os.path.join(self.cache_dir, f"{location_id}_analysis.json")
                if os.path.exists(cache_file):
                    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
                    if file_age < timedelta(days=7):  # Satellite imagery doesn't change frequently
                        with open(cache_file, 'r') as f:
                            results["locations"][location_id] = json.load(f)
                            logger.info(f"Loaded cached satellite data for {location['name']}")
                            continue
                
                # Fetch satellite imagery
                image_data = self._fetch_satellite_image(location['lat'], location['lon'])
                if not image_data:
                    logger.warning(f"No satellite image available for {location['name']}")
                    continue
                
                # Process image to detect cars
                car_count, occupied_percentage = self._detect_cars_in_parking_lot(image_data)
                
                # Calculate metrics
                location_data = {
                    "name": location["name"],
                    "ticker": location.get("ticker"),
                    "coordinates": {
                        "lat": location["lat"],
                        "lon": location["lon"]
                    },
                    "analysis": {
                        "car_count": car_count,
                        "occupied_percentage": occupied_percentage,
                        "traffic_category": self._categorize_traffic(occupied_percentage),
                        "timestamp": datetime.now().isoformat()
                    },
                    "historical": self._generate_historical_trend(occupied_percentage)
                }
                
                # Save to cache
                with open(cache_file, 'w') as f:
                    json.dump(location_data, f, indent=2)
                
                # Add to results
                results["locations"][location_id] = location_data
                
            except Exception as e:
                logger.error(f"Error analyzing satellite imagery for {location['name']}: {str(e)}")
                continue
        
        # Save overall results
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def analyze_agricultural_areas(self, 
                                  crop_regions: List[Dict[str, Any]] = None,
                                  use_mock: bool = True) -> Dict[str, Any]:
        """
        Analyze agricultural regions to predict crop yields and commodity movements.
        
        Args:
            crop_regions (List[Dict]): List of crop region dictionaries
            use_mock (bool): Whether to use mock data instead of real API
            
        Returns:
            Dict[str, Any]: Agricultural analysis results
        """
        if crop_regions is None:
            # Default to major crop producing regions
            crop_regions = [
                {"name": "Corn Belt", "lat": 41.8781, "lon": -93.0977, "crop": "corn", "ticker": "CORN"},
                {"name": "Wheat Fields", "lat": 47.7511, "lon": -120.7401, "crop": "wheat", "ticker": "WEAT"},
                {"name": "Soybean Valley", "lat": 39.7684, "lon": -86.1581, "crop": "soybeans", "ticker": "SOYB"},
                {"name": "Cotton Country", "lat": 32.7767, "lon": -96.7970, "crop": "cotton", "ticker": "BAL"},
                {"name": "Coffee Region", "lat": -23.5505, "lon": -46.6333, "crop": "coffee", "ticker": "JO"}
            ]
        
        # Use mock data if requested or if no API key
        if use_mock or not self.api_key:
            return self._generate_mock_agriculture_data(crop_regions)
        
        # Initialize results
        results = {
            "generated_at": datetime.now().isoformat(),
            "regions": {}
        }
        
        # Process each crop region
        for region in crop_regions:
            try:
                region_id = f"{region['name']}_{region['crop']}".replace(" ", "_")
                
                # Check cache first
                cache_file = os.path.join(self.cache_dir, f"{region_id}_analysis.json")
                if os.path.exists(cache_file):
                    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
                    if file_age < timedelta(days=7):  # Agricultural images don't change daily
                        with open(cache_file, 'r') as f:
                            results["regions"][region_id] = json.load(f)
                            logger.info(f"Loaded cached agricultural data for {region['name']}")
                            continue
                
                # Fetch satellite imagery
                image_data = self._fetch_satellite_image(region['lat'], region['lon'])
                if not image_data:
                    logger.warning(f"No satellite image available for {region['name']}")
                    continue
                
                # Process image to analyze crop health
                crop_health, vegetation_index = self._analyze_vegetation(image_data)
                
                # Calculate yield prediction
                predicted_yield, yield_change = self._predict_crop_yield(vegetation_index, region['crop'])
                
                # Generate region data
                region_data = {
                    "name": region["name"],
                    "crop": region["crop"],
                    "ticker": region.get("ticker"),
                    "coordinates": {
                        "lat": region["lat"],
                        "lon": region["lon"]
                    },
                    "analysis": {
                        "crop_health": crop_health,
                        "vegetation_index": vegetation_index,
                        "predicted_yield": predicted_yield,
                        "yield_change": yield_change,
                        "timestamp": datetime.now().isoformat()
                    },
                    "price_impact": self._estimate_price_impact(yield_change),
                    "historical": self._generate_yield_trend(predicted_yield)
                }
                
                # Save to cache
                with open(cache_file, 'w') as f:
                    json.dump(region_data, f, indent=2)
                
                # Add to results
                results["regions"][region_id] = region_data
                
            except Exception as e:
                logger.error(f"Error analyzing agricultural data for {region['name']}: {str(e)}")
                continue
        
        # Save overall results
        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _fetch_satellite_image(self, lat: float, lon: float) -> Optional[bytes]:
        """
        Fetch satellite image for given coordinates.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            Optional[bytes]: Image data or None if error
        """
        # This is a placeholder. In a real implementation, you would call a satellite API
        logger.warning("Satellite image fetching not implemented")
        return None
    
    def _detect_cars_in_parking_lot(self, image_data: bytes) -> Tuple[int, float]:
        """
        Use computer vision to detect cars in parking lot image.
        
        Args:
            image_data (bytes): Satellite image data
            
        Returns:
            Tuple[int, float]: Car count and parking lot occupancy percentage
        """
        # This is a placeholder. In a real implementation, you would:
        # 1. Use OpenCV or a pre-trained object detection model to detect cars
        # 2. Count the detected cars
        # 3. Estimate the total parking lot size
        # 4. Calculate the occupancy percentage
        
        # Mock result
        car_count = random.randint(20, 200)
        occupancy = car_count / random.randint(200, 400)
        return car_count, min(occupancy, 1.0)
    
    def _analyze_vegetation(self, image_data: bytes) -> Tuple[str, float]:
        """
        Analyze vegetation health from satellite imagery.
        
        Args:
            image_data (bytes): Satellite image data
            
        Returns:
            Tuple[str, float]: Crop health category and vegetation index
        """
        # This is a placeholder. In a real implementation, you would:
        # 1. Calculate NDVI (Normalized Difference Vegetation Index)
        # 2. Analyze crop health based on NDVI values
        # 3. Return health category and index
        
        # Mock result
        vegetation_index = random.uniform(0.3, 0.8)
        crop_health = "poor" if vegetation_index < 0.4 else "fair" if vegetation_index < 0.6 else "good"
        return crop_health, vegetation_index
    
    def _predict_crop_yield(self, vegetation_index: float, crop_type: str) -> Tuple[float, float]:
        """
        Predict crop yield based on vegetation index.
        
        Args:
            vegetation_index (float): NDVI or similar index
            crop_type (str): Type of crop
            
        Returns:
            Tuple[float, float]: Predicted yield and change from average
        """
        # This is a placeholder. In a real implementation, you would:
        # 1. Use a regression model trained on historical data
        # 2. Consider both the vegetation index and crop type
        # 3. Adjust for weather conditions
        
        # Define base yields for different crops (bushels/acre or similar unit)
        base_yields = {
            "corn": 180.0,
            "wheat": 60.0,
            "soybeans": 55.0,
            "cotton": 900.0,  # lbs/acre
            "coffee": 1500.0  # lbs/acre
        }
        
        # Get base yield for this crop
        base = base_yields.get(crop_type.lower(), 100.0)
        
        # Calculate yield as a function of vegetation index (simplified model)
        # vegetation_index of 0.5 gives the base yield
        # vegetation_index of 0.8 gives 30% above base
        # vegetation_index of 0.3 gives 30% below base
        yield_factor = 1.0 + ((vegetation_index - 0.5) * 1.5)
        predicted_yield = base * yield_factor
        
        # Calculate change from average (simplified)
        yield_change = (yield_factor - 1.0) * 100.0
        
        return predicted_yield, yield_change
    
    def _estimate_price_impact(self, yield_change: float) -> Dict[str, float]:
        """
        Estimate impact on commodity prices based on yield change.
        
        Args:
            yield_change (float): Percentage change in yield
            
        Returns:
            Dict[str, float]: Price impact data
        """
        # Simple inverse relationship model: 
        # - If yield goes up 10%, price goes down approximately 7%
        # - Elasticity varies with magnitude of change (larger changes have non-linear effects)
        
        elasticity = 0.7  # base elasticity
        if abs(yield_change) > 15:
            elasticity = 0.8  # larger elasticity for bigger changes
        
        # Calculate price impact (negative yield change = positive price change)
        price_change = -yield_change * elasticity
        
        # Calculate volatility based on yield change
        volatility = 5 + abs(yield_change) * 0.3  # Base volatility + scaled factor
        
        return {
            "price_change_percent": price_change,
            "price_direction": "up" if price_change > 0 else "down",
            "confidence": max(0.4, min(0.9, 0.7 - abs(yield_change) * 0.01)),  # Lower confidence for extreme changes
            "volatility": volatility
        }
    
    def _categorize_traffic(self, occupancy: float) -> str:
        """
        Categorize retail traffic based on parking lot occupancy.
        
        Args:
            occupancy (float): Percentage of occupied parking spaces
            
        Returns:
            str: Traffic category
        """
        if occupancy < 0.3:
            return "low"
        elif occupancy < 0.6:
            return "moderate"
        elif occupancy < 0.85:
            return "high"
        else:
            return "very_high"
    
    def _generate_historical_trend(self, current_value: float, points: int = 7) -> List[Dict[str, Any]]:
        """
        Generate historical trend data for visualization.
        
        Args:
            current_value (float): Current metric value
            points (int): Number of data points to generate
            
        Returns:
            List[Dict[str, Any]]: Historical trend data
        """
        # In a real implementation, this would use actual historical data
        # For the mock version, we generate realistic-looking trends
        
        # Start with current value and work backwards
        values = [current_value]
        
        # Add some weekly cyclical pattern and slight trend
        trend = random.uniform(-0.05, 0.05)  # Slight up or down trend
        
        for i in range(1, points):
            # Previous value plus trend, cyclical pattern, and some noise
            prev = values[-1]
            week_cycle = 0.1 * math.sin(i * math.pi / 3.5)  # Weekly cycle
            noise = random.uniform(-0.05, 0.05)
            
            new_value = prev + trend + week_cycle + noise
            # Ensure values stay in reasonable bounds
            new_value = max(0.1, min(1.0, new_value))
            values.append(new_value)
        
        # Reverse to get chronological order
        values.reverse()
        
        # Format as dated entries
        now = datetime.now()
        result = []
        
        for i, value in enumerate(values):
            date = now - timedelta(days=points-i-1)
            result.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": value
            })
        
        return result
    
    def _generate_yield_trend(self, current_yield: float, points: int = 12) -> List[Dict[str, Any]]:
        """
        Generate yield trend data for visualization.
        
        Args:
            current_yield (float): Current predicted yield
            points (int): Number of data points to generate
            
        Returns:
            List[Dict[str, Any]]: Yield trend data
        """
        # For agriculture, we use monthly data points
        values = [current_yield]
        
        # Crops typically follow a seasonal pattern
        # Start with current value and work backwards
        trend = random.uniform(-0.02, 0.02)  # Slight overall trend
        
        for i in range(1, points):
            # Previous value plus trend, seasonal pattern, and some noise
            prev = values[-1]
            # Seasonal pattern (1-year cycle)
            seasonal = 0.1 * current_yield * math.sin(i * math.pi / 6.0)
            noise = random.uniform(-0.02, 0.02) * current_yield
            
            new_value = prev + trend * current_yield + seasonal + noise
            # Ensure values stay positive
            new_value = max(current_yield * 0.7, new_value)
            values.append(new_value)
        
        # Reverse to get chronological order
        values.reverse()
        
        # Format as dated entries
        now = datetime.now()
        result = []
        
        for i, value in enumerate(values):
            date = now - timedelta(days=30*(points-i-1))
            result.append({
                "date": date.strftime("%Y-%m"),
                "value": value
            })
        
        return result
    
    def _generate_mock_retail_data(self, locations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate mock retail traffic data for demonstration purposes.
        
        Args:
            locations (List[Dict]): List of retail locations
            
        Returns:
            Dict[str, Any]: Mock retail traffic data
        """
        results = {
            "generated_at": datetime.now().isoformat(),
            "note": "MOCK DATA - For demonstration purposes only",
            "locations": {}
        }
        
        for location in locations:
            location_id = f"{location['name']}_{location['lat']}_{location['lon']}".replace(" ", "_")
            
            # Generate random data that makes sense for the retailer
            if "Walmart" in location["name"]:
                base_traffic = random.uniform(0.6, 0.85)  # Typically busy
            elif "Target" in location["name"]:
                base_traffic = random.uniform(0.55, 0.8)  # Medium-high traffic
            elif "Costco" in location["name"]:
                base_traffic = random.uniform(0.7, 0.9)  # Very busy
            else:
                base_traffic = random.uniform(0.4, 0.7)  # Average traffic
            
            # Add day-of-week effect (current day)
            day_of_week = datetime.now().weekday()
            if day_of_week == 5:  # Saturday
                day_factor = 0.2  # More busy on weekends
            elif day_of_week == 6:  # Sunday
                day_factor = 0.15  # Busy but less than Saturday
            elif day_of_week == 4:  # Friday
                day_factor = 0.1   # Somewhat busy
            else:
                day_factor = -0.05  # Slightly less busy on weekdays
            
            occupancy = min(0.95, max(0.1, base_traffic + day_factor))
            car_count = int(occupancy * random.randint(200, 400))
            
            # Create location data
            results["locations"][location_id] = {
                "name": location["name"],
                "ticker": location.get("ticker"),
                "coordinates": {
                    "lat": location["lat"],
                    "lon": location["lon"]
                },
                "analysis": {
                    "car_count": car_count,
                    "occupied_percentage": occupancy,
                    "traffic_category": self._categorize_traffic(occupancy),
                    "timestamp": datetime.now().isoformat()
                },
                "historical": self._generate_historical_trend(occupancy)
            }
            
            # Add sales prediction
            avg_purchase = {
                "Walmart": random.uniform(50, 80),
                "Target": random.uniform(60, 100),
                "Costco": random.uniform(120, 200),
                "Home Depot": random.uniform(90, 150),
                "Best Buy": random.uniform(100, 200)
            }
            
            store_type = next((k for k in avg_purchase.keys() if k in location["name"]), "Retail")
            avg_transaction = avg_purchase.get(store_type, random.uniform(70, 120))
            
            # Estimate hourly sales based on car count and average transaction
            hourly_sales = car_count * avg_transaction * 0.6 / 2  # Assume 60% conversion, 2 hour avg visit
            
            results["locations"][location_id]["sales_estimate"] = {
                "hourly": int(hourly_sales),
                "daily": int(hourly_sales * 12),  # 12 hours of operation
                "avg_transaction": round(avg_transaction, 2),
                "conversion_rate": random.uniform(0.55, 0.65)
            }
            
            # Add stock impact
            ticker = location.get("ticker")
            if ticker:
                # Compare to average traffic
                avg_traffic = sum(item["value"] for item in results["locations"][location_id]["historical"]) / len(results["locations"][location_id]["historical"])
                traffic_change = (occupancy - avg_traffic) / avg_traffic
                
                # Stock price impact (simplified model)
                results["locations"][location_id]["stock_impact"] = {
                    "ticker": ticker,
                    "traffic_change": round(traffic_change * 100, 2),
                    "estimated_impact": round(traffic_change * random.uniform(0.4, 0.7) * 100, 2),
                    "confidence": random.uniform(0.5, 0.8)
                }
        
        return results
    
    def _generate_mock_agriculture_data(self, crop_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate mock agricultural data for demonstration purposes.
        
        Args:
            crop_regions (List[Dict]): List of crop regions
            
        Returns:
            Dict[str, Any]: Mock agricultural data
        """
        results = {
            "generated_at": datetime.now().isoformat(),
            "note": "MOCK DATA - For demonstration purposes only",
            "regions": {}
        }
        
        for region in crop_regions:
            region_id = f"{region['name']}_{region['crop']}".replace(" ", "_")
            
            # Seasonal adjustment (Northern Hemisphere)
            current_month = datetime.now().month
            if region['lat'] > 0:  # Northern Hemisphere
                if 4 <= current_month <= 6:  # Spring/early summer
                    seasonal_factor = random.uniform(0.5, 0.7)  # Growing season
                elif 7 <= current_month <= 9:  # Late summer/early fall
                    seasonal_factor = random.uniform(0.7, 0.9)  # Peak growing
                elif 10 <= current_month <= 11:  # Fall
                    seasonal_factor = random.uniform(0.4, 0.6)  # Harvest
                else:  # Winter
                    seasonal_factor = random.uniform(0.2, 0.4)  # Dormant
            else:  # Southern Hemisphere (reversed seasons)
                if 10 <= current_month <= 12 or current_month <= 2:  # Spring/early summer
                    seasonal_factor = random.uniform(0.5, 0.7)
                elif 1 <= current_month <= 3:  # Late summer/early fall
                    seasonal_factor = random.uniform(0.7, 0.9)
                elif 4 <= current_month <= 6:  # Fall
                    seasonal_factor = random.uniform(0.4, 0.6)
                else:  # Winter
                    seasonal_factor = random.uniform(0.2, 0.4)
            
            # Add crop-specific adjustments
            crop_factors = {
                "corn": random.uniform(0.9, 1.1),
                "wheat": random.uniform(0.85, 1.15),
                "soybeans": random.uniform(0.95, 1.05),
                "cotton": random.uniform(0.8, 1.2),
                "coffee": random.uniform(0.9, 1.1)
            }
            
            crop_factor = crop_factors.get(region['crop'].lower(), 1.0)
            
            # Calculate vegetation index
            vegetation_index = seasonal_factor * crop_factor
            vegetation_index = max(0.1, min(0.9, vegetation_index))
            
            # Determine crop health
            crop_health = "poor" if vegetation_index < 0.4 else "fair" if vegetation_index < 0.6 else "good"
            
            # Predict yield
            predicted_yield, yield_change = self._predict_crop_yield(vegetation_index, region['crop'])
            
            # Create region data
            results["regions"][region_id] = {
                "name": region["name"],
                "crop": region["crop"],
                "ticker": region.get("ticker"),
                "coordinates": {
                    "lat": region["lat"],
                    "lon": region["lon"]
                },
                "analysis": {
                    "crop_health": crop_health,
                    "vegetation_index": round(vegetation_index, 2),
                    "predicted_yield": round(predicted_yield, 1),
                    "yield_change": round(yield_change, 1),
                    "timestamp": datetime.now().isoformat()
                },
                "price_impact": self._estimate_price_impact(yield_change),
                "historical": self._generate_yield_trend(predicted_yield)
            }
            
            # Add weather risk factors
            weather_risk = random.uniform(0, 1)
            weather_impact = {
                "risk_level": "low" if weather_risk < 0.3 else "medium" if weather_risk < 0.7 else "high",
                "risk_factors": []
            }
            
            # Add potential risk factors based on region and season
            risk_types = ["drought", "flooding", "frost", "pest", "disease", "heat_stress"]
            weights = [0.3, 0.2, 0.1, 0.2, 0.1, 0.1]  # Higher weights = more common
            
            # Adjust weights based on season
            if current_month in [6, 7, 8]:  # Summer (northern hemisphere)
                weights = [0.4, 0.1, 0.0, 0.2, 0.1, 0.2]  # More drought and heat risk
            elif current_month in [12, 1, 2]:  # Winter
                weights = [0.1, 0.3, 0.3, 0.1, 0.1, 0.1]  # More frost risk
            
            # Randomly select 0-3 risk factors
            risk_count = random.choices([0, 1, 2, 3], [0.4, 0.3, 0.2, 0.1])[0]
            selected_risks = random.choices(risk_types, weights=weights, k=risk_count)
            
            for risk in selected_risks:
                severity = random.uniform(0.2, 0.8)
                weather_impact["risk_factors"].append({
                    "type": risk,
                    "severity": round(severity, 2),
                    "potential_yield_impact": round(-severity * random.uniform(5, 15), 1)
                })
            
            results["regions"][region_id]["weather_impact"] = weather_impact
            
        return results

# Function to get satellite retail data
def get_retail_satellite_data(api_key: Optional[str] = None, use_mock: bool = True) -> Dict[str, Any]:
    """
    Get retail satellite imagery analysis data.
    
    Args:
        api_key (Optional[str]): API key for satellite imagery provider
        use_mock (bool): Whether to use mock data
        
    Returns:
        Dict[str, Any]: Retail satellite data
    """
    analyzer = SatelliteImageryAnalysis(api_key=api_key)
    return analyzer.analyze_retail_parking_lots(use_mock=use_mock)

# Function to get agricultural satellite data
def get_agricultural_satellite_data(api_key: Optional[str] = None, use_mock: bool = True) -> Dict[str, Any]:
    """
    Get agricultural satellite imagery analysis data.
    
    Args:
        api_key (Optional[str]): API key for satellite imagery provider
        use_mock (bool): Whether to use mock data
        
    Returns:
        Dict[str, Any]: Agricultural satellite data
    """
    analyzer = SatelliteImageryAnalysis(api_key=api_key)
    return analyzer.analyze_agricultural_areas(use_mock=use_mock)

if __name__ == "__main__":
    # Run web scraping pipeline
    entity_sentiment = run_web_scraping()
    
    # Print summary
    for symbol, data in entity_sentiment.items():
        if data['article_count'] > 0:
            print(f"{symbol}: {data['article_count']} articles, Avg Sentiment: {data['avg_sentiment']:.2f}")
            if data['recent_headlines']:
                print("  Recent headlines:")
                for headline in data['recent_headlines']:
                    print(f"    - {headline['title']} ({headline['sentiment']:.2f})")
            print("") 