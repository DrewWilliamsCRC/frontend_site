#!/usr/bin/env python3
import os
import sys
import sqlite3
from datetime import datetime, timedelta
import json
import random
import argparse

# Add the project root to the path so we can import the app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def setup_database(db_path, with_sample_data=False):
    """Set up the alternative data database structure"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create tables
    print("Creating tables...")
    
    # News sentiment table
    c.execute('''
    CREATE TABLE IF NOT EXISTS news_sentiment (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        entity TEXT NOT NULL,
        sentiment_score REAL NOT NULL,
        mentions INTEGER NOT NULL,
        UNIQUE(timestamp, entity)
    )
    ''')
    
    # News headlines table
    c.execute('''
    CREATE TABLE IF NOT EXISTS news_headlines (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        headline TEXT NOT NULL,
        source TEXT NOT NULL,
        sentiment_score REAL NOT NULL,
        url TEXT
    )
    ''')
    
    # Reddit sentiment table
    c.execute('''
    CREATE TABLE IF NOT EXISTS reddit_sentiment (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        entity TEXT NOT NULL,
        sentiment_score REAL NOT NULL,
        mentions INTEGER NOT NULL,
        UNIQUE(timestamp, entity)
    )
    ''')
    
    # Subreddit sentiment table
    c.execute('''
    CREATE TABLE IF NOT EXISTS subreddit_sentiment (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        subreddit TEXT NOT NULL,
        sentiment_score REAL NOT NULL,
        UNIQUE(timestamp, subreddit)
    )
    ''')
    
    # Reddit posts table
    c.execute('''
    CREATE TABLE IF NOT EXISTS reddit_posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        title TEXT NOT NULL,
        subreddit TEXT NOT NULL,
        upvotes INTEGER NOT NULL,
        post_id TEXT NOT NULL,
        sentiment_score REAL NOT NULL,
        url TEXT
    )
    ''')
    
    # Retail satellite data table
    c.execute('''
    CREATE TABLE IF NOT EXISTS retail_satellite (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        retailer TEXT NOT NULL,
        location TEXT NOT NULL,
        cars_detected INTEGER NOT NULL,
        traffic_change REAL NOT NULL,
        UNIQUE(timestamp, retailer, location)
    )
    ''')
    
    # Retail traffic historical data
    c.execute('''
    CREATE TABLE IF NOT EXISTS retail_traffic_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        traffic_index REAL NOT NULL,
        UNIQUE(date)
    )
    ''')
    
    # Retail stock impact table
    c.execute('''
    CREATE TABLE IF NOT EXISTS retail_stock_impact (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        symbol TEXT NOT NULL,
        company TEXT NOT NULL,
        predicted_impact REAL NOT NULL,
        locations_analyzed INTEGER NOT NULL,
        UNIQUE(timestamp, symbol)
    )
    ''')
    
    # Agricultural satellite data table
    c.execute('''
    CREATE TABLE IF NOT EXISTS agricultural_satellite (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        crop TEXT NOT NULL,
        region TEXT NOT NULL,
        vegetation_index REAL NOT NULL,
        yield_change REAL NOT NULL,
        UNIQUE(timestamp, crop, region)
    )
    ''')
    
    # Agricultural yield historical data
    c.execute('''
    CREATE TABLE IF NOT EXISTS agricultural_yield_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        crop TEXT NOT NULL,
        yield_index REAL NOT NULL,
        UNIQUE(date, crop)
    )
    ''')
    
    # Agricultural price impact table
    c.execute('''
    CREATE TABLE IF NOT EXISTS agricultural_price_impact (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        commodity TEXT NOT NULL,
        predicted_impact REAL NOT NULL,
        regions_analyzed INTEGER NOT NULL,
        UNIQUE(timestamp, commodity)
    )
    ''')
    
    conn.commit()
    print("Tables created successfully!")
    
    # Add sample data if requested
    if with_sample_data:
        print("Adding sample data...")
        add_sample_data(conn)
        print("Sample data added successfully!")
        
    conn.close()
    print(f"Database setup complete at {db_path}")

def add_sample_data(conn):
    """Add sample data to the database for testing"""
    c = conn.cursor()
    now = datetime.now()
    
    # Sample financial entities
    financial_entities = [
        "Apple", "Microsoft", "Google", "Amazon", "Tesla", "Meta", "NVIDIA",
        "JPMorgan", "Bank of America", "Goldman Sachs", "S&P 500", "NASDAQ",
        "Bitcoin", "Ethereum", "Dow Jones", "Federal Reserve", "Treasury", "Inflation"
    ]
    
    # Sample news sources
    news_sources = ["Yahoo Finance", "CNBC", "Bloomberg", "MarketWatch", "Seeking Alpha", "Wall Street Journal"]
    
    # Sample subreddits
    subreddits = ["wallstreetbets", "investing", "stocks", "cryptocurrency", "finance", "economics"]
    
    # Sample retailers
    retailers = [
        {"name": "Walmart", "locations": ["Austin, TX", "Chicago, IL", "Phoenix, AZ", "Miami, FL", "Seattle, WA"]},
        {"name": "Target", "locations": ["Denver, CO", "Atlanta, GA", "Boston, MA", "San Diego, CA", "Portland, OR"]},
        {"name": "Costco", "locations": ["Houston, TX", "New York, NY", "Los Angeles, CA", "Dallas, TX", "Orlando, FL"]},
        {"name": "Home Depot", "locations": ["Philadelphia, PA", "Las Vegas, NV", "San Antonio, TX", "Charlotte, NC", "Columbus, OH"]},
        {"name": "Best Buy", "locations": ["Minneapolis, MN", "Detroit, MI", "Nashville, TN", "Baltimore, MD", "Sacramento, CA"]}
    ]
    
    # Sample crops and regions
    crops = ["Corn", "Soybeans", "Wheat", "Cotton", "Rice"]
    regions = ["Midwest US", "California", "Texas", "Argentina", "Brazil", "Ukraine", "Australia", "China", "India", "Canada"]
    
    # Add news sentiment data
    for entity in financial_entities:
        # Add current data
        sentiment = random.uniform(-0.8, 0.8)
        mentions = random.randint(5, 100)
        
        c.execute(
            "INSERT OR REPLACE INTO news_sentiment (timestamp, entity, sentiment_score, mentions) VALUES (?, ?, ?, ?)",
            (now, entity, sentiment, mentions)
        )
    
    # Add news headlines
    headlines = [
        "Market rallies as tech stocks lead gains",
        "Federal Reserve signals potential rate cut in next meeting",
        "Inflation data comes in lower than expected",
        "Tech giant announces major product unveiling next week",
        "Banking sector faces pressure amid economic uncertainty",
        "Retail sales exceed expectations for third straight month",
        "Cryptocurrency market sees major correction",
        "Energy prices surge amid global supply concerns",
        "Manufacturing data shows unexpected growth",
        "Consumer confidence hits 5-year high"
    ]
    
    for headline in headlines:
        source = random.choice(news_sources)
        sentiment = random.uniform(-0.5, 0.5)
        timestamp = now - timedelta(hours=random.randint(0, 24))
        
        c.execute(
            "INSERT INTO news_headlines (timestamp, headline, source, sentiment_score, url) VALUES (?, ?, ?, ?, ?)",
            (timestamp, headline, source, sentiment, f"https://example.com/news/{random.randint(10000, 99999)}")
        )
    
    # Add Reddit sentiment data
    for entity in financial_entities[:10]:  # Use a subset
        sentiment = random.uniform(-0.8, 0.8)
        mentions = random.randint(10, 500)
        
        c.execute(
            "INSERT OR REPLACE INTO reddit_sentiment (timestamp, entity, sentiment_score, mentions) VALUES (?, ?, ?, ?)",
            (now, entity, sentiment, mentions)
        )
    
    # Add subreddit sentiment
    for subreddit in subreddits:
        sentiment = random.uniform(-0.6, 0.6)
        
        c.execute(
            "INSERT OR REPLACE INTO subreddit_sentiment (timestamp, subreddit, sentiment_score) VALUES (?, ?, ?)",
            (now, subreddit, sentiment)
        )
    
    # Add Reddit posts
    reddit_posts = [
        "DD: Why I think this tech stock will triple in the next year",
        "Market crash incoming? Here's my analysis",
        "Just bought my first shares ever. Any advice?",
        "Breaking: Major partnership announced between tech giants",
        "Is anyone else concerned about the current market valuation?",
        "My portfolio is up 30% this year - here's what I'm holding",
        "Thoughts on today's Fed announcement?",
        "Why this retail stock is undervalued right now",
        "PSA: Don't panic sell during market corrections",
        "Analysis of the latest earnings season"
    ]
    
    for post in reddit_posts:
        subreddit = random.choice(subreddits)
        upvotes = random.randint(10, 5000)
        sentiment = random.uniform(-0.7, 0.7)
        timestamp = now - timedelta(hours=random.randint(0, 48))
        
        c.execute(
            "INSERT INTO reddit_posts (timestamp, title, subreddit, upvotes, post_id, sentiment_score, url) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (timestamp, post, subreddit, upvotes, f"post_{random.randint(10000, 99999)}", sentiment, f"https://reddit.com/r/{subreddit}/comments/{random.randint(10000, 99999)}")
        )
    
    # Add retail satellite data
    for retailer in retailers:
        for location in retailer["locations"]:
            cars = random.randint(50, 500)
            traffic_change = random.uniform(-20, 30)
            
            c.execute(
                "INSERT OR REPLACE INTO retail_satellite (timestamp, retailer, location, cars_detected, traffic_change) VALUES (?, ?, ?, ?, ?)",
                (now, retailer["name"], location, cars, traffic_change)
            )
    
    # Add retail traffic history
    # Generate 30 days of historical data
    for i in range(30):
        date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        traffic_index = 100 + random.uniform(-15, 15) + 5 * math.sin(i/5)  # Add some cyclicity
        
        c.execute(
            "INSERT OR REPLACE INTO retail_traffic_history (date, traffic_index) VALUES (?, ?)",
            (date, traffic_index)
        )
    
    # Add retail stock impact
    retail_stocks = [
        {"symbol": "WMT", "company": "Walmart"},
        {"symbol": "TGT", "company": "Target"},
        {"symbol": "COST", "company": "Costco"},
        {"symbol": "HD", "company": "Home Depot"},
        {"symbol": "BBY", "company": "Best Buy"},
        {"symbol": "LOW", "company": "Lowe's"},
        {"symbol": "AMZN", "company": "Amazon"}
    ]
    
    for stock in retail_stocks:
        impact = random.uniform(-5, 5)
        locations = random.randint(5, 30)
        
        c.execute(
            "INSERT OR REPLACE INTO retail_stock_impact (timestamp, symbol, company, predicted_impact, locations_analyzed) VALUES (?, ?, ?, ?, ?)",
            (now, stock["symbol"], stock["company"], impact, locations)
        )
    
    # Add agricultural satellite data
    for crop in crops:
        for region in random.sample(regions, 3):  # Each crop in 3 random regions
            vegetation = random.uniform(0.3, 0.9)
            yield_change = random.uniform(-15, 20)
            
            c.execute(
                "INSERT OR REPLACE INTO agricultural_satellite (timestamp, crop, region, vegetation_index, yield_change) VALUES (?, ?, ?, ?, ?)",
                (now, crop, region, vegetation, yield_change)
            )
    
    # Add agricultural yield history
    # Generate 30 days of historical data for each crop
    for crop in crops:
        base_yield = random.uniform(80, 120)
        for i in range(30):
            date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            # Add some trend and seasonality
            yield_index = base_yield + i * 0.2 + 10 * math.sin(i/10) + random.uniform(-5, 5)
            
            c.execute(
                "INSERT OR REPLACE INTO agricultural_yield_history (date, crop, yield_index) VALUES (?, ?, ?)",
                (date, crop, yield_index)
            )
    
    # Add agricultural price impact
    agricultural_commodities = ["Corn", "Wheat", "Soybeans", "Cotton", "Rice", "Coffee", "Sugar"]
    
    for commodity in agricultural_commodities:
        impact = random.uniform(-8, 8)
        regions = random.randint(3, 15)
        
        c.execute(
            "INSERT OR REPLACE INTO agricultural_price_impact (timestamp, commodity, predicted_impact, regions_analyzed) VALUES (?, ?, ?, ?)",
            (now, commodity, impact, regions)
        )
    
    conn.commit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up the alternative data database')
    parser.add_argument('--db-path', default='instance/alternative_data.db', help='Path to the database file')
    parser.add_argument('--with-sample-data', action='store_true', help='Add sample data to the database')
    
    args = parser.parse_args()
    
    # Create directory for database if it doesn't exist
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    
    # Import math module only if sample data is requested
    if args.with_sample_data:
        import math
    
    setup_database(args.db_path, args.with_sample_data) 