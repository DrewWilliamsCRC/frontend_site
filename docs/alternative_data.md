# Alternative Data Sources

This document outlines the alternative data sources integrated into the AI Dashboard, how to set them up, and how to use them effectively.

## Overview

The alternative data integration provides insights beyond traditional market metrics by analyzing:

1. **News Sentiment Analysis**: Tracks financial news sentiment for entities and markets
2. **Social Media Sentiment**: Monitors Reddit discussions for market insights
3. **Retail Satellite Analysis**: Uses satellite imagery of retail locations to predict sales
4. **Agricultural Satellite Analysis**: Tracks crop conditions to predict commodity prices

## Setup Instructions

### 1. Database Setup

The alternative data features use a dedicated SQLite database to store historical data. Set it up with:

```bash
# Create database structure
python scripts/setup_alt_data_db.py

# Create database with sample data for testing
python scripts/setup_alt_data_db.py --with-sample-data
```

### 2. API Keys Configuration

Add the following to your `.env` file:

```
# Reddit API credentials
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT="FinanceAI/1.0"

# Satellite imagery API (replace with your provider)
SATELLITE_API_KEY=your_api_key
```

### 3. Feature Toggles

You can enable/disable individual alternative data features in `config.py`:

```python
# Alternative Data Configuration
ENABLE_NEWS_SENTIMENT = True
ENABLE_REDDIT_SENTIMENT = True
ENABLE_RETAIL_SATELLITE = True
ENABLE_AGRICULTURAL_SATELLITE = True
```

## API Endpoints

The following API endpoints are available for alternative data:

### News Sentiment

```
GET /api/alternative-data/news-sentiment
```

Returns:
```json
{
  "entities": [
    {
      "name": "Apple",
      "sentiment_score": 0.65,
      "mentions": 42
    },
    ...
  ],
  "headlines": [
    {
      "text": "Market rallies as tech stocks lead gains",
      "source": "CNBC",
      "date": "2023-05-20T14:30:00",
      "sentiment": 0.45
    },
    ...
  ]
}
```

### Reddit Sentiment

```
GET /api/alternative-data/reddit-sentiment
```

Returns:
```json
{
  "top_entities": [
    {
      "name": "Tesla",
      "sentiment": 0.32,
      "mentions": 156
    },
    ...
  ],
  "subreddit_sentiment": {
    "wallstreetbets": 0.12,
    "investing": 0.05,
    "stocks": -0.03
  },
  "top_posts": [
    {
      "title": "DD: Why I think this tech stock will triple",
      "subreddit": "wallstreetbets",
      "upvotes": 3240,
      "date": "2023-05-20T10:15:00",
      "sentiment": 0.75
    },
    ...
  ]
}
```

### Retail Satellite Data

```
GET /api/alternative-data/retail-satellite
```

Returns:
```json
{
  "locations": [
    {
      "retailer": "Walmart",
      "location": "Austin, TX",
      "cars_detected": 245,
      "traffic_change": 12.5
    },
    ...
  ],
  "stock_impact": [
    {
      "symbol": "WMT",
      "company": "Walmart",
      "predicted_impact": 2.3,
      "locations_analyzed": 12
    },
    ...
  ],
  "historical_traffic": [
    {
      "date": "2023-05-20",
      "traffic_index": 112.5
    },
    ...
  ]
}
```

### Agricultural Satellite Data

```
GET /api/alternative-data/agricultural-satellite
```

Returns:
```json
{
  "regions": [
    {
      "crop": "Corn",
      "region": "Midwest US",
      "vegetation_index": 0.72,
      "yield_change": 8.3
    },
    ...
  ],
  "price_impact": [
    {
      "commodity": "Corn",
      "predicted_impact": 3.2,
      "regions_analyzed": 8
    },
    ...
  ],
  "historical_yields": [
    {
      "date": "2023-05-20",
      "crop": "Corn",
      "yield_index": 105.2
    },
    ...
  ]
}
```

### Combined Summary

```
GET /api/alternative-data/summary
```

Returns:
```json
{
  "top_sentiment_entities": [...],
  "top_reddit_entities": [...],
  "high_traffic_retail": [...],
  "retail_stock_impact": [...],
  "agricultural_yield_changes": [...],
  "agricultural_price_impact": [...]
}
```

## Frontend Components

The alternative data is displayed in the dashboard with the following components:

### News Sentiment Panel

- **Positive Entities**: Lists entities with the most positive sentiment
- **Negative Entities**: Lists entities with the most negative sentiment
- **Sentiment Distribution**: Chart showing sentiment breakdown
- **Recent Headlines**: Latest financial news with sentiment indicators

### Social Media Panel

- **Most Mentioned Entities**: Top entities discussed on Reddit
- **Subreddit Sentiment**: Sentiment analysis by financial subreddit
- **Sentiment Analysis**: Pie chart of sentiment categories
- **Top Posts**: Recent high-impact posts from financial subreddits

### Retail Traffic Panel

- **High Traffic Locations**: Retail locations with notable traffic changes
- **Stock Price Impact**: Predicted impact on retailer stock prices
- **Traffic Trend**: Historical graph of retail traffic patterns

### Agricultural Panel

- **Yield Changes**: Regions with significant crop yield changes
- **Price Impact**: Predicted impact on commodity prices
- **Crop Yield Trend**: Historical graph of crop yields by type

## Extending the System

### Adding New Data Sources

To add a new alternative data source:

1. Create a new class in `ai_experiments/alternative_data_sources.py`
2. Add an API endpoint in `app.py`
3. Create a database schema in `scripts/setup_alt_data_db.py`
4. Add UI components in `templates/ai_dashboard.html`
5. Implement visualization in `static/js/ai-dashboard.js`

### Data Collection Frequency

By default, data is collected:

- News sentiment: Every 3 hours
- Reddit sentiment: Every 6 hours
- Satellite imagery: Daily

You can adjust the collection frequency in the caching parameters of the API endpoints.

## Troubleshooting

### Missing Data

If data isn't showing up in the dashboard:

1. Check API responses in browser developer tools
2. Verify database connection and tables with `sqlite3 instance/alternative_data.db .tables`
3. Ensure API keys are properly configured in `.env`
4. Check Python logs for scraping or parsing errors

### Performance Optimization

For production use, consider:

1. Moving to a more robust database like PostgreSQL
2. Setting up scheduled tasks to pre-process data
3. Implementing more aggressive caching
4. Optimizing image processing for satellite data 