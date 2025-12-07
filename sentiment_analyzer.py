from newsapi import NewsApiClient
from textblob import TextBlob
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# NewsAPI configuration
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', 'YOUR_API_KEY_HERE')

class SentimentAnalyzer:
    """Analyze stock sentiment from news articles"""
    
    def __init__(self, api_key=None):
        """
        Initialize sentiment analyzer
        
        Args:
            api_key: NewsAPI key (optional, will use NEWS_API_KEY if not provided)
        """
        self.api_key = api_key or NEWS_API_KEY
        self.newsapi = None
        
        # Only initialize if API key is valid
        if self.api_key and self.api_key != 'YOUR_API_KEY_HERE':
            try:
                self.newsapi = NewsApiClient(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize NewsAPI: {e}")
    
    def analyze_text_sentiment(self, text):
        """
        Analyze sentiment of a single text using TextBlob
        
        Args:
            text: Text to analyze
        
        Returns:
            Sentiment polarity (-1 to 1)
        """
        if not text:
            return 0.0
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def get_stock_news(self, ticker, days=7):
        """
        Fetch recent news articles for a stock
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
        
        Returns:
            List of news articles
        """
        if not self.newsapi:
            return []
        
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Search for stock-related news
            # Try company name variations
            queries = [ticker]
            
            # Add common company names for popular tickers
            company_names = {
                'AAPL': 'Apple',
                'GOOGL': 'Google',
                'MSFT': 'Microsoft',
                'AMZN': 'Amazon',
                'TSLA': 'Tesla',
                'META': 'Meta Facebook',
                'NVDA': 'NVIDIA',
                'TCS': 'TCS Tata'
            }
            
            if ticker.upper() in company_names:
                queries.append(company_names[ticker.upper()])
            
            all_articles = []
            
            for query in queries:
                try:
                    response = self.newsapi.get_everything(
                        q=query,
                        from_param=from_date.strftime('%Y-%m-%d'),
                        to=to_date.strftime('%Y-%m-%d'),
                        language='en',
                        sort_by='relevancy',
                        page_size=20
                    )
                    
                    if response and 'articles' in response:
                        all_articles.extend(response['articles'])
                except Exception as e:
                    print(f"Error fetching news for {query}: {e}")
                    continue
            
            # Remove duplicates based on title
            seen_titles = set()
            unique_articles = []
            for article in all_articles:
                if article['title'] not in seen_titles:
                    seen_titles.add(article['title'])
                    unique_articles.append(article)
            
            return unique_articles[:15]  # Return top 15 articles
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def analyze_stock_sentiment(self, ticker, days=7):
        """
        Analyze overall sentiment for a stock
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to analyze
        
        Returns:
            Dictionary with sentiment analysis results
        """
        articles = self.get_stock_news(ticker, days)
        
        if not articles:
            return {
                'overall_sentiment': 0.0,
                'sentiment_label': 'Neutral',
                'num_articles': 0,
                'articles': [],
                'daily_sentiment': [],
                'error': 'No news articles found or API key not configured'
            }
        
        # Analyze sentiment for each article
        article_sentiments = []
        processed_articles = []
        
        for article in articles:
            # Combine title and description for better sentiment analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            sentiment = self.analyze_text_sentiment(text)
            
            article_sentiments.append(sentiment)
            processed_articles.append({
                'title': article.get('title', 'No title'),
                'description': article.get('description', 'No description'),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'sentiment': round(sentiment, 3)
            })
        
        # Calculate overall sentiment
        overall_sentiment = sum(article_sentiments) / len(article_sentiments) if article_sentiments else 0.0
        
        # Determine sentiment label
        if overall_sentiment > 0.1:
            sentiment_label = 'Positive'
        elif overall_sentiment < -0.1:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        
        # Calculate daily sentiment trend
        daily_sentiment = self._calculate_daily_sentiment(processed_articles, days)
        
        return {
            'overall_sentiment': round(overall_sentiment, 3),
            'sentiment_label': sentiment_label,
            'num_articles': len(articles),
            'articles': processed_articles[:10],  # Return top 10 for display
            'daily_sentiment': daily_sentiment
        }
    
    def _calculate_daily_sentiment(self, articles, days):
        """Calculate sentiment trend over days"""
        daily_sentiments = {}
        
        for article in articles:
            try:
                pub_date = article['published_at'][:10]  # Get YYYY-MM-DD
                if pub_date not in daily_sentiments:
                    daily_sentiments[pub_date] = []
                daily_sentiments[pub_date].append(article['sentiment'])
            except:
                continue
        
        # Calculate average sentiment per day
        daily_avg = []
        for date in sorted(daily_sentiments.keys()):
            avg_sentiment = sum(daily_sentiments[date]) / len(daily_sentiments[date])
            daily_avg.append({
                'date': date,
                'sentiment': round(avg_sentiment, 3)
            })
        
        return daily_avg


def get_stock_sentiment(ticker, days=7, api_key=None):
    """
    Convenience function to get stock sentiment
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days to analyze
        api_key: NewsAPI key (optional)
    
    Returns:
        Dictionary with sentiment analysis
    """
    analyzer = SentimentAnalyzer(api_key=api_key)
    return analyzer.analyze_stock_sentiment(ticker, days)
