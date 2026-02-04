import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime, timedelta
import yfinance as yf
import requests
from pathlib import Path
import sys
import os

# add resolver to path 
sys.path.append(str(Path(__file__).parent))
from resolver.resolver import CompanyResolver

# Import FinBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(
    page_title="UK Credit Risk Scorer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown("""
<style>
    /* Main background - gradient dark blue to black */
    .stApp {
        background: linear-gradient(135deg, #0a1929 0%, #000000 100%);
        color: white;
    }
    
    /* All text white */
    .stApp, .stMarkdown, .stText, h1, h2, h3, p, span, div {
        color: white !important;
    }
	
    /* Input boxes */
    .stTextInput input {
        background-color: #1a2332;
        color: white;
        border: 2px solid #2d3e50;
        border-radius: 8px;
        font-size: 18px;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #1e3a8a 0%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s;
    }

    .stButton button:hover {
        background: linear-gradient(90deg, #1e40af 0%, #2563eb 100%);
        transform: scale(1.05);
    }
	
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 48px !important;
        font-weight: bold !important;
    }
	
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
	
	
    /* Divider */
    hr {
        border-color: #2d3e50;
    }
	
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    # Load FAME resolver and FinBERT model (cached for performance)
	
    # load fame company resolver
    csv_path = os.path.join(os.path.dirname(__file__), "data/raw/fame_public_universe.csv")
    resolver = CompanyResolver(csv_path)
	
    # Load FinBERT model
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
	
    return resolver, tokenizer, model
	
# Get API keys from Streamlit secrets
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")
ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", "")

def get_financial_ratios(yahoo_ticker):
    # get financial ratios from yahoo
    try:
        stock = yf.Ticker(yahoo_ticker)
        info = stock.info 
		
        ratios = {
            'Current Ratio': round(info.get('currentRatio', 1.5), 2),
            'Debt to Equity': round(info.get('debtToEquity', 80) / 100, 2) if info.get('debtToEquity') else 0.8,
            'Return on Assets': round(info.get('returnOnAssets', 0.085) * 100, 1) if info.get('returnOnAssets') else 8.5,
            'Profit Margin': round(info.get('profitMargins', 0.10) * 100, 1) if info.get('profitMargins') else 10.0,
            'Quick Ratio': round(info.get('quickRatio', 1.2), 2),
            'Interest Coverage': 4.5  # Simplified - harder to calculate
        }
        return ratios
		
    except Exception as e:
        st.warning(f"Could not fetch Yahoo Finance data: {e}")
        # Return default values
        return {
            'Current Ratio': 1.5,
            'Debt to Equity': 0.8,
            'Return on Assets': 8.5,
            'Profit Margin': 10.0,
            'Quick Ratio': 1.2,
            'Interest Coverage': 4.5
        }
def fetch_company_news(company_name, ticker=None):
    # Fetch recent news articles about the company
    articles = []
	
	
    # Try NewsAPI
    try:
        from_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': company_name,
            'from': from_date,
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': NEWS_API_KEY,
            'pageSize': 10
        }
		        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            for article in data.get('articles', [])[:6]:
                if article.get('title'):
                    articles.append({
                        'headline': article['title'],
                        'description': article.get('description', ''),
                        'date': article['publishedAt'][:10]
                    })
    except Exception as e:
        st.warning(f"NewsAPI unavailable: {e}")
    
    # Fallback to mock news if API fails + still looking for api without limitations this helps app not crash 
    if len(articles) == 0:
        articles = [
            {
                'headline': f'{company_name} reports quarterly earnings',
                'description': 'Company announces financial results',
                'date': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
            },
            {
                'headline': f'{company_name} announces strategic initiatives',
                'description': 'Growth plans unveiled',
                'date': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            }
        ]
    return articles[:5]  # Return max 5 for sentiment trennd
	
def analyze_sentiment(text, tokenizer, model):
    # Analyse sentiment using FinBERT model
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # FinBERT outputs: [positive, negative, neutral]
        positive = predictions[0][0].item()
        negative = predictions[0][1].item()
        neutral = predictions[0][2].item()
        # Convert to 0-1 scale (0=negative, 1=positive)
        sentiment = (positive * 1.0) + (neutral * 0.5) + (negative * 0.0)
        
        return round(sentiment, 3)    
    except Exception as e:
        st.warning(f"FinBERT analysis failed: {e}")
        return 0.5  # Return neutral sentiment on error
	
def calculate_credit_score(financials, avg_sentiment):
    # Calculate credit score from financials and sentiment (will be replaced by ML model later)
	
    # Weighted formula combining financial ratios
    fin_score = (
        financials['Current Ratio'] * 10 +
        (2 - financials['Debt to Equity']) * 15 +
        financials['Return on Assets'] * 3 +
        financials['Profit Margin'] * 2 +
        financials['Interest Coverage'] * 5
    )
    # Add sentiment boost
    sentiment_boost = avg_sentiment * 20
    base_score = fin_score + sentiment_boost
    
    # score has to be between 0-100
    score = int(min(100, max(0, base_score)))
    
    return score

def plot_credit_score_gauge(score):
    # Create credit score gauge - Red if <50, Green if >=50
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='none')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis('off')
	
	
    # Determine color based on score threshold
    color = '#22c55e' if score >= 50 else '#ef4444'
	
    # Background arc (gray)
    arc_bg = patches.Wedge(
        center=(50, 0), 
        r=30, 
        theta1=0, 
        theta2=180, 
        width=8,
        facecolor='#1e293b',
        edgecolor='none'
    )
    ax.add_patch(arc_bg)
    
    # Score arc 
    angle = (score / 100) * 180
    arc_score = patches.Wedge(
        center=(50, 0),
        r=30,
        theta1=0,
        theta2=angle,
        width=8,
        facecolor=color,
        edgecolor='none'
    )
    ax.add_patch(arc_score)
    # Score text
    ax.text(50, 15, str(score), 
            ha='center', va='center',
            fontsize=48, fontweight='bold',
            color=color)
			
    ax.text(50, 5, 'out of 100',
            ha='center', va='center',
            fontsize=14, color='white')
			
    # Rating text
    if score >= 80:
        rating = 'Excellent'
    elif score >= 70:
        rating = 'Good'
    elif score >= 50:
        rating = 'Fair'
    elif score >= 30:
        rating = 'Poor'
    else:
        rating = 'High Risk'
		
    ax.text(50, -5, rating,
            ha='center', va='center',
            fontsize=20, fontweight='bold',
            color=color)
			
    plt.tight_layout()
    return fig

def plot_financial_ratios(financials):
    # Create bar charts for financials
	
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
    # Extract data
    metrics = list(financials.keys())
    values = list(financials.values())

    # Gradient blue colors
    colors = ['#1e3a8a', '#1e40af', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd']
	
    # Create horizontal bars
    bars = ax.barh(metrics, values, color=colors, height=0.6)
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax.text(value + max(values) * 0.02, i, f'{value}',
                va='center', fontsize=12, color='white', fontweight='bold')
				
    # Styling
    ax.set_xlabel('Value', fontsize=14, color='white', fontweight='bold')
    ax.set_title('Financial Ratios', fontsize=16, color='white', fontweight='bold', pad=20)
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(colors='white', labelsize=11)
    ax.grid(axis='x', alpha=0.2, color='white', linestyle='--')
    plt.tight_layout()
    return fig
	
def plot_sentiment_trend(sentiment_scores):
    # Create sentiment trend line chart over time

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='none')
	
    # Time labels for last 6 months
 months= ['6 months ago', '5 months ago', '4 months ago', '3 months ago', '2 months ago', 'Last month']
 
    # Pad sentiment scores if needed
    while len(sentiment_scores) < 6:
        sentiment_scores.append(0.5)
		
    # Convert to percentages
    sentiment_pct = [s * 100 for s in sentiment_scores[:6]]
	
    # Plot line with markers
    ax.plot(months, sentiment_pct, 
            color='#3b82f6', linewidth=3, marker='o', 
            markersize=10, markerfacecolor='#60a5fa',
            markeredgecolor='white', markeredgewidth=2)
			
    # Fill area under curve
    ax.fill_between(range(len(months)), sentiment_pct, alpha=0.3, color='#3b82f6')
	
    # Add neutral reference line at 50%
    ax.axhline(y=50, color='#94a3b8', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(len(months)-0.5, 52, 'Neutral', color='#94a3b8', fontsize=10)
	
    # Styling
    ax.set_ylabel('Sentiment Score (%)', fontsize=14, color='white', fontweight='bold')
    ax.set_title('Sentiment Trend Over Time', fontsize=16, color='white', fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(colors='white', labelsize=10)
    ax.grid(axis='y', alpha=0.2, color='white', linestyle='--')
	
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
	
    plt.tight_layout()
    return fig
	
	
def main():
    
    # Load resources
    with st.spinner('Loading AI models...'):
        resolver, tokenizer, model = load_resources()

 # Header
    st.markdown("<h1 style='text-align: center; font-size: 48px; margin-bottom: 10px;'> UK Company Credit Risk Scorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #94a3b8; margin-bottom: 40px;'>AI-Powered Financial Analysis with Yahoo Finance & FinBERT</p>", unsafe_allow_html=True)
	
    # Search box
    col1, col2, col3 = st.columns([1, 2, 1])
	
    with col2:
        company_input = st.text_input(
            "",
            placeholder="Enter company name or ticker (e.g., Tesco, TSCO)",
            key="search_input",
            label_visibility="collapsed"
        )
        search_button = st.button(" Analyze Company", use_container_width=True)
		
    # Analyse when button clicked
    if search_button and company_input:
	
        with st.spinner(' Searching FAME database...'):
            # Search for company in FAME database
            match = resolver.resolve_one(company_input, min_similarity=70.0)
			
            if not match:
                # Company not found - show error and suggestions
                st.error(f" Company '{company_input}' not found in FAME database. Please check the spelling or try the ticker symbol.")
				
                # Show similar matches
                matches = resolver.search(company_input, limit=5)
                if matches:
                    st.info(" Did you mean one of these?")
                    for m in matches:
                        st.write(f"• **{m.company_name}** (Ticker: {m.ticker_symbol}) - Similarity: {m.similarity:.0f}%")
                return
				
        # Company found
        st.success(f" Found: **{match.company_name}** (Ticker: {match.ticker_symbol})")
		
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Fetch financial data
        status_text.text(" Fetching financial data from Yahoo Finance...")
        progress_bar.progress(20)
		
        yahoo_ticker = match.yahoo_ticker
        financials = get_financial_ratios(yahoo_ticker) if yahoo_ticker else {}
		
        # Step 2: Fetch news
        status_text.text(" Fetching recent news articles...")
        progress_bar.progress(40)
        news_articles = fetch_company_news(match.company_name, match.ticker_symbol)
		
        # Step 3: Analyse sentiment with FinBERT
        status_text.text(" Running FinBERT sentiment analysis...")
        progress_bar.progress(60)
        sentiments = []
        for article in news_articles:
            text = f"{article['headline']}. {article['description']}"
            sentiment = analyze_sentiment(text, tokenizer, model)
            sentiments.append(sentiment)
        avg_sentiment = np.mean(sentiments) if sentiments else 0.5
        
        # Step 4: Calculate credit score
        status_text.text(" Calculating credit score...")
        progress_bar.progress(80)
		
	
        credit_score = calculate_credit_score(financials, avg_sentiment)
		
        # Complete
        progress_bar.progress(100)
        status_text.text(" Analysis complete!")
		
        # Clear progress indicators after 1 second
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
		
        # Display results
        st.markdown("---")
        st.markdown(f"<h2 style='text-align: center; margin-bottom: 30px;'>Analysis Results for {match.company_name}</h2>", unsafe_allow_html=True)
		
        # Row 1: Credit Score Gauge
        st.markdown("<h3 style='color: white;'>Credit Score</h3>", unsafe_allow_html=True)
        fig_gauge = plot_credit_score_gauge(credit_score)
        st.pyplot(fig_gauge)
        plt.close()
		
        st.markdown("---")
		
        # Row 2: Financial Ratios
        st.markdown("<h3 style='color: white;'>Financial Ratios</h3>", unsafe_allow_html=True)
        fig_ratios = plot_financial_ratios(financials)
        st.pyplot(fig_ratios)
        plt.close()
		
        st.markdown("---")
		
        # Row 3: Sentiment Trend
        st.markdown("<h3 style='color: white;'>Sentiment Trend</h3>", unsafe_allow_html=True)
        fig_sentiment = plot_sentiment_trend(sentiments)
        st.pyplot(fig_sentiment)
        plt.close()
		
        # Additional Info
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
		
        with col1:
            st.metric("FAME Credit Score", f"{match.credit_score:.0f}" if match.credit_score else "N/A")
        
        with col2:
            st.metric("Average Sentiment", f"{avg_sentiment * 100:.0f}%")
        with col3:
            st.metric("Yahoo Ticker", yahoo_ticker or "N/A")
if __name__ == "__main__":
    main()


