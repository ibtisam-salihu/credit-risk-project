import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yfinance as yf
from pathlib import Path
import sys
import os
from io import BytesIO

# Add resolver to path
sys.path.append(str(Path(__file__).parent))
from resolver.resolver import CompanyResolver

st.set_page_config(
    page_title="UK Credit Risk Scorer",
    page_icon="📊",
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
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #15803d 0%, #16a34a 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-size: 18px;
        font-weight: bold;
    }
    .stDownloadButton button:hover {
        background: linear-gradient(90deg, #16a34a 0%, #22c55e 100%);
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
            
            /* to reduce padding and spacing */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
            
    /* Reduces headings margins */
    h1 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
	
    h2, h3 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
    }
	
    /* Reduces the metric size */
    [data-testid="stMetricValue"] {
        font-size: 32px !important;
    }
    /* Hides Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resolver():
    # Load FAME company resolver (cached for performance)
    csv_path = os.path.join(os.path.dirname(__file__), "data/raw/fame_public_universe.csv")
    return CompanyResolver(csv_path)
	
	
def get_financial_ratios(yahoo_ticker):
    # fetches financial ratios from Yahoo Finance
    try:
        stock = yf.Ticker(yahoo_ticker)
        info = stock.info
		
        ratios = {
            'Current Ratio': round(info.get('currentRatio', 1.5), 2),
            'Debt to Equity': round(info.get('debtToEquity', 80) / 100, 2) if info.get('debtToEquity') else 0.8,
            'Return on Assets': round(info.get('returnOnAssets', 0.085) * 100, 1) if info.get('returnOnAssets') else 8.5,
            'Profit Margin': round(info.get('profitMargins', 0.10) * 100, 1) if info.get('profitMargins') else 10.0,
            'Quick Ratio': round(info.get('quickRatio', 1.2), 2),
            'Interest Coverage': round(info.get('interestCoverage', 4.5), 2) if info.get('interestCoverage') else 4.5
        }
        return ratios
		
    except Exception as e:
        st.warning(f"Could not fetch complete Yahoo Finance data: {e}")
        # Return default values
        return {
            'Current Ratio': 1.5,
            'Debt to Equity': 0.8,
            'Return on Assets': 8.5,
            'Profit Margin': 10.0,
            'Quick Ratio': 1.2,
            'Interest Coverage': 4.5
        }
		
def calculate_credit_score(financials):
    # Calculate credit score from financial ratios only/mock
    
    # Weighted formula based on financial health indicators
    score = (
        financials['Current Ratio'] * 10 +        
        (2 - financials['Debt to Equity']) * 15 +    
        financials['Return on Assets'] * 3 +         
        financials['Profit Margin'] * 2 +            
        financials['Quick Ratio'] * 8 +              
        financials['Interest Coverage'] * 4          
    )
	
    # Ensure score is between 0-100
    score = int(min(100, max(0, score)))
	
    return score
	
def create_export_data(company_name, ticker, credit_score, financials, fame_score):
    # Create CSV data for export
	
    # Summary section
    summary_data = {
        'Metric': ['Company Name', 'Ticker Symbol', 'Credit Score', 'FAME Credit Score', 'Analysis Date'],
        'Value': [
            company_name,
            ticker,
            f"{credit_score}/100",
            f"{fame_score:.0f}" if fame_score else "N/A",
            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
    }
    summary_df = pd.DataFrame(summary_data)
	
    # Financial ratios section
    ratios_data = {
        'Financial Ratio': list(financials.keys()),
        'Value': list(financials.values())
    }
    ratios_df = pd.DataFrame(ratios_data)
	
    # Combine into CSV format
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        ratios_df.to_excel(writer, sheet_name='Financial Ratios', index=False)
		
    return output.getvalue()
	
def plot_credit_score_gauge(score):
    # Create credit score gauge - Red if <50, Green if >=50
	
    fig, ax = plt.subplots(figsize=(4, 2), facecolor='none')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.axis('off')
	
    # different colours based on score 
    color = '#22c55e' if score >= 50 else '#ef4444'  # Green if >=50, Red if <50
    
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
	
    # Score arc (colored)
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
    # Creates horizontal bar chart of financial ratios
	
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='none')
	
    # Extracting data
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
    ax.set_title('Financial Ratios from Yahoo Finance', fontsize=16, color='white', fontweight='bold', pad=20)
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(colors='white', labelsize=11)
    ax.grid(axis='x', alpha=0.2, color='white', linestyle='--')
	
    plt.tight_layout()
    return fig
	
def main():

    # Load resolver
    with st.spinner('Loading database...'):
        resolver = load_resolver()
		
    # Header
    st.markdown("<h1 style='text-align: center; font-size: 44px; margin: 6px;'>UK Credit Risk Scorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 16px; color: #94a3b8; margin: 20px;'>Financial Analysis Powered by Yahoo Finance</p>", unsafe_allow_html=True)
	
    # Search box
    col1, col2, col3 = st.columns([1, 2, 1])
	
    with col2:
        company_input = st.text_input(
            "",
            placeholder="Enter company name or ticker (e.g., Tesco, TSCO)",
            key="search_input",
            label_visibility="collapsed"
        )
        search_button = st.button("Analyse Company", use_container_width=True)
		
    # Analyse when button clicked
    if search_button and company_input:
        company_input = company_input.strip()
	
        with st.spinner('Searching database...'):
            # Search for company in FAME database
            match = resolver.resolve_one(company_input, min_similarity=60.0)
			
            if not match:
                # Company not found - show error and suggestions
                st.error(f"Company '{company_input}' not found in database. Please check the spelling or try the ticker.")
				
                # Show similar matches
                matches = resolver.search(company_input, limit=5)
                if matches:
                    st.info("Did you mean one of these?")
                    for m in matches:
                        st.write(f"• **{m.company_name}** (Ticker: {m.ticker_symbol}) - Similarity: {m.similarity:.0f}%")
                return
	
if  st.markdown(f"<h2 style='text-align:center; margin: 15px 0 10px 0;'>{match.company_name}</h2>", unsafe_allow_html=True): 
		
        #center export button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            export_data = create_export_data(
                match.company_name,
                match.ticker_symbol,
                credit_score,
                financials,
                match.credit_score
            )
            st.download_button(
                label="Export to Excel",
                data=export_data,
                file_name=f"{match.company_name.replace(' ', '_')}_Analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        #two coloums so layout is side by side 
        left_col, right_col = st.columns([1, 1.2], gap="small")
		
        #left sde: Credit Score Gauge
        with left_col:
            with st.container(border=True):
                st.markdown("**Credit Score**")
                fig_gauge = plot_credit_score_gauge(credit_score)
                st.pyplot(fig_gauge, use_container_width=True)
                plt.close()
				
                #metircs 
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Calculated", f"{credit_score}/100")
                with m2:
                    st.metric("FAME", f"{match.credit_score:.0f}" if match.credit_score else "N/A")
					
        #right side : Financial Ratios
        with right_col:
            with st.container(border=True):
                st.markdown("**Financial Ratios**")
                fig_ratios = plot_financial_ratios(financials)
                st.pyplot(fig_ratios, use_container_width=True)
                plt.close()
			
    #additonal indo
st.markdown("")  # Small spacer
info1, info2 = st.columns(2)
with info1:
        st.caption(f"**Yahoo Ticker:** {yahoo_ticker or 'N/A'}")
with info2:
        st.caption(f"**SIC Code:** {match.sic_code or 'N/A'}")

        # Company found
        st.success(f"Success: **{match.company_name}** (Ticker: {match.ticker_symbol})")
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
		
        # Step 1: Fetch financial data
        status_text.text("Calculating...")
        progress_bar.progress(33)
        
        yahoo_ticker = match.yahoo_ticker
        financials = get_financial_ratios(yahoo_ticker) if yahoo_ticker else {}
		
        # Step 2: Calculate credit score
        status_text.text("Calculating credit score...")
        progress_bar.progress(66)
		
        credit_score = calculate_credit_score(financials)
		
        # Complete
        progress_bar.progress(100)
        status_text.text("Results are ready")
		
        # Clear progress indicators after 1 second
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

               #store the session
if 'show_results' not in st.session_state:
        st.session_state.show_results = False
			
        # Display results
st.markdown("---")
		
        # Export button at the top
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
            export_data = create_export_data(
                match.company_name,
                match.ticker_symbol,
                credit_score,
                financials,
                match.credit_score
            )
            st.download_button(
                label="Export Data to Excel?",
                data=export_data,
                file_name=f"{match.company_name.replace(' ', '_')}_Credit_Analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
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
		
        # Additional Info
st.markdown("---")
col1, col2, col3 = st.columns(3)
		
with col1:
            st.metric("FAME Credit Score", f"{match.credit_score:.0f}" if match.credit_score else "N/A")
			
with col2:
            st.metric("Yahoo Ticker", yahoo_ticker or "N/A")
with col3:
            st.metric("Calculated Score", f"{credit_score}/100")
			
			
if __name__ == "__main__":
    main()