import streamlit as st
from data_processor import load_and_process  # Kitchen import
import pandas as pd
import plotly.express as px
from scipy import stats
import numpy as np

# Page config - MUST BE FIRST!
st.set_page_config(page_title="NLP for Market Intelligence (Prototype)", page_icon="🤖", layout="wide")

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
    }
    .stTabs [role="tab"] {
        background-color: #e0e0e0;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title (Will be shown on the main page)
st.title("🤖 NLP for Market Intelligence Dashboard")


# Create Cover Page Toggle
if 'show_cover' not in st.session_state:
    st.session_state.show_cover = True

# --- START OF FOCUSED COVER PAGE ---

# Cover Page
if st.session_state.show_cover:
    
    # Hero Section - Focused on the "Prototype" idea
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 padding: 40px; border-radius: 15px; margin-bottom: 30px; color: white;'>
        <h1 style='text-align: center; font-size: 2.5em; margin: 0;'>
            AI for Market Intelligence: A Prototype
        </h1>
        <h2 style='text-align: center; font-size: 1.5em; margin-top: 10px; font-weight: 300;'>
            نموذج أولي: استخدام الذكاء الاصطناعي في أبحاث السوق
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction - Simple, Bilingual Explanation
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("The Idea (EN)")
        st.markdown("""
        This application is a **prototype** to demonstrate how Natural Language Processing (NLP) can be used for **Market Research and Business Intelligence**.
        
        Instead of manually reading thousands of customer reviews, we use AI to:
        * Automatically categorize products.
        * Extract key customer preferences (like "hydrating" or "long-lasting").
        * Analyze trends in customer satisfaction and popularity.
        
        The data used here is a Kaggle dataset of Nykaa cosmetics reviews, serving as a real-world example.
        """)

    with col2:
        st.header("الفكرة (AR)")
        st.markdown("""
        <div dir="rtl">
        
        هذا التطبيق هو **نموذج أولي (Prototype)** يوضح كيف يمكن استخدام **معالجة اللغات الطبيعية (NLP)** في **أبحاث السوق وذكاء الأعمال**.
        
        بدلاً من قراءة آلاف التقييمات يدوياً، نستخدم الذكاء الاصطناعي من أجل:
        * تصنيف المنتجات تلقائياً.
        * استخلاص تفضيلات العملاء الرئيسية (مثل "ترطيب" أو "يدوم طويلاً").
        * تحليل اتجاهات السوق ومدى رضا العملاء.
        
        البيانات المستخدمة هنا هي مثال تطبيقي من موقع Kaggle، وتتضمن تقييمات عملاء لمنتجات "Nykaa".
        </div>
        """, unsafe_allow_html=True)
    
    # --- CTA BUTTON ---
    st.markdown("---")
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        if st.button("🚀 Explore the Dashboard | استكشف لوحة المعلومات", 
                      use_container_width=True, 
                      type="primary"):
            st.session_state.show_cover = False
            st.rerun()
    st.markdown("---")
    
    st.stop()  # Stop here to show only cover page

# --- END OF NEW COVER PAGE ---


# Quick access button to return to cover
if st.sidebar.button("📖 View Cover Page | عرض الصفحة التعريفية", use_container_width=True):
    st.session_state.show_cover = True
    st.rerun()

st.sidebar.markdown("---")

st.markdown("""
**Dashboard Context:** This analysis is based on customer reviews from Nykaa (2019-2022) as a proof-of-concept. 
We use NLP to categorize products and extract key claims. 
Metrics include review volume (popularity) and average ratings. 
The 'Other' category is excluded for clarity.
""")

# Load from kitchen (cached)
@st.cache_data
def get_data():
    return load_and_process()

try:
    cat_df, claims_df, nlp_model = get_data()
    
    # Limit to 2019-2022 and drop 'Other'
    cat_df = cat_df[(cat_df['Year'] >= 2019) & (cat_df['Year'] <= 2022) & (cat_df['Category'] != 'Other')]
    claims_df = claims_df[(claims_df['Year'] >= 2019) & (claims_df['Year'] <= 2022)]
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    cat_df = pd.DataFrame()
    claims_df = pd.DataFrame()
    nlp_model = None

# Business-Friendly Sidebar
st.sidebar.title("📊 Dashboard Controls")
st.sidebar.markdown("---")

# Only show controls if data is loaded
if not cat_df.empty:
    # Year Selection
    st.sidebar.subheader("📅 Time Period")
    selected_year = st.sidebar.selectbox(
        "Which year would you like to analyze?", 
        options=sorted(cat_df['Year'].unique(), reverse=True),
        help="Select a year to view market insights and trends"
    )
    
    st.sidebar.markdown("---")
    
    # --- START: CORRECTED CATEGORY SELECTION BLOCK ---
    
    st.sidebar.subheader("🏷️ Product Categories")
    all_categories = list(cat_df['Category'].unique())
    
    # Quick select buttons
    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Select All", use_container_width=True):
        st.session_state.selected_cats = all_categories # Set state for the key
        st.rerun()
    if col_b.button("Clear All", use_container_width=True):
        st.session_state.selected_cats = [] # Set state for the key
        st.rerun()
    
    # Initialize session state if needed OR CLEAN IT
    if 'selected_cats' not in st.session_state:
        st.session_state.selected_cats = all_categories
    else:
        # --- THIS IS THE FIX ---
        # Clean the list: only keep categories that are still in all_categories
        st.session_state.selected_cats = [cat for cat in st.session_state.selected_cats if cat in all_categories]
    
    # This was the line causing the error
    # By using key='selected_cats', the widget will:
    # 1. READ its state from st.session_state.selected_cats (which we just cleaned)
    # 2. WRITE its output back to st.session_state.selected_cats
    # The 'default' parameter is no longer needed because the key handles it.
    selected_cats = st.sidebar.multiselect(
        "Choose categories to analyze:", 
        options=all_categories,
        key='selected_cats',  # Use the key to manage state
        help="Filter data to specific product categories"
    )
    
    # --- END: CORRECTED BLOCK ---
    
    st.sidebar.markdown("---")
    
    # Quality Filter
    st.sidebar.subheader("⭐ Quality Filter")
    min_rating = st.sidebar.slider(
        "Show only products with rating above:", 
        min_value=1.0, 
        max_value=5.0, 
        value=1.0, 
        step=0.5,
        help="Filter out lower-rated products to focus on quality"
    )
    
    st.sidebar.markdown("---")
    
    # Display Options
    st.sidebar.subheader("📈 Display Options")
    top_n = st.sidebar.select_slider(
        "Number of top items to show:",
        options=[5, 10, 15, 20],
        value=10,
        help="Adjust how many items appear in ranking charts"
    )
    
    show_insights = st.sidebar.checkbox(
        "Show business insights", 
        value=True,
        help="Display actionable recommendations below each chart"
    )
    
    st.sidebar.markdown("---")
    
    # Info Section
    with st.sidebar.expander("ℹ️ About This Dashboard"):
        st.markdown("""
        **What you'll see:**
        - **Opportunities:** Which categories to prioritize
        - **Category Analysis:** Volume and quality trends
        - **Claim Analysis:** What customers care about
        
        **How to use:**
        1. Select your time period
        2. Choose categories to compare
        3. Adjust filters as needed
        4. Review insights for action items
        """)
    
else:
    st.sidebar.warning("⚠️ No data available. Please check data_processor.py")
    selected_year = None
    selected_cats = []
    min_rating = 1.0
    top_n = 10
    show_insights = True

# Year data (no YoY)
@st.cache_data
def get_year_data(cat_df, claims_df, year, cats, min_rat):
    cat_year = cat_df[
        (cat_df['Year'] == year) & 
        (cat_df['Category'].isin(cats)) & 
        (cat_df['Avg_Rating'] >= min_rat)
    ].copy()
    
    claim_year = claims_df[claims_df['Year'] == year].copy()
    
    if not cat_year.empty:
        def norm(s):
            if len(s) > 1 and s.max() > s.min():
                return (s - s.min()) / (s.max() - s.min())
            else:
                return pd.Series(0.5, index=s.index)
        
        cat_year['Norm_Volume'] = norm(cat_year['Sales_Volume'])
        cat_year['Norm_Rating'] = norm(cat_year['Avg_Rating'])
        cat_year['Opportunity_Score'] = (cat_year['Norm_Volume'] + cat_year['Norm_Rating']) / 2
    
    return cat_year, claim_year

# Process year data if available
if not cat_df.empty and selected_year is not None:
    cat_year, claim_year = get_year_data(cat_df, claims_df, selected_year, selected_cats, min_rating)
    
    if not cat_year.empty:
        ranking = cat_year[['Category', 'Opportunity_Score']].sort_values(
            'Opportunity_Score', 
            ascending=False
        ).round(2)
    else:
        ranking = pd.DataFrame()
else:
    cat_year = pd.DataFrame()
    claim_year = pd.DataFrame()
    ranking = pd.DataFrame()

# KPIs
if not cat_year.empty:
    col1, col2 = st.columns(2)
    col1.metric("Total Reviews", f"{cat_year['Sales_Volume'].sum():,.0f}")
    col2.metric("Avg Rating", f"{cat_year['Avg_Rating'].mean():.1f} ⭐")

# Tabs for organized UI
tab1, tab2, tab3 = st.tabs(["Opportunities", "Category Analysis", "Claim Analysis"])

with tab1:
    st.subheader(f"Opportunities in {selected_year if selected_year else 'N/A'}")
    
    if not ranking.empty:
        st.dataframe(ranking, use_container_width=True)
        
        fig_bar = px.bar(
            ranking, 
            x='Category', 
            y='Opportunity_Score', 
            title="Priority Score (0-1)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("""
        **Chart Explanation:** This bar chart shows a normalized score (0-1) combining review volume (popularity) 
        and average rating (customer satisfaction). Higher scores indicate categories worth prioritizing for 
        marketing or stock.
        """)
        
        if show_insights:
            with st.expander("Insights"):
                st.markdown("- **Top Category:** Focus on the highest score for quick wins.")
                st.markdown("- **Low Scores:** Investigate why (e.g., competition or product issues).")
    else:
        st.info("No category data for this year or filters.")

with tab2:
    st.subheader("Category Analysis")
    
    if not cat_year.empty:
        # Volume by Category
        fig_volume = px.bar(
            cat_year, 
            x='Category', 
            y='Sales_Volume', 
            color='Avg_Rating',
            title="Review Volume by Category", 
            labels={'Sales_Volume': 'Volume'},
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_volume, use_container_width=True)
        
        st.markdown("""
        **Chart Explanation:** Bars represent review count (volume) per category, colored by average rating. 
        Taller bars show more popular categories; warmer colors indicate higher satisfaction.
        """)
        
        if show_insights:
            with st.expander("Insights"):
                st.markdown("- **High volume/low rating:** Potential for product improvements.")
                st.markdown("- **Low volume/high rating:** Niche opportunities to expand.")
        
        # Rating Distribution
        fig_box = px.box(
            cat_year, 
            x='Category', 
            y='Avg_Rating', 
            title="Rating Distribution by Category"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown("""
        **Chart Explanation:** Box plots show rating spread per category. Median line is average; box edges are 
        quartiles. Narrow boxes mean consistent ratings; outliers highlight extremes.
        """)
        
        if show_insights:
            with st.expander("Insights"):
                st.markdown("- **Consistent high ratings:** Reliable categories for promotions.")
                st.markdown("- **Wide spread:** Mixed feedback—analyze reviews for pain points.")
    else:
        st.info("No category data available for selected filters.")

with tab3:
    st.subheader("Claim Analysis")
    
    if not claim_year.empty:
        # Top Claims
        top_claims = claim_year.nlargest(top_n, 'Mention_Count')
        
        fig_claim = px.bar(
            top_claims, 
            x='Claim', 
            y='Mention_Count',
            title=f"Top {top_n} Claim Mentions", 
            color='Avg_Claim_Rating',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_claim, use_container_width=True)
        
        st.markdown("""
        **Chart Explanation:** Bars show how often claims (e.g., 'Hydrating') appear in reviews, colored by 
        average rating for that claim. Taller bars are more mentioned; warmer colors mean higher satisfaction.
        """)
        
        if show_insights:
            with st.expander("Insights"):
                st.markdown("- **Top claims:** Integrate into product marketing.")
                st.markdown("- **Low-rated claims:** Address quality issues.")
        
        # Claim Share
        fig_pie = px.pie(
            claim_year, 
            values='Mention_Count', 
            names='Claim', 
            title="Claim Share"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("""
        **Chart Explanation:** Pie slices represent proportion of mentions per claim. Larger slices are 
        dominant trends in customer feedback.
        """)
        
        # This was 'if show_insighs:' - fixed typo to 'show_insights'
        if show_insights:
            with st.expander("Insights"):
                st.markdown("- **Dominant claims:** Key customer priorities.")
                st.markdown("- **Small slices:** Emerging or niche opportunities.")
        
        # Scatter: Mentions vs Rating
        fig_scatter = px.scatter(
            claim_year, 
            x='Mention_Count', 
            y='Avg_Claim_Rating', 
            color='Claim', 
            size='Mention_Count',
            title="Claims: Mentions vs Rating"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("""
        **Chart Explanation:** Points plot claims by mention count (x-axis, size) and rating (y-axis). 
        Right/high points are popular and well-liked; left/low are areas for improvement.
        """)
        
        if show_insights:
            with st.expander("Insights"):
                st.markdown("- **High mention/high rating:** Strengths to leverage.")
                st.markdown("- **High mention/low rating:** Urgent fixes needed.")
    else:
        st.info("No claim data available for this year.")

# NLP Tester (handle if model is None)
st.sidebar.subheader("Test NLP")
new_prod = st.sidebar.text_input("Product Name:")

if new_prod:
    if nlp_model is not None:
        try:
            pred = nlp_model.predict([new_prod])[0]
            st.sidebar.success(f"Category: {pred}")
        except Exception as e:
            st.sidebar.error(f"Prediction error: {e}")
    else:
        st.sidebar.warning("NLP model unavailable; using heuristic fallback.")

st.caption("Data from Kaggle Nykaa Reviews | NLP via TF-IDF + LR")
