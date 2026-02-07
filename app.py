import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error
import plotly.express as px
from datetime import datetime
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Page Config
st.set_page_config(page_title="Tourism Analytics Platform", page_icon="🌍", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    h1 {color: #1e3a8a; font-weight: 700; border-bottom: 3px solid #3b82f6;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(os.getcwd(), "Dataset")

@st.cache_data
def load_and_process():
    try:
        files = {}
        for name in ["City", "User", "Transaction", "Item", "Type"]:
            file_path = os.path.join(DATA_PATH, f"{name}.xlsx")
            if os.path.exists(file_path):
                files[name] = pd.read_excel(file_path)
            else:
                st.error(f"File not found: {file_path}")
                return pd.DataFrame()
        
        # Merge all dataframes
        df = files["Transaction"].merge(files["User"], on="UserId", how="left") \
            .merge(files["Item"], on="AttractionId", how="left") \
            .merge(files["Type"], on="AttractionTypeId", how="left") \
            .merge(files["City"].add_prefix("User_"), left_on="CityId", right_on="User_CityId", how="left")
        
        # Drop rows with missing values
        df = df.dropna()
        
        # Ensure required columns exist
        required_cols = ['Rating', 'VisitMode', 'Attraction', 'UserId']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Required column '{col}' not found in data")
                return pd.DataFrame()
                
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def train_models():
    """Train models with cached data"""
    df = load_and_process()
    
    if df.empty:
        return None, None, None, 0, 0, 0, [], []
    
    feature_cols = ['VisitYear', 'VisitMonth', 'AttractionTypeId', 'ContinentId', 'CountryId']
    
    # Check if all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns for modeling: {missing_cols}")
        return None, None, None, 0, 0, 0, [], []
    
    # Regression
    X_reg = df[feature_cols].copy()
    y_reg = df['Rating'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)  # Reduced for faster training
    reg_model.fit(X_train, y_train)
    
    y_pred = reg_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Classification
    class_cols = ['VisitYear', 'VisitMonth', 'ContinentId', 'CountryId']
    X_class = df[class_cols].copy()
    
    le = LabelEncoder()
    y_class = le.fit_transform(df['VisitMode'])
    
    X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
    
    class_model = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0, n_estimators=50)
    class_model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, class_model.predict(X_test))
    
    return reg_model, class_model, le, r2, mae, acc, feature_cols, class_cols

@st.cache_data
def get_recommendations(user_id, top_n=10):
    """Get recommendations for a user"""
    df = load_and_process()
    
    if df.empty:
        return pd.DataFrame({'Attraction': ["No data available"], 'Avg Rating': [0]})
    
    try:
        # Create user-item matrix
        user_item_matrix = df.pivot_table(index='UserId', columns='Attraction', values='Rating').fillna(0)
        
        # Calculate similarity
        user_similarity = cosine_similarity(user_item_matrix)
        user_similarity_df = pd.DataFrame(user_similarity, 
                                          index=user_item_matrix.index, 
                                          columns=user_item_matrix.index)
        
        if user_id not in user_similarity_df.index:
            return pd.DataFrame({'Attraction': ["User not found"], 'Avg Rating': [0]})
        
        # Find similar users
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6].index
        
        # Get recommendations from similar users
        recommendations = df[df['UserId'].isin(similar_users)].groupby('Attraction')['Rating'] \
            .agg(['mean', 'count']).sort_values(by='mean', ascending=False).head(top_n)
        
        recommendations = recommendations.reset_index()
        recommendations.columns = ['Attraction', 'Avg Rating', 'Rating Count']
        
        return recommendations[['Attraction', 'Avg Rating']].round(2)
    except Exception as e:
        return pd.DataFrame({'Attraction': [f"Error: {str(e)}"], 'Avg Rating': [0]})

def main():
    st.title("🌍 Tourism Experience Analytics Platform")
    st.markdown("**Advanced ML-Powered Tourism Insights & Recommendations**")
    st.markdown("---")
    
    # Load data first
    with st.spinner("Loading data..."):
        df = load_and_process()
    
    if df.empty:
        st.error("⚠️ No data loaded. Please check:")
        st.error("1. Excel files exist in the Dataset folder")
        st.error("2. Files are named: City.xlsx, User.xlsx, Transaction.xlsx, Item.xlsx, Type.xlsx")
        st.error("3. Files contain the required columns")
        return
    
    # Train models
    with st.spinner("Training ML models..."):
        reg_model, class_model, le, r2, mae, acc, feature_cols, class_cols = train_models()
    
    if reg_model is None:
        st.error("⚠️ Failed to train models. Please check your data structure.")
        return
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")
        
        # Get unique values safely
        visit_years = sorted(df['VisitYear'].unique()) if 'VisitYear' in df.columns else [2023]
        visit_months = sorted(df['VisitMonth'].unique()) if 'VisitMonth' in df.columns else list(range(1, 13))
        continents = sorted(df['ContinentId'].unique()) if 'ContinentId' in df.columns else [1]
        countries = sorted(df['CountryId'].unique()) if 'CountryId' in df.columns else [1]
        attraction_types = sorted(df['AttractionTypeId'].unique()) if 'AttractionTypeId' in df.columns else [1]
        user_ids = sorted(df['UserId'].unique()) if 'UserId' in df.columns else [1]
        
        visit_year = st.selectbox("Visit Year", visit_years)
        visit_month = st.selectbox("Visit Month", visit_months)
        continent = st.selectbox("Continent", continents)
        country = st.selectbox("Country", countries)
        attraction_type = st.selectbox("Attraction Type", attraction_types)
        user_id = st.selectbox("User ID", user_ids)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Predictions", "🌟 Recommendations", "📈 Analytics"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Unique Users", f"{df['UserId'].nunique():,}")
        col3.metric("Total Attractions", f"{df['Attraction'].nunique():,}")
        avg_rating = df['Rating'].mean() if 'Rating' in df.columns else 0
        col4.metric("Avg Rating", f"{avg_rating:.2f}")
        
        c1, c2 = st.columns(2)
        with c1:
            if 'VisitYear' in df.columns:
                visits = df.groupby('VisitYear').size().reset_index(name='Visits')
                fig = px.line(visits, x='VisitYear', y='Visits', markers=True, title="Visit Trends")
                st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            if 'ContinentId' in df.columns:
                continent_dist = df['ContinentId'].value_counts().reset_index()
                continent_dist.columns = ['Continent', 'Count']
                fig = px.pie(continent_dist, values='Count', names='Continent', hole=0.4, title="Geographic Distribution")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⭐ Rating Prediction")
            input_data = pd.DataFrame([[visit_year, visit_month, attraction_type, continent, country]], columns=feature_cols)
            pred_rating = reg_model.predict(input_data)[0]
            
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 15px; text-align: center;'>
                <h1 style='color: white; font-size: 48px; margin: 0;'>{pred_rating:.2f}</h1>
                <p style='color: white; font-size: 18px;'>out of 5.0</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            m1, m2 = st.columns(2)
            m1.metric("R² Score", f"{r2:.4f}")
            m2.metric("MAE", f"{mae:.4f}")
        
        with col2:
            st.subheader("🧭 Visit Mode Prediction")
            input_data = pd.DataFrame([[visit_year, visit_month, continent, country]], columns=class_cols)
            pred_mode = le.inverse_transform(class_model.predict(input_data))[0]
            
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 30px; border-radius: 15px; text-align: center;'>
                <h1 style='color: white; font-size: 36px; margin: 0;'>{pred_mode}</h1>
                <p style='color: white; font-size: 16px;'>Recommended Mode</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
            st.metric("Classification Accuracy", f"{acc:.2%}")
    
    with tab3:
        with st.spinner("Generating recommendations..."):
            recs = get_recommendations(user_id)
        
        st.subheader(f"Top 10 Attractions for User #{user_id}")
        st.dataframe(recs, use_container_width=True, hide_index=True, height=400)
        
        if not recs.empty and recs.iloc[0]['Avg Rating'] > 0:
            fig = px.bar(recs, x='Avg Rating', y='Attraction', orientation='h', 
                        color='Avg Rating', color_continuous_scale='RdYlGn', 
                        title='Recommended Attractions by Rating')
            st.plotly_chart(fig, use_container_width=True)
        
        if not recs.empty:
            csv = recs.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Recommendations", 
                csv, 
                f'recommendations_{datetime.now().strftime("%Y%m%d")}.csv', 
                'text/csv'
            )
    
    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            if 'AttractionType' in df.columns:
                top_types = df['AttractionType'].value_counts().head(10)
                fig = px.bar(x=top_types.values, y=top_types.index, orientation='h', 
                            title='Top 10 Attraction Types', labels={'x': 'Visits', 'y': 'Type'})
                st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            if 'User_CityName' in df.columns:
                city_counts = df['User_CityName'].value_counts().head(10)
                fig = px.bar(x=city_counts.values, y=city_counts.index, orientation='h',
                            title='Top 10 Visited Cities', labels={'x': 'Visits', 'y': 'City'})
                st.plotly_chart(fig, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            if 'VisitMonth' in df.columns:
                monthly = df.groupby('VisitMonth').size().reset_index(name='Visits')
                fig = px.bar(monthly, x='VisitMonth', y='Visits', title='Monthly Visit Patterns')
                st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            if 'Rating' in df.columns:
                fig = px.histogram(df, x='Rating', nbins=20, title='Rating Distribution')
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #64748b;'>
            <p><strong>Tourism Analytics Platform</strong> | Powered by ML © 2024</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()