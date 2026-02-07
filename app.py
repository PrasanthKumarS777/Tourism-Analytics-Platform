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
    files = {name: pd.read_excel(os.path.join(DATA_PATH, f"{name}.xlsx")) 
             for name in ["City", "User", "Transaction", "Item", "Type"]}
    
    df = files["Transaction"].merge(files["User"], on="UserId", how="left") \
        .merge(files["Item"], on="AttractionId", how="left") \
        .merge(files["Type"], on="AttractionTypeId", how="left") \
        .merge(files["City"].add_prefix("User_"), left_on="CityId", right_on="User_CityId", how="left").dropna()
    return df

@st.cache_resource
def train_models(df):
    feature_cols = ['VisitYear', 'VisitMonth', 'AttractionTypeId', 'ContinentId', 'CountryId']
    
    # Regression
    X_reg = df[feature_cols].copy()
    y_reg = df['Rating'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
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
    
    class_model = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0)
    class_model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, class_model.predict(X_test))
    
    return reg_model, class_model, le, r2, mae, acc, feature_cols, class_cols

@st.cache_data
def get_recommendations(_df, user_id, top_n=10):
    pivot = _df.pivot_table(index='UserId', columns='Attraction', values='Rating').fillna(0)
    sim_df = pd.DataFrame(cosine_similarity(pivot), index=pivot.index, columns=pivot.index)
    
    if user_id not in sim_df.index:
        return pd.DataFrame({'Attraction': ["No data"], 'Avg Rating': [0]})
    
    similar_users = sim_df[user_id].sort_values(ascending=False)[1:6].index
    recs = _df[_df['UserId'].isin(similar_users)].groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(top_n).reset_index()
    recs.columns = ['Attraction', 'Avg Rating']
    return recs.round(2)

def main():
    st.title("🌍 Tourism Experience Analytics Platform")
    st.markdown("**Advanced ML-Powered Tourism Insights & Recommendations**")
    st.markdown("---")
    
    df = load_and_process()
    reg_model, class_model, le, r2, mae, acc, feature_cols, class_cols = train_models(df)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")
        visit_year = st.selectbox("Visit Year", sorted(df['VisitYear'].unique()))
        visit_month = st.selectbox("Visit Month", sorted(df['VisitMonth'].unique()))
        continent = st.selectbox("Continent", sorted(df['ContinentId'].unique()))
        country = st.selectbox("Country", sorted(df['CountryId'].unique()))
        attraction_type = st.selectbox("Attraction Type", sorted(df['AttractionTypeId'].unique()))
        user_id = st.selectbox("User ID", sorted(df['UserId'].unique()))
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Predictions", "🌟 Recommendations", "📈 Analytics"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{len(df):,}")
        col2.metric("Unique Users", f"{df['UserId'].nunique():,}")
        col3.metric("Total Attractions", f"{df['Attraction'].nunique():,}")
        col4.metric("Avg Rating", f"{df['Rating'].mean():.2f}")
        
        c1, c2 = st.columns(2)
        with c1:
            visits = df.groupby('VisitYear').size().reset_index(name='Visits')
            fig = px.line(visits, x='VisitYear', y='Visits', markers=True, title="Visit Trends")
            st.plotly_chart(fig, use_container_width=True, key='visits_chart')
        with c2:
            continent_dist = df['ContinentId'].value_counts().reset_index()
            continent_dist.columns = ['Continent', 'Count']
            fig = px.pie(continent_dist, values='Count', names='Continent', hole=0.4, title="Geographic Distribution")
            st.plotly_chart(fig, use_container_width=True, key='geo_chart')
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⭐ Rating Prediction")
            input_data = pd.DataFrame([[visit_year, visit_month, attraction_type, continent, country]], columns=feature_cols)
            pred_rating = reg_model.predict(input_data)[0]
            
            st.markdown(f"<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; text-align: center;'><h1 style='color: white; font-size: 48px; margin: 0;'>{pred_rating:.2f}</h1><p style='color: white; font-size: 18px;'>out of 5.0</p></div>", unsafe_allow_html=True)
            st.markdown("---")
            m1, m2 = st.columns(2)
            m1.metric("R² Score", f"{r2:.4f}")
            m2.metric("MAE", f"{mae:.4f}")
        
        with col2:
            st.subheader("🧭 Visit Mode Prediction")
            input_data = pd.DataFrame([[visit_year, visit_month, continent, country]], columns=class_cols)
            pred_mode = le.inverse_transform(class_model.predict(input_data))[0]
            
            st.markdown(f"<div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 30px; border-radius: 15px; text-align: center;'><h1 style='color: white; font-size: 36px; margin: 0;'>{pred_mode}</h1><p style='color: white; font-size: 16px;'>Recommended Mode</p></div>", unsafe_allow_html=True)
            st.markdown("---")
            st.metric("Classification Accuracy", f"{acc:.2%}")
    
    with tab3:
        recs = get_recommendations(df, user_id)
        st.subheader(f"Top 10 Attractions for User #{user_id}")
        st.dataframe(recs, use_container_width=True, hide_index=True, height=400)
        
        if recs.iloc[0]['Avg Rating'] > 0:
            fig = px.bar(recs, x='Avg Rating', y='Attraction', orientation='h', 
                        color='Avg Rating', color_continuous_scale='RdYlGn', 
                        title='Recommended Attractions by Rating')
            st.plotly_chart(fig, use_container_width=True, key='rec_chart')
        
        csv = recs.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Recommendations", csv, f'recommendations_{datetime.now().strftime("%Y%m%d")}.csv', 'text/csv')
    
    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            top_types = df['AttractionType'].value_counts().head(10)
            fig = px.bar(x=top_types.values, y=top_types.index, orientation='h', 
                        title='Top 10 Attraction Types', labels={'x': 'Visits', 'y': 'Type'})
            st.plotly_chart(fig, use_container_width=True, key='types_chart')
        with c2:
            city_counts = df['User_CityName'].value_counts().head(10)
            fig = px.bar(x=city_counts.values, y=city_counts.index, orientation='h',
                        title='Top 10 Visited Cities', labels={'x': 'Visits', 'y': 'City'})
            st.plotly_chart(fig, use_container_width=True, key='cities_chart')
        
        c1, c2 = st.columns(2)
        with c1:
            monthly = df.groupby('VisitMonth').size().reset_index(name='Visits')
            fig = px.bar(monthly, x='VisitMonth', y='Visits', title='Monthly Visit Patterns')
            st.plotly_chart(fig, use_container_width=True, key='monthly_chart')
        with c2:
            fig = px.histogram(df, x='Rating', nbins=20, title='Rating Distribution')
            st.plotly_chart(fig, use_container_width=True, key='rating_chart')
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #64748b;'><p><strong>Tourism Analytics Platform</strong> | Powered by ML © 2024</p></div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()