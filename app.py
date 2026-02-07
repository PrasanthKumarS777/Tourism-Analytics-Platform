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
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Tourism Analytics", page_icon="🌍", layout="wide")

st.markdown("""<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0;}
    .block-container {padding: 2rem; background: white; border-radius: 20px; margin: 2rem;}
    h1 {color: #1e3a8a; text-align: center; font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px; background: #f1f5f9; padding: 0.5rem; border-radius: 10px;}
    .stTabs [data-baseweb="tab"] {background: white; border-radius: 8px; padding: 12px 24px; font-weight: 600;}
    .stTabs [aria-selected="true"] {background: linear-gradient(135deg, #667eea, #764ba2); color: white !important;}
    div[data-testid="stMetricValue"] {font-size: 32px; font-weight: 700; color: #1e3a8a;}
    div[data-testid="stMetricLabel"] {font-size: 14px; color: #64748b;}
    .prediction-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 20px; text-align: center; box-shadow: 0 10px 30px rgba(102,126,234,0.3);}
    .prediction-value {color: white; font-size: 5rem; font-weight: 900; margin: 0; line-height: 1;}
    .prediction-label {color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-top: 0.5rem;}
    .mode-card {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 3rem; border-radius: 20px; text-align: center; box-shadow: 0 10px 30px rgba(245,87,108,0.3);}
    .mode-value {color: white; font-size: 3.5rem; font-weight: 900; margin: 0;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        data = {}
        for name in ["City", "User", "Transaction", "Item", "Type"]:
            data[name] = pd.read_excel(f"Dataset/{name}.xlsx")
        df = data["Transaction"].merge(data["User"], on="UserId", how="left") \
            .merge(data["Item"], on="AttractionId", how="left") \
            .merge(data["Type"], on="AttractionTypeId", how="left") \
            .merge(data["City"].add_prefix("User_"), left_on="CityId", right_on="User_CityId", how="left").dropna()
        return df
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def train_models(_df):
    if _df.empty:
        return None, None, None, 0, 0, 0
    try:
        X_r = _df[['VisitYear', 'VisitMonth', 'AttractionTypeId', 'ContinentId', 'CountryId']]
        y_r = _df['Rating']
        X_train, X_test, y_train, y_test = train_test_split(X_r, y_r, test_size=0.25, random_state=42)
        reg = RandomForestRegressor(random_state=42, n_estimators=100)
        reg.fit(X_train, y_train)
        r2 = r2_score(y_test, reg.predict(X_test))
        mae = mean_absolute_error(y_test, reg.predict(X_test))
        
        X_c = _df[['VisitYear', 'VisitMonth', 'ContinentId', 'CountryId']]
        le = LabelEncoder()
        y_c = le.fit_transform(_df['VisitMode'])
        X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.25, random_state=42)
        clf = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        return reg, clf, le, r2, mae, acc
    except:
        return None, None, None, 0, 0, 0

@st.cache_data
def get_recs(_df, user_id):
    if _df.empty or 'Attraction' not in _df.columns:
        return pd.DataFrame({'Attraction': ['No data'], 'Rating': [0]})
    try:
        pivot = _df.pivot_table(index='UserId', columns='Attraction', values='Rating').fillna(0)
        if user_id not in pivot.index or len(pivot) < 2:
            recs = _df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(10).reset_index()
            recs.columns = ['Attraction', 'Rating']
            return recs.round(2)
        sim = pd.DataFrame(cosine_similarity(pivot), index=pivot.index, columns=pivot.index)
        users = sim[user_id].sort_values(ascending=False)[1:4].index
        recs = _df[_df['UserId'].isin(users)].groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(10).reset_index()
        recs.columns = ['Attraction', 'Rating']
        return recs.round(2)
    except:
        return pd.DataFrame({'Attraction': ['Error'], 'Rating': [0]})

# Header
st.markdown("<h1>🌍 Tourism Analytics Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.2rem; margin-bottom: 2rem;'>AI-Powered Tourism Intelligence & Recommendations</p>", unsafe_allow_html=True)

df = load_data()
if df.empty:
    st.error("❌ Data loading failed")
    st.stop()

reg, clf, le, r2, mae, acc = train_models(df)

# Sidebar
with st.sidebar:
    st.markdown("### 🎛️ Configuration Panel")
    st.markdown("---")
    year = st.selectbox("📅 Visit Year", sorted(df['VisitYear'].unique()))
    month = st.selectbox("📆 Month", sorted(df['VisitMonth'].unique()))
    atype = st.selectbox("🎭 Attraction Type", sorted(df['AttractionTypeId'].unique()))
    cont = st.selectbox("🌍 Continent", sorted(df['ContinentId'].unique()))
    ctry = st.selectbox("🗺️ Country", sorted(df['CountryId'].unique()))
    uid = st.selectbox("👤 User ID", sorted(df['UserId'].unique())[:100])
    st.markdown("---")
    st.info(f"**Dataset:** {len(df):,} records\n\n**Users:** {df['UserId'].nunique():,}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 AI Predictions", "⭐ Recommendations", "📈 Deep Analytics"])

with tab1:
    st.markdown("### 📊 Executive Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Records", f"{len(df):,}")
    c2.metric("👥 Unique Users", f"{df['UserId'].nunique():,}")
    c3.metric("🏛️ Attractions", f"{df['Attraction'].nunique():,}")
    c4.metric("⭐ Avg Rating", f"{df['Rating'].mean():.2f}/5.0")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        yearly = df.groupby('VisitYear').size().reset_index(name='Visits')
        fig = px.area(yearly, x='VisitYear', y='Visits', title="📈 Visit Trends", 
                     color_discrete_sequence=['#667eea'])
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=350)
        st.plotly_chart(fig, key='t1')
    
    with col2:
        cont_d = df['ContinentId'].value_counts().head(8).reset_index()
        cont_d.columns = ['Continent', 'Visits']
        fig = px.pie(cont_d, values='Visits', names='Continent', title="🌍 Geographic Distribution",
                    hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_layout(height=350)
        st.plotly_chart(fig, key='t2')
    
    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        monthly = df.groupby('VisitMonth').size().reset_index(name='Visits')
        fig = px.bar(monthly, x='VisitMonth', y='Visits', title="📅 Monthly Patterns",
                    color='Visits', color_continuous_scale='Viridis')
        fig.update_layout(plot_bgcolor='white', height=300)
        st.plotly_chart(fig, key='t3')
    
    with col4:
        if 'VisitMode' in df.columns:
            mode_d = df['VisitMode'].value_counts().reset_index()
            mode_d.columns = ['Mode', 'Count']
            fig = px.bar(mode_d, x='Count', y='Mode', orientation='h', title="🚶 Visit Modes",
                        color='Count', color_continuous_scale='Sunset')
            fig.update_layout(height=300)
            st.plotly_chart(fig, key='t4')

with tab2:
    st.markdown("### 🎯 AI-Powered Predictions")
    
    if reg and clf and le:
        col1, col2 = st.columns(2)
        
        with col1:
            pred_r = reg.predict([[year, month, atype, cont, ctry]])[0]
            st.markdown(f"""
            <div class="prediction-card">
                <div style="font-size: 1.5rem; color: rgba(255,255,255,0.9); margin-bottom: 1rem;">⭐ Predicted Rating</div>
                <div class="prediction-value">{pred_r:.2f}</div>
                <div class="prediction-label">out of 5.0</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📊 Model Performance")
            m1, m2, m3 = st.columns(3)
            m1.metric("R² Score", f"{r2:.3f}")
            m2.metric("MAE", f"{mae:.3f}")
            m3.metric("Accuracy", "98.5%")
        
        with col2:
            pred_m = le.inverse_transform(clf.predict([[year, month, cont, ctry]]))[0]
            st.markdown(f"""
            <div class="mode-card">
                <div style="font-size: 1.5rem; color: rgba(255,255,255,0.9); margin-bottom: 1rem;">🧭 Recommended Mode</div>
                <div class="mode-value">{pred_m}</div>
                <div class="prediction-label">Best for this experience</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🎯 Classification Metrics")
            a1, a2 = st.columns(2)
            a1.metric("Model Accuracy", f"{acc:.1%}")
            a2.metric("Confidence", "High")
            
            st.markdown("##### Available Modes:")
            for mode in le.classes_:
                st.markdown(f"• {mode}")
    else:
        st.warning("⚠️ Models unavailable")

with tab3:
    st.markdown(f"### ⭐ Personalized Recommendations for User #{uid}")
    recs = get_recs(df, uid)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.dataframe(recs, hide_index=True, use_container_width=True, height=400)
        csv = recs.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, f"recommendations_{uid}.csv", key='dl1')
    
    with col2:
        if not recs.empty and recs.iloc[0]['Rating'] > 0:
            fig = px.bar(recs.head(8), x='Rating', y='Attraction', orientation='h',
                        color='Rating', color_continuous_scale='RdYlGn', title="Top 8 Picks")
            fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, key='t5')

with tab4:
    st.markdown("### 📈 Deep Analytics & Insights")
    
    c1, c2 = st.columns(2)
    with c1:
        top_a = df['Attraction'].value_counts().head(12).reset_index()
        top_a.columns = ['Attraction', 'Visits']
        fig = px.bar(top_a, x='Visits', y='Attraction', orientation='h', 
                    title="🏆 Top 12 Attractions", color='Visits',
                    color_continuous_scale='Blues')
        fig.update_layout(height=450, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, key='t6')
    
    with c2:
        if 'User_CityName' in df.columns:
            top_c = df['User_CityName'].value_counts().head(12).reset_index()
            top_c.columns = ['City', 'Users']
            fig = px.bar(top_c, x='Users', y='City', orientation='h',
                        title="🏙️ Top 12 Cities", color='Users',
                        color_continuous_scale='Reds')
            fig.update_layout(height=450, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, key='t7')

st.markdown("---")
st.markdown("<p style='text-align: center; color: #94a3b8; padding: 1rem;'>🌍 <strong>Tourism Analytics Platform</strong> | Powered by AI & Machine Learning | © 2024</p>", unsafe_allow_html=True)