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
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Tourism Analytics", page_icon="🌍", layout="wide")

# Clean, professional CSS
st.markdown("""<style>
    .main {background-color: #f8fafc;}
    h1 {color: #1e293b; font-weight: 700; padding-bottom: 1rem; border-bottom: 4px solid #3b82f6;}
    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    .stTabs [data-baseweb="tab"] {background: #e2e8f0; border-radius: 8px; padding: 8px 20px; font-weight: 600;}
    .stTabs [aria-selected="true"] {background: #3b82f6; color: white;}
    div[data-testid="stMetricValue"] {font-size: 28px; font-weight: 700; color: #1e293b;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        data = {name: pd.read_excel(f"Dataset/{name}.xlsx") for name in ["City", "User", "Transaction", "Item", "Type"]}
        df = data["Transaction"].merge(data["User"], on="UserId", how="left") \
            .merge(data["Item"], on="AttractionId", how="left") \
            .merge(data["Type"], on="AttractionTypeId", how="left") \
            .merge(data["City"].add_prefix("User_"), left_on="CityId", right_on="User_CityId", how="left").dropna()
        return df
    except:
        return pd.DataFrame()

@st.cache_resource
def train_models(_df):
    if _df.empty: return None, None, None, 0, 0, 0
    try:
        X_r = _df[['VisitYear', 'VisitMonth', 'AttractionTypeId', 'ContinentId', 'CountryId']]
        y_r = _df['Rating']
        X_train, X_test, y_train, y_test = train_test_split(X_r, y_r, test_size=0.25, random_state=42)
        reg = RandomForestRegressor(random_state=42, n_estimators=50, max_depth=10, n_jobs=-1)
        reg.fit(X_train, y_train)
        r2 = r2_score(y_test, reg.predict(X_test))
        mae = mean_absolute_error(y_test, reg.predict(X_test))
        
        X_c = _df[['VisitYear', 'VisitMonth', 'ContinentId', 'CountryId']]
        le = LabelEncoder()
        y_c = le.fit_transform(_df['VisitMode'])
        X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.25, random_state=42)
        clf = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0, n_estimators=50, max_depth=5)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        return reg, clf, le, r2, mae, acc
    except:
        return None, None, None, 0, 0, 0

@st.cache_data
def get_recs(_df, user_id):
    if _df.empty or 'Attraction' not in _df.columns:
        return pd.DataFrame({'Attraction': ['No Data'], 'Rating': [0]})
    try:
        pivot = _df.pivot_table(index='UserId', columns='Attraction', values='Rating').fillna(0)
        if user_id not in pivot.index:
            return _df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(10).reset_index()
        sim = pd.DataFrame(cosine_similarity(pivot), index=pivot.index, columns=pivot.index)
        users = sim[user_id].sort_values(ascending=False)[1:4].index
        return _df[_df['UserId'].isin(users)].groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(10).reset_index()
    except:
        return pd.DataFrame({'Attraction': ['Error'], 'Rating': [0]})

# Header
st.title("🌍 Tourism Experience Analytics Platform")
st.markdown("**Advanced Machine Learning for Tourism Insights & Recommendations**")
st.markdown("---")

df = load_data()
if df.empty:
    st.error("❌ Data loading failed")
    st.stop()

reg, clf, le, r2, mae, acc = train_models(df)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    year = st.selectbox("📅 Visit Year", sorted(df['VisitYear'].unique()))
    month = st.selectbox("📆 Month", sorted(df['VisitMonth'].unique()))
    atype = st.selectbox("🎭 Attraction Type", sorted(df['AttractionTypeId'].unique()))
    cont = st.selectbox("🌍 Continent", sorted(df['ContinentId'].unique()))
    ctry = st.selectbox("🗺️ Country", sorted(df['CountryId'].unique()))
    uid = st.selectbox("👤 User ID", sorted(df['UserId'].unique())[:50])

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Predictions", "⭐ Recommendations", "📈 Analytics"])

with tab1:
    st.subheader("📊 Executive Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Records", f"{len(df):,}")
    c2.metric("👥 Unique Users", f"{df['UserId'].nunique():,}")
    c3.metric("🏛️ Attractions", f"{df['Attraction'].nunique():,}")
    c4.metric("⭐ Avg Rating", f"{df['Rating'].mean():.2f}/5")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        yearly = df.groupby('VisitYear').size().reset_index(name='Visits')
        fig = px.line(yearly, x='VisitYear', y='Visits', title="📈 Visit Trends", markers=True)
        fig.update_traces(line_color='#3b82f6', line_width=3)
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, key='c1')
    
    with col2:
        cont_d = df['ContinentId'].value_counts().head(6).reset_index()
        cont_d.columns = ['Continent', 'Count']
        fig = px.pie(cont_d, values='Count', names='Continent', title="🌍 Geographic Distribution")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, key='c2')

with tab2:
    st.subheader("🎯 ML-Powered Predictions")
    if reg and clf and le:
        col1, col2 = st.columns(2)
        with col1:
            pred_r = reg.predict([[year, month, atype, cont, ctry]])[0]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 2.5rem; border-radius: 15px; text-align: center;">
                <h3 style="color: white; margin: 0;">⭐ Predicted Rating</h3>
                <h1 style="color: white; font-size: 4rem; margin: 1rem 0;">{pred_r:.2f}</h1>
                <p style="color: white; opacity: 0.9;">out of 5.0</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            m1, m2 = st.columns(2)
            m1.metric("R² Score", f"{r2:.3f}")
            m2.metric("MAE", f"{mae:.3f}")
        
        with col2:
            pred_m = le.inverse_transform(clf.predict([[year, month, cont, ctry]]))[0]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb, #f5576c); padding: 2.5rem; border-radius: 15px; text-align: center;">
                <h3 style="color: white; margin: 0;">🧭 Recommended Mode</h3>
                <h1 style="color: white; font-size: 3rem; margin: 1rem 0;">{pred_m}</h1>
                <p style="color: white; opacity: 0.9;">Best for this experience</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.metric("Model Accuracy", f"{acc:.1%}")

with tab3:
    st.subheader(f"⭐ Recommendations for User #{uid}")
    recs = get_recs(df, uid)
    recs.columns = ['Attraction', 'Rating']
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(recs, hide_index=True, height=350)
    with col2:
        if not recs.empty and recs.iloc[0]['Rating'] > 0:
            fig = px.bar(recs.head(6), x='Rating', y='Attraction', orientation='h', 
                        color='Rating', color_continuous_scale='RdYlGn')
            fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, key='c3')

with tab4:
    st.subheader("📈 Advanced Analytics")
    c1, c2 = st.columns(2)
    with c1:
        top_a = df['Attraction'].value_counts().head(10).reset_index()
        top_a.columns = ['Attraction', 'Visits']
        fig = px.bar(top_a, x='Visits', y='Attraction', orientation='h', title="🏆 Top 10 Attractions",
                    color='Visits', color_continuous_scale='Blues')
        fig.update_layout(height=400, margin=dict(l=0,r=0,t=40,b=0), yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, key='c4')
    
    with c2:
        if 'User_CityName' in df.columns:
            top_c = df['User_CityName'].value_counts().head(10).reset_index()
            top_c.columns = ['City', 'Users']
            fig = px.bar(top_c, x='Users', y='City', orientation='h', title="🏙️ Top 10 Cities",
                        color='Users', color_continuous_scale='Reds')
            fig.update_layout(height=400, margin=dict(l=0,r=0,t=40,b=0), yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, key='c5')

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>Tourism Analytics Platform © 2024 | Powered by Machine Learning</p>", unsafe_allow_html=True)