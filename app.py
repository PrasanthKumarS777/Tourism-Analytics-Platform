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
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Tourism Analytics", page_icon="🌍", layout="wide")

st.markdown("""<style>
.main {background-color: #f5f7fa;}
h1 {color: #1e3a8a; font-weight: 700; border-bottom: 3px solid #3b82f6;}
.stMetric {background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    files = ["City", "User", "Transaction", "Item", "Type"]
    data = {name: pd.read_excel(f"Dataset/{name}.xlsx") for name in files}
    df = data["Transaction"].merge(data["User"], on="UserId") \
        .merge(data["Item"], on="AttractionId") \
        .merge(data["Type"], on="AttractionTypeId") \
        .merge(data["City"].add_prefix("User_"), left_on="CityId", right_on="User_CityId").dropna()
    return df

@st.cache_resource
def train_models(_df):
    # Regression
    X_r = _df[['VisitYear', 'VisitMonth', 'AttractionTypeId', 'ContinentId', 'CountryId']]
    y_r = _df['Rating']
    X_train, X_test, y_train, y_test = train_test_split(X_r, y_r, test_size=0.25, random_state=42)
    reg = RandomForestRegressor(random_state=42, n_estimators=100)
    reg.fit(X_train, y_train)
    r2 = r2_score(y_test, reg.predict(X_test))
    mae = mean_absolute_error(y_test, reg.predict(X_test))
    
    # Classification
    X_c = _df[['VisitYear', 'VisitMonth', 'ContinentId', 'CountryId']]
    le = LabelEncoder()
    y_c = le.fit_transform(_df['VisitMode'])
    X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.25, random_state=42)
    clf = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    
    return reg, clf, le, r2, mae, acc

@st.cache_data
def get_recs(_df, user_id):
    pivot = _df.pivot_table(index='UserId', columns='Attraction', values='Rating').fillna(0)
    if user_id not in pivot.index:
        return _df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(10).reset_index()
    sim = pd.DataFrame(cosine_similarity(pivot), index=pivot.index, columns=pivot.index)
    users = sim[user_id].sort_values(ascending=False)[1:4].index
    return _df[_df['UserId'].isin(users)].groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(10).reset_index()

st.title("🌍 Tourism Analytics Platform")
st.markdown("**ML-Powered Tourism Insights**")
st.markdown("---")

df = load_data()
reg, clf, le, r2, mae, acc = train_models(df)

with st.sidebar:
    st.title("⚙️ Settings")
    year = st.selectbox("Year", sorted(df['VisitYear'].unique()))
    month = st.selectbox("Month", sorted(df['VisitMonth'].unique()))
    atype = st.selectbox("Attraction Type", sorted(df['AttractionTypeId'].unique()))
    cont = st.selectbox("Continent", sorted(df['ContinentId'].unique()))
    ctry = st.selectbox("Country", sorted(df['CountryId'].unique()))
    uid = st.selectbox("User ID", sorted(df['UserId'].unique()))

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Predictions", "🌟 Recommendations", "📈 Analytics"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Users", f"{df['UserId'].nunique():,}")
    c3.metric("Attractions", f"{df['Attraction'].nunique():,}")
    c4.metric("Avg Rating", f"{df['Rating'].mean():.2f}")
    
    col1, col2 = st.columns(2)
    with col1:
        yearly = df.groupby('VisitYear').size().reset_index(name='Visits')
        st.plotly_chart(px.line(yearly, x='VisitYear', y='Visits', title="Trends"), key='t1')
    with col2:
        cont_d = df['ContinentId'].value_counts().reset_index()
        st.plotly_chart(px.pie(cont_d, values='count', names='ContinentId', title="Continents"), key='t2')

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⭐ Rating Prediction")
        pred_r = reg.predict([[year, month, atype, cont, ctry]])[0]
        st.markdown(f"""<div style="text-align:center; padding:30px; background:linear-gradient(135deg,#667eea,#764ba2); border-radius:15px;">
        <h1 style="color:white; font-size:72px; margin:0;">{pred_r:.2f}</h1><p style="color:white;">out of 5.0</p></div>""", unsafe_allow_html=True)
        st.metric("R² Score", f"{r2:.3f}")
        st.metric("MAE", f"{mae:.3f}")
    
    with col2:
        st.subheader("🧭 Visit Mode")
        pred_m = le.inverse_transform(clf.predict([[year, month, cont, ctry]]))[0]
        st.markdown(f"""<div style="text-align:center; padding:30px; background:linear-gradient(135deg,#f093fb,#f5576c); border-radius:15px;">
        <h1 style="color:white; font-size:48px; margin:0;">{pred_m}</h1></div>""", unsafe_allow_html=True)
        st.metric("Accuracy", f"{acc:.1%}")

with tab3:
    st.subheader(f"🌟 Recommendations for User #{uid}")
    recs = get_recs(df, uid)
    recs.columns = ['Attraction', 'Avg Rating']
    st.dataframe(recs, hide_index=True)
    st.plotly_chart(px.bar(recs.head(8), x='Avg Rating', y='Attraction', orientation='h', color='Avg Rating'), key='t3')
    st.download_button("📥 Download", recs.to_csv(index=False), f"recs_{uid}.csv")

with tab4:
    st.subheader("📈 Analytics")
    c1, c2 = st.columns(2)
    with c1:
        top_a = df['Attraction'].value_counts().head(10).reset_index()
        st.plotly_chart(px.bar(top_a, x='count', y='Attraction', orientation='h', title='Top Attractions'), key='t4')
    with c2:
        if 'User_CityName' in df.columns:
            top_c = df['User_CityName'].value_counts().head(10).reset_index()
            st.plotly_chart(px.bar(top_c, x='count', y='User_CityName', orientation='h', title='Top Cities'), key='t5')

st.markdown("---")
st.markdown("<div style='text-align:center; color:#64748b;'><p><strong>Tourism Analytics v2.0</strong> | ML Powered</p></div>", unsafe_allow_html=True)