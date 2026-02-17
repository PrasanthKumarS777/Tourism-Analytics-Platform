import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Tourism Intelligence Hub",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Professional Styling
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styling */
    h1 {
        background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-weight: 800;
        letter-spacing: 2px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255,255,255,0.9);
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(120deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Metric Container */
    div[data-testid="stMetric"] {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2b5876 0%, #4e4376 100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* DataFrames */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    
    /* Info Boxes */
    .info-box {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 15px 0;
        border-left: 6px solid #667eea;
    }
    
    /* Prediction Cards */
    .prediction-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
    }
    
    /* Download Button */
    .download-btn {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Data Loading with Progress
@st.cache_data(show_spinner=False)
def load_data():
    """Load and merge all tourism datasets with error handling"""
    try:
        with st.spinner("🔄 Loading tourism data..."):
            datasets = {}
            data_files = ["City", "User", "Transaction", "Item", "Type"]
            
            for name in data_files:
                file_path = f"Dataset/{name}.xlsx"
                if os.path.exists(file_path):
                    datasets[name] = pd.read_excel(file_path)
                else:
                    st.warning(f"⚠️ {name}.xlsx not found")
                    return pd.DataFrame()
            
            # Merge all datasets
            df = datasets["Transaction"].merge(
                datasets["User"], on="UserId", how="left"
            ).merge(
                datasets["Item"], on="AttractionId", how="left"
            ).merge(
                datasets["Type"], on="AttractionTypeId", how="left"
            ).merge(
                datasets["City"].add_prefix("User_"), 
                left_on="CityId", 
                right_on="User_CityId", 
                how="left"
            ).dropna()
            
            return df
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def train_models(_df):
    """Train ML models with enhanced metrics"""
    if _df.empty:
        return None, None, None, 0, 0, 0, None, None
    
    try:
        with st.spinner("🤖 Training ML models..."):
            # Rating Prediction Model
            X_r = _df[['VisitYear', 'VisitMonth', 'AttractionTypeId', 'ContinentId', 'CountryId']]
            y_r = _df['Rating']
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                X_r, y_r, test_size=0.25, random_state=42
            )
            
            reg = RandomForestRegressor(
                random_state=42, 
                n_estimators=100, 
                max_depth=15, 
                min_samples_split=5,
                n_jobs=-1
            )
            reg.fit(X_train_r, y_train_r)
            
            y_pred_r = reg.predict(X_test_r)
            r2 = r2_score(y_test_r, y_pred_r)
            mae = mean_absolute_error(y_test_r, y_pred_r)
            
            # Visit Mode Classification Model
            X_c = _df[['VisitYear', 'VisitMonth', 'ContinentId', 'CountryId']]
            le = LabelEncoder()
            y_c = le.fit_transform(_df['VisitMode'])
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                X_c, y_c, test_size=0.25, random_state=42
            )
            
            clf = XGBClassifier(
                eval_metric='logloss',
                random_state=42,
                verbosity=0,
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1
            )
            clf.fit(X_train_c, y_train_c)
            
            y_pred_c = clf.predict(X_test_c)
            acc = accuracy_score(y_test_c, y_pred_c)
            
            # Feature importance
            feature_importance_r = pd.DataFrame({
                'feature': X_r.columns,
                'importance': reg.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance_c = pd.DataFrame({
                'feature': X_c.columns,
                'importance': clf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return reg, clf, le, r2, mae, acc, feature_importance_r, feature_importance_c
            
    except Exception as e:
        st.error(f"❌ Error training models: {str(e)}")
        return None, None, None, 0, 0, 0, None, None

@st.cache_data(show_spinner=False)
def get_recommendations(_df, user_id, top_n=10):
    """Get personalized recommendations using collaborative filtering"""
    if _df.empty or 'Attraction' not in _df.columns:
        return pd.DataFrame({'Attraction': ['No Data'], 'Rating': [0], 'Confidence': [0]})
    
    try:
        # Create user-item matrix
        pivot = _df.pivot_table(
            index='UserId', 
            columns='Attraction', 
            values='Rating'
        ).fillna(0)
        
        if user_id not in pivot.index:
            # Return popular items for new users
            popular = _df.groupby('Attraction').agg({
                'Rating': ['mean', 'count']
            }).reset_index()
            popular.columns = ['Attraction', 'Rating', 'Visits']
            popular['Confidence'] = (popular['Rating'] * np.log1p(popular['Visits'])) / 10
            return popular.nlargest(top_n, 'Confidence')[['Attraction', 'Rating', 'Confidence']]
        
        # Calculate user similarity
        sim_matrix = cosine_similarity(pivot)
        sim_df = pd.DataFrame(sim_matrix, index=pivot.index, columns=pivot.index)
        
        # Find similar users
        similar_users = sim_df[user_id].sort_values(ascending=False)[1:11].index
        
        # Get recommendations
        recommendations = _df[_df['UserId'].isin(similar_users)].groupby('Attraction').agg({
            'Rating': ['mean', 'count']
        }).reset_index()
        recommendations.columns = ['Attraction', 'Rating', 'Count']
        recommendations['Confidence'] = recommendations['Rating'] * np.sqrt(recommendations['Count']) / 5
        
        # Remove already visited attractions
        user_visited = _df[_df['UserId'] == user_id]['Attraction'].unique()
        recommendations = recommendations[~recommendations['Attraction'].isin(user_visited)]
        
        return recommendations.nlargest(top_n, 'Confidence')[['Attraction', 'Rating', 'Confidence']]
        
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {str(e)}")
        return pd.DataFrame({'Attraction': ['Error'], 'Rating': [0], 'Confidence': [0]})

def create_kpi_card(title, value, delta=None, icon="📊"):
    """Create a beautiful KPI card"""
    delta_html = ""
    if delta:
        color = "green" if delta > 0 else "red"
        arrow = "↑" if delta > 0 else "↓"
        delta_html = f'<p style="color: {color}; font-size: 14px; margin: 5px 0 0 0;">{arrow} {abs(delta):.1f}%</p>'
    
    return f"""
    <div class="info-box">
        <h4 style="margin: 0; color: #64748b; font-size: 14px;">{icon} {title}</h4>
        <h2 style="margin: 10px 0 0 0; color: #1e293b; font-size: 36px; font-weight: 800;">{value}</h2>
        {delta_html}
    </div>
    """

def export_to_csv(dataframe, filename):
    """Export dataframe to CSV"""
    return dataframe.to_csv(index=False).encode('utf-8')

# ==================== MAIN APP ====================

# Header
st.markdown("""
    <h1>🌍 TOURISM INTELLIGENCE HUB</h1>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.9); border-radius: 10px; margin-bottom: 20px;">
        <p style="font-size: 18px; color: #64748b; margin: 0;">
            <strong>Advanced Analytics & ML-Powered Insights for Tourism Industry</strong>
        </p>
        <p style="font-size: 14px; color: #94a3b8; margin: 5px 0 0 0;">
            Real-time predictions • Personalized recommendations • Business intelligence
        </p>
    </div>
""", unsafe_allow_html=True)

# Load Data
df = load_data()

if df.empty:
    st.error("❌ **Data Loading Failed** - Please ensure Dataset folder contains all required Excel files")
    st.stop()

# Train Models
reg, clf, le, r2, mae, acc, feat_imp_r, feat_imp_c = train_models(df)

# Sidebar - Enhanced Filters
with st.sidebar:
    st.markdown("## ⚙️ CONTROL PANEL")
    st.markdown("---")
    
    # Date Range Filter
    st.markdown("### 📅 TIME PERIOD")
    year_range = st.slider(
        "Select Year Range",
        int(df['VisitYear'].min()),
        int(df['VisitYear'].max()),
        (int(df['VisitYear'].min()), int(df['VisitYear'].max()))
    )
    
    # Filter data based on year range
    df_filtered = df[(df['VisitYear'] >= year_range[0]) & (df['VisitYear'] <= year_range[1])]
    
    st.markdown("### 🎯 PREDICTION INPUTS")
    year = st.selectbox("📅 Visit Year", sorted(df['VisitYear'].unique()), index=len(df['VisitYear'].unique())-1)
    month = st.selectbox("📆 Month", sorted(df['VisitMonth'].unique()))
    atype = st.selectbox("🎭 Attraction Type", sorted(df['AttractionTypeId'].unique()))
    cont = st.selectbox("🌍 Continent", sorted(df['ContinentId'].unique()))
    ctry = st.selectbox("🗺️ Country", sorted(df['CountryId'].unique()))
    
    st.markdown("### 👤 USER SELECTION")
    uid = st.selectbox("User ID", sorted(df['UserId'].unique())[:100])
    
    st.markdown("---")
    st.markdown("### 📊 MODEL PERFORMANCE")
    st.metric("Rating Model R²", f"{r2:.3f}")
    st.metric("Mode Model Accuracy", f"{acc:.1%}")
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px;">
            <p style="margin: 0; font-size: 12px;">© 2024 Tourism Intelligence</p>
            <p style="margin: 5px 0 0 0; font-size: 11px; opacity: 0.7;">Powered by ML & AI</p>
        </div>
    """, unsafe_allow_html=True)

# Main Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Dashboard",
    "🎯 ML Predictions",
    "⭐ Smart Recommendations",
    "📈 Deep Analytics",
    "🔍 Trend Analysis",
    "📋 Data Explorer"
])

# ==================== TAB 1: EXECUTIVE DASHBOARD ====================
with tab1:
    st.markdown("## 📊 EXECUTIVE DASHBOARD")
    st.markdown("Real-time KPIs and business metrics")
    
    # Top KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(create_kpi_card(
            "Total Visits",
            f"{len(df_filtered):,}",
            icon="📋"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_kpi_card(
            "Unique Users",
            f"{df_filtered['UserId'].nunique():,}",
            icon="👥"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_kpi_card(
            "Attractions",
            f"{df_filtered['Attraction'].nunique():,}",
            icon="🏛️"
        ), unsafe_allow_html=True)
    
    with col4:
        avg_rating = df_filtered['Rating'].mean()
        st.markdown(create_kpi_card(
            "Avg Rating",
            f"{avg_rating:.2f}/5.0",
            icon="⭐"
        ), unsafe_allow_html=True)
    
    with col5:
        st.markdown(create_kpi_card(
            "Continents",
            f"{df_filtered['ContinentId'].nunique()}",
            icon="🌍"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Visit Trends Over Time
        yearly = df_filtered.groupby('VisitYear').size().reset_index(name='Visits')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly['VisitYear'],
            y=yearly['Visits'],
            mode='lines+markers',
            line=dict(color='#667eea', width=4),
            marker=dict(size=10, color='#764ba2'),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)'
        ))
        fig.update_layout(
            title="📈 Visit Trends Over Time",
            xaxis_title="Year",
            yaxis_title="Number of Visits",
            height=350,
            hovermode='x unified',
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Geographic Distribution
        cont_data = df_filtered['ContinentId'].value_counts().reset_index()
        cont_data.columns = ['Continent', 'Count']
        fig = px.pie(
            cont_data,
            values='Count',
            names='Continent',
            title="🌍 Geographic Distribution",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            height=350,
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly Distribution
        monthly = df_filtered.groupby('VisitMonth').size().reset_index(name='Visits')
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly['Month'] = monthly['VisitMonth'].apply(lambda x: month_names[int(x)-1] if x <= 12 else str(x))
        
        fig = px.bar(
            monthly,
            x='Month',
            y='Visits',
            title="📆 Seasonal Patterns",
            color='Visits',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            height=350,
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rating Distribution
        fig = px.histogram(
            df_filtered,
            x='Rating',
            title="⭐ Rating Distribution",
            nbins=20,
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            height=350,
            xaxis_title="Rating",
            yaxis_title="Frequency",
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 2: ML PREDICTIONS ====================
with tab2:
    st.markdown("## 🎯 MACHINE LEARNING PREDICTIONS")
    st.markdown("AI-powered predictions for ratings and travel modes")
    
    if reg and clf and le:
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating Prediction
            pred_rating = reg.predict([[year, month, atype, cont, ctry]])[0]
            pred_rating = max(1, min(5, pred_rating))  # Clamp between 1-5
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 3rem; border-radius: 20px; text-align: center;
                        box-shadow: 0 15px 35px rgba(0,0,0,0.3);">
                <h3 style="color: white; margin: 0; font-size: 20px;">⭐ PREDICTED RATING</h3>
                <h1 style="color: white; font-size: 5rem; margin: 1.5rem 0; font-weight: 900;">{pred_rating:.2f}</h1>
                <p style="color: white; opacity: 0.95; font-size: 18px; margin: 0;">out of 5.0 stars</p>
                <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 2px solid rgba(255,255,255,0.3);">
                    <p style="color: white; margin: 0; font-size: 14px;">Confidence: High</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Model Metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🎯 R² Score", f"{r2:.4f}", help="Coefficient of determination")
            with col_b:
                st.metric("📊 MAE", f"{mae:.4f}", help="Mean Absolute Error")
            
            # Feature Importance
            if feat_imp_r is not None and not feat_imp_r.empty:
                st.markdown("#### 📈 Feature Importance (Rating)")
                fig = px.bar(
                    feat_imp_r,
                    x='importance',
                    y='feature',
                    orientation='h',
                    color='importance',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    yaxis={'categoryorder':'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Visit Mode Prediction
            pred_mode = le.inverse_transform(clf.predict([[year, month, cont, ctry]]))[0]
            mode_proba = clf.predict_proba([[year, month, cont, ctry]])[0]
            confidence = max(mode_proba) * 100
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 3rem; border-radius: 20px; text-align: center;
                        box-shadow: 0 15px 35px rgba(0,0,0,0.3);">
                <h3 style="color: white; margin: 0; font-size: 20px;">🧭 RECOMMENDED MODE</h3>
                <h1 style="color: white; font-size: 3.5rem; margin: 1.5rem 0; font-weight: 900;">{pred_mode}</h1>
                <p style="color: white; opacity: 0.95; font-size: 18px; margin: 0;">Best travel mode for this experience</p>
                <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 2px solid rgba(255,255,255,0.3);">
                    <p style="color: white; margin: 0; font-size: 14px;">Confidence: {confidence:.1f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Model Accuracy
            st.metric("🎯 Model Accuracy", f"{acc:.2%}", help="Classification accuracy on test set")
            
            # Mode Probabilities
            st.markdown("#### 📊 Mode Probabilities")
            mode_df = pd.DataFrame({
                'Mode': le.classes_,
                'Probability': mode_proba * 100
            }).sort_values('Probability', ascending=False)
            
            fig = px.bar(
                mode_df,
                x='Probability',
                y='Mode',
                orientation='h',
                color='Probability',
                color_continuous_scale='Reds',
                text='Probability'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                height=300,
                showlegend=False,
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Models not available. Please check data quality.")

# ==================== TAB 3: SMART RECOMMENDATIONS ====================
with tab3:
    st.markdown(f"## ⭐ PERSONALIZED RECOMMENDATIONS FOR USER #{uid}")
    st.markdown("AI-powered attraction recommendations based on collaborative filtering")
    
    # Get recommendations
    recs = get_recommendations(df, uid, top_n=15)
    
    if not recs.empty and recs.iloc[0]['Rating'] > 0:
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            user_visits = len(df[df['UserId'] == uid])
            st.metric("🎫 User Visits", f"{user_visits}")
        with col2:
            user_avg_rating = df[df['UserId'] == uid]['Rating'].mean()
            st.metric("⭐ Avg Rating Given", f"{user_avg_rating:.2f}")
        with col3:
            st.metric("🎯 Recommendations", len(recs))
        
        st.markdown("---")
        
        # Display recommendations
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### 📋 Top Recommendations")
            
            # Style the dataframe
            styled_recs = recs.head(10).copy()
            styled_recs['Rating'] = styled_recs['Rating'].round(2)
            styled_recs['Confidence'] = styled_recs['Confidence'].round(3)
            styled_recs.index = range(1, len(styled_recs) + 1)
            
            st.dataframe(
                styled_recs,
                height=400,
                use_container_width=True
            )
            
            # Download button
            csv = export_to_csv(styled_recs, 'recommendations.csv')
            st.download_button(
                label="📥 Download Recommendations",
                data=csv,
                file_name=f'recommendations_user_{uid}.csv',
                mime='text/csv',
            )
        
        with col2:
            st.markdown("### 📊 Recommendation Scores")
            
            # Bar chart
            top_6 = recs.head(6)
            fig = px.bar(
                top_6,
                x='Confidence',
                y='Attraction',
                orientation='h',
                color='Rating',
                color_continuous_scale='RdYlGn',
                text='Rating'
            )
            fig.update_traces(texttemplate='%{text:.2f}⭐', textposition='outside')
            fig.update_layout(
                height=400,
                yaxis={'categoryorder':'total ascending'},
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # User's Visit History
        st.markdown("---")
        st.markdown("### 📜 User's Visit History")
        
        user_history = df[df['UserId'] == uid][['Attraction', 'Rating', 'VisitYear', 'VisitMonth']].sort_values(
            ['VisitYear', 'VisitMonth'], ascending=False
        ).head(10)
        
        if not user_history.empty:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(user_history, hide_index=True, use_container_width=True)
            with col2:
                # Rating trend
                fig = px.line(
                    user_history.iloc[::-1],
                    y='Rating',
                    title="Rating Trend",
                    markers=True
                )
                fig.update_layout(height=250, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No recommendations available for this user.")

# ==================== TAB 4: DEEP ANALYTICS ====================
with tab4:
    st.markdown("## 📈 DEEP ANALYTICS & INSIGHTS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top Attractions
        st.markdown("### 🏆 Top 15 Attractions")
        top_attr = df_filtered['Attraction'].value_counts().head(15).reset_index()
        top_attr.columns = ['Attraction', 'Visits']
        
        fig = px.bar(
            top_attr,
            x='Visits',
            y='Attraction',
            orientation='h',
            color='Visits',
            color_continuous_scale='Blues',
            text='Visits'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=500,
            yaxis={'categoryorder':'total ascending'},
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top User Cities
        st.markdown("### 🏙️ Top 15 User Cities")
        if 'User_CityName' in df_filtered.columns:
            top_cities = df_filtered['User_CityName'].value_counts().head(15).reset_index()
            top_cities.columns = ['City', 'Users']
            
            fig = px.bar(
                top_cities,
                x='Users',
                y='City',
                orientation='h',
                color='Users',
                color_continuous_scale='Reds',
                text='Users'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=500,
                yaxis={'categoryorder':'total ascending'},
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Visit Mode Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚗 Visit Mode Distribution")
        mode_dist = df_filtered['VisitMode'].value_counts().reset_index()
        mode_dist.columns = ['Mode', 'Count']
        
        fig = px.pie(
            mode_dist,
            values='Count',
            names='Mode',
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎭 Attraction Type Performance")
        type_perf = df_filtered.groupby('AttractionTypeId').agg({
            'Rating': 'mean',
            'UserId': 'count'
        }).reset_index()
        type_perf.columns = ['Type', 'Avg_Rating', 'Count']
        
        fig = px.scatter(
            type_perf,
            x='Count',
            y='Avg_Rating',
            size='Count',
            color='Avg_Rating',
            color_continuous_scale='RdYlGn',
            hover_data=['Type'],
            text='Type'
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 5: TREND ANALYSIS ====================
with tab5:
    st.markdown("## 🔍 TREND ANALYSIS")
    
    # Year-over-Year Growth
    st.markdown("### 📈 Year-over-Year Growth Analysis")
    yearly_stats = df.groupby('VisitYear').agg({
        'UserId': 'count',
        'Rating': 'mean'
    }).reset_index()
    yearly_stats.columns = ['Year', 'Visits', 'Avg_Rating']
    yearly_stats['Growth_%'] = yearly_stats['Visits'].pct_change() * 100
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Visit Volume', 'Average Rating'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=yearly_stats['Year'], y=yearly_stats['Visits'], 
               name='Visits', marker_color='#667eea'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Avg_Rating'],
                   mode='lines+markers', name='Avg Rating', 
                   line=dict(color='#f5576c', width=3)),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display growth table
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            yearly_stats.style.format({
                'Visits': '{:,.0f}',
                'Avg_Rating': '{:.2f}',
                'Growth_%': '{:.2f}%'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        total_growth = ((yearly_stats['Visits'].iloc[-1] / yearly_stats['Visits'].iloc[0]) - 1) * 100
        st.metric(
            "Total Growth",
            f"{total_growth:.1f}%",
            delta=f"{yearly_stats['Growth_%'].iloc[-1]:.1f}% YoY"
        )
    
    st.markdown("---")
    
    # Monthly Patterns
    st.markdown("### 📆 Monthly Patterns & Seasonality")
    monthly_pattern = df.groupby(['VisitYear', 'VisitMonth']).size().reset_index(name='Visits')
    
    fig = px.line(
        monthly_pattern,
        x='VisitMonth',
        y='Visits',
        color='VisitYear',
        markers=True,
        title="Monthly Visit Patterns by Year"
    )
    fig.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    heatmap_data = df.pivot_table(
        index='VisitMonth',
        columns='VisitYear',
        values='UserId',
        aggfunc='count'
    ).fillna(0)
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Year", y="Month", color="Visits"),
        aspect="auto",
        color_continuous_scale='YlOrRd'
    )
    fig.update_layout(height=400, title="Visit Heatmap: Month vs Year")
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 6: DATA EXPLORER ====================
with tab6:
    st.markdown("## 📋 DATA EXPLORER")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_continent = st.multiselect(
            "Filter by Continent",
            options=sorted(df['ContinentId'].unique()),
            default=[]
        )
    with col2:
        filter_mode = st.multiselect(
            "Filter by Visit Mode",
            options=sorted(df['VisitMode'].unique()),
            default=[]
        )
    with col3:
        rating_range = st.slider(
            "Filter by Rating",
            float(df['Rating'].min()),
            float(df['Rating'].max()),
            (float(df['Rating'].min()), float(df['Rating'].max()))
        )
    
    # Apply filters
    df_explorer = df_filtered.copy()
    if filter_continent:
        df_explorer = df_explorer[df_explorer['ContinentId'].isin(filter_continent)]
    if filter_mode:
        df_explorer = df_explorer[df_explorer['VisitMode'].isin(filter_mode)]
    df_explorer = df_explorer[
        (df_explorer['Rating'] >= rating_range[0]) & 
        (df_explorer['Rating'] <= rating_range[1])
    ]
    
    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Filtered Records", f"{len(df_explorer):,}")
    col2.metric("Avg Rating", f"{df_explorer['Rating'].mean():.2f}")
    col3.metric("Unique Attractions", f"{df_explorer['Attraction'].nunique()}")
    col4.metric("Unique Users", f"{df_explorer['UserId'].nunique()}")
    
    # Data Table
    st.markdown("### 📊 Filtered Data")
    display_cols = ['Attraction', 'Rating', 'VisitYear', 'VisitMonth', 'VisitMode', 
                   'ContinentId', 'CountryId']
    
    st.dataframe(
        df_explorer[display_cols].head(100),
        hide_index=True,
        use_container_width=True,
        height=400
    )
    
    # Download options
    col1, col2 = st.columns(2)
    with col1:
        csv_all = export_to_csv(df_explorer[display_cols], 'filtered_data.csv')
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_all,
            file_name='tourism_filtered_data.csv',
            mime='text/csv',
        )
    
    with col2:
        # Summary stats
        if st.button("📊 Generate Summary Report"):
            summary = df_explorer.describe()
            csv_summary = export_to_csv(summary, 'summary.csv')
            st.download_button(
                label="📥 Download Summary Statistics",
                data=csv_summary,
                file_name='summary_statistics.csv',
                mime='text/csv',
            )

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        <h3 style="color: white; margin: 0;">🌍 Tourism Intelligence Hub</h3>
        <p style="color: white; opacity: 0.9; margin: 10px 0 0 0;">
            © 2024 | Powered by Advanced Machine Learning & AI | Data-Driven Tourism Insights
        </p>
    </div>
""", unsafe_allow_html=True)