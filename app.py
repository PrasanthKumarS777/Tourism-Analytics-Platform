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

# Page setup
st.set_page_config(
    page_title="Tourism Analytics Platform", 
    page_icon="🌍", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f5f7fa; padding: 20px;}
    h1 {color: #1e3a8a; font-weight: 700; border-bottom: 3px solid #3b82f6; padding-bottom: 15px;}
    h2 {color: #334155; font-weight: 600; margin-top: 25px;}
    .stMetric {background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;}
    .stButton button {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: 600; border-radius: 8px; padding: 10px 24px;}
    div[data-testid="stSidebar"] {background: linear-gradient(180deg, #1e3a8a 0%, #1e40af 100%);}
    div[data-testid="stSidebar"] * {color: white !important;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = "Dataset"

# ========== DATA LOADING & PREPROCESSING (from your File.py) ==========
@st.cache_data
def load_and_process_data():
    """Load and merge all 9 Excel files - Based on your File.py"""
    try:
        # Load all Excel files
        files = {
            "City": "City.xlsx",
            "Continent": "Continent.xlsx", 
            "Country": "Country.xlsx",
            "Item": "Item.xlsx",
            "Mode": "Mode.xlsx",
            "Region": "Region.xlsx",
            "Transaction": "Transaction.xlsx",
            "Type": "Type.xlsx",
            "User": "User.xlsx"
        }
        
        data = {}
        for name, file in files.items():
            filepath = os.path.join(DATA_PATH, file)
            data[name] = pd.read_excel(filepath)
        
        # Extract individual dataframes
        city = data["City"]
        user = data["User"]
        trans = data["Transaction"]
        item = data["Item"]
        typ = data["Type"]
        
        # Merge step by step (as in your File.py)
        merged = trans.merge(user, on="UserId", how="left") \
                      .merge(item, on="AttractionId", how="left") \
                      .merge(typ, on="AttractionTypeId", how="left") \
                      .merge(city.add_prefix("User_"), left_on="CityId", right_on="User_CityId", how="left")
        
        # Drop missing values
        df = merged.dropna()
        
        # Ensure numeric columns
        numeric_cols = ['VisitYear', 'VisitMonth', 'Rating', 'AttractionTypeId', 'ContinentId', 'CountryId', 'CityId']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)[:200]}")
        return pd.DataFrame()

# ========== ML MODELS (from your File.py) ==========
@st.cache_resource
def train_regression_model(_df):
    """Train Rating Prediction Model - Based on your File.py Step 3"""
    if _df.empty:
        return None, 0, 0, 0, []
    
    try:
        # Check required columns
        required_cols = ['VisitYear', 'VisitMonth', 'AttractionTypeId', 'ContinentId', 'CountryId', 'Rating']
        missing_cols = [col for col in required_cols if col not in _df.columns]
        
        if missing_cols:
            st.warning(f"Missing columns for regression: {missing_cols}")
            return None, 0, 0, 0, []
        
        # Prepare features and target
        X = _df[['VisitYear', 'VisitMonth', 'AttractionTypeId', 'ContinentId', 'CountryId']]
        y = _df['Rating']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Train model
        model = RandomForestRegressor(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        return model, r2, mae, X_test.shape[0], X.columns.tolist()
        
    except Exception as e:
        st.error(f"Error training regression model: {str(e)[:200]}")
        return None, 0, 0, 0, []

@st.cache_resource
def train_classification_model(_df):
    """Train Visit Mode Prediction Model - Based on your File.py Step 4"""
    if _df.empty or 'VisitMode' not in _df.columns:
        return None, None, 0, []
    
    try:
        # Check required columns
        required_cols = ['VisitYear', 'VisitMonth', 'ContinentId', 'CountryId', 'VisitMode']
        missing_cols = [col for col in required_cols if col not in _df.columns]
        
        if missing_cols:
            st.warning(f"Missing columns for classification: {missing_cols}")
            return None, None, 0, []
        
        # Prepare features and target
        X = _df[['VisitYear', 'VisitMonth', 'ContinentId', 'CountryId']]
        y = _df['VisitMode']
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)
        
        # Train model
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        return model, le, acc, X.columns.tolist()
        
    except Exception as e:
        st.error(f"Error training classification model: {str(e)[:200]}")
        return None, None, 0, []

@st.cache_data
def get_recommendations(_df, user_id, top_n=10):
    """Get attraction recommendations - Based on your File.py Step 5"""
    if _df.empty or 'Attraction' not in _df.columns:
        return pd.DataFrame({'Attraction': ['Sample Attraction 1', 'Sample Attraction 2'], 
                           'Avg Rating': [4.5, 4.2]})
    
    try:
        # Create user-item matrix
        pivot = _df.pivot_table(index='UserId', columns='Attraction', values='Rating').fillna(0)
        
        if user_id not in pivot.index:
            # Return top attractions overall
            top_attractions = _df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(top_n)
            recs = top_attractions.reset_index()
            recs.columns = ['Attraction', 'Avg Rating']
            return recs.round(2)
        
        # Calculate similarity
        sim_matrix = cosine_similarity(pivot)
        sim_df = pd.DataFrame(sim_matrix, index=pivot.index, columns=pivot.index)
        
        # Find similar users
        similar_users = sim_df[user_id].sort_values(ascending=False)[1:4].index
        
        # Get recommendations from similar users
        similar_ratings = _df[_df['UserId'].isin(similar_users)]
        recommendations = similar_ratings.groupby('Attraction')['Rating'].agg(['mean', 'count']).sort_values(by='mean', ascending=False).head(top_n)
        
        recommendations = recommendations.reset_index()
        recommendations.columns = ['Attraction', 'Avg Rating', 'Rating Count']
        
        return recommendations[['Attraction', 'Avg Rating']].round(2)
        
    except Exception as e:
        # Return popular attractions as fallback
        popular = _df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(top_n)
        return popular.reset_index().round(2)

# ========== MAIN APP ==========
def main():
    st.title("🌍 Tourism Experience Analytics Platform")
    st.markdown("**Advanced ML-Powered Tourism Insights & Recommendations**")
    st.markdown("---")
    
    # Load data
    with st.spinner("📊 Loading and processing data..."):
        df = load_and_process_data()
    
    if df.empty:
        st.error("❌ Failed to load data. Please check your Excel files in the Dataset folder.")
        return
    
    # Train ML models
    with st.spinner("🤖 Training Machine Learning models..."):
        reg_model, r2_score, mae_score, test_size, reg_features = train_regression_model(df)
        class_model, label_encoder, acc_score, class_features = train_classification_model(df)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")
        
        # Get unique values safely
        visit_years = sorted(df['VisitYear'].unique()) if 'VisitYear' in df.columns else [2023, 2024]
        visit_months = sorted(df['VisitMonth'].unique()) if 'VisitMonth' in df.columns else list(range(1, 13))
        attraction_types = sorted(df['AttractionTypeId'].unique()) if 'AttractionTypeId' in df.columns else [1, 2, 3]
        continents = sorted(df['ContinentId'].unique()) if 'ContinentId' in df.columns else [1, 2, 3]
        countries = sorted(df['CountryId'].unique()) if 'CountryId' in df.columns else [101, 102, 103]
        user_ids = sorted(df['UserId'].unique()) if 'UserId' in df.columns else df['UserId'].iloc[:5].tolist() if len(df) > 0 else [1, 2, 3]
        
        st.subheader("📊 Prediction Inputs")
        visit_year = st.selectbox("Visit Year", visit_years, index=0)
        visit_month = st.selectbox("Visit Month", visit_months, index=0)
        attraction_type = st.selectbox("Attraction Type ID", attraction_types, index=0)
        continent = st.selectbox("Continent ID", continents, index=0)
        country = st.selectbox("Country ID", countries, index=0)
        
        st.subheader("🌟 Recommendations")
        user_id = st.selectbox("User ID", user_ids, index=0)
        
        st.markdown("---")
        st.info(f"**Data Summary:**")
        st.write(f"• {len(df):,} total records")
        st.write(f"• {df['UserId'].nunique():,} unique users")
        st.write(f"• {df['Attraction'].nunique():,} attractions")
        if 'Rating' in df.columns:
            st.write(f"• Avg rating: {df['Rating'].mean():.2f}/5")
        if 'VisitMode' in df.columns:
            st.write(f"• Visit modes: {df['VisitMode'].nunique()}")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "⭐ Rating Predictor", "🧭 Visit Mode", "🌟 Recommendations", "📈 Analytics"])
    
    # TAB 1: DASHBOARD
    with tab1:
        st.subheader("Executive Dashboard")
        
        # KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        with col2:
            st.metric("Unique Users", f"{df['UserId'].nunique():,}")
        with col3:
            st.metric("Total Attractions", f"{df['Attraction'].nunique():,}")
        with col4:
            avg_rating = df['Rating'].mean() if 'Rating' in df.columns else 0
            st.metric("Average Rating", f"{avg_rating:.2f}")
        
        # Charts Row 1
        c1, c2 = st.columns(2)
        with c1:
            # Visit trends by year
            if 'VisitYear' in df.columns:
                yearly_visits = df.groupby('VisitYear').size().reset_index(name='Visits')
                fig = px.line(yearly_visits, x='VisitYear', y='Visits', 
                             title="Visit Trends Over Years", markers=True,
                             line_shape="spline")
                fig.update_traces(line=dict(color="#3b82f6", width=3))
                st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            # Geographic distribution
            if 'ContinentId' in df.columns:
                continent_dist = df['ContinentId'].value_counts().head(10).reset_index()
                continent_dist.columns = ['Continent', 'Count']
                fig = px.pie(continent_dist, values='Count', names='Continent',
                            title="Top 10 Continents by Visits", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
        
        # Charts Row 2
        c3, c4 = st.columns(2)
        with c3:
            # Monthly patterns
            if 'VisitMonth' in df.columns:
                monthly = df.groupby('VisitMonth').size().reset_index(name='Visits')
                fig = px.bar(monthly, x='VisitMonth', y='Visits',
                            title="Monthly Visit Patterns",
                            color='Visits', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        
        with c4:
            # Rating distribution
            if 'Rating' in df.columns:
                fig = px.histogram(df, x='Rating', nbins=20,
                                  title="Rating Distribution",
                                  color_discrete_sequence=['#10b981'])
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: RATING PREDICTION
    with tab2:
        st.subheader("⭐ Rating Prediction")
        st.write("Predict the expected rating for a tourism experience")
        
        if reg_model is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Prediction input display
                input_data = pd.DataFrame({
                    'Feature': reg_features,
                    'Value': [visit_year, visit_month, attraction_type, continent, country]
                })
                st.write("**Input Parameters:**")
                st.dataframe(input_data, use_container_width=True, hide_index=True)
                
                # Make prediction
                prediction_input = np.array([[visit_year, visit_month, attraction_type, continent, country]])
                predicted_rating = reg_model.predict(prediction_input)[0]
                
                # Display prediction with style
                st.markdown("---")
                st.markdown(f"""
                <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;">
                    <h2 style="color: white; margin: 0;">Predicted Rating</h2>
                    <h1 style="color: white; font-size: 72px; margin: 10px 0;">{predicted_rating:.2f}</h1>
                    <p style="color: white; font-size: 18px;">out of 5.0</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.write("**Model Performance:**")
                st.metric("R² Score", f"{r2_score:.3f}")
                st.metric("Mean Absolute Error", f"{mae_score:.3f}")
                st.metric("Test Samples", f"{test_size:,}")
                
                st.markdown("---")
                st.write("**Interpretation:**")
                if predicted_rating >= 4.0:
                    st.success("✅ **Excellent Experience** - Highly recommended!")
                elif predicted_rating >= 3.0:
                    st.info("📊 **Good Experience** - Worth visiting")
                else:
                    st.warning("⚠️ **Average Experience** - Consider alternatives")
        else:
            st.warning("Rating prediction model not available. Check if all required features exist in your data.")
    
    # TAB 3: VISIT MODE PREDICTION
    with tab3:
        st.subheader("🧭 Visit Mode Prediction")
        st.write("Predict the preferred visit mode (Family, Solo, Couple, etc.)")
        
        if class_model is not None and label_encoder is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                input_data = pd.DataFrame({
                    'Feature': class_features,
                    'Value': [visit_year, visit_month, continent, country]
                })
                st.write("**Input Parameters:**")
                st.dataframe(input_data, use_container_width=True, hide_index=True)
                
                # Make prediction
                prediction_input = np.array([[visit_year, visit_month, continent, country]])
                predicted_mode_encoded = class_model.predict(prediction_input)[0]
                predicted_mode = label_encoder.inverse_transform([predicted_mode_encoded])[0]
                
                # Display prediction
                st.markdown("---")
                st.markdown(f"""
                <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 15px;">
                    <h2 style="color: white; margin: 0;">Recommended Visit Mode</h2>
                    <h1 style="color: white; font-size: 48px; margin: 10px 0;">{predicted_mode}</h1>
                    <p style="color: white; font-size: 16px;">Based on similar user preferences</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.write("**Model Performance:**")
                st.metric("Accuracy", f"{acc_score:.1%}")
                st.metric("Unique Modes", f"{len(label_encoder.classes_)}")
                
                st.markdown("---")
                st.write("**Available Modes:**")
                for mode in label_encoder.classes_:
                    st.write(f"• {mode}")
        else:
            st.warning("Visit mode prediction not available. Ensure 'VisitMode' column exists in your data.")
    
    # TAB 4: RECOMMENDATIONS
    with tab4:
        st.subheader("🌟 Personalized Recommendations")
        st.write(f"Top attractions recommended for User #{user_id}")
        
        with st.spinner("Finding the best attractions for you..."):
            recommendations = get_recommendations(df, user_id, top_n=10)
        
        if not recommendations.empty:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.write("**Recommended Attractions:**")
                st.dataframe(recommendations, use_container_width=True, hide_index=True)
                
                # Download button
                csv = recommendations.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Recommendations",
                    csv,
                    f"recommendations_user_{user_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            
            with col2:
                # Visualization
                fig = px.bar(recommendations.head(8), 
                            x='Avg Rating', y='Attraction',
                            orientation='h',
                            title=f'Top Recommendations for User {user_id}',
                            color='Avg Rating',
                            color_continuous_scale='RdYlGn')
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("**Recommendation Logic:**")
                st.write("""
                • Based on users with similar preferences
                • Considers attraction ratings from similar users
                • Prioritizes highly-rated attractions
                • Uses collaborative filtering (cosine similarity)
                """)
        else:
            st.info("No specific recommendations available. Showing popular attractions:")
            popular = df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(10)
            st.dataframe(popular.reset_index(), use_container_width=True)
    
    # TAB 5: ANALYTICS
    with tab5:
        st.subheader("📈 Advanced Analytics")
        
        # Row 1: Top attractions and cities
        a1, a2 = st.columns(2)
        with a1:
            if 'Attraction' in df.columns:
                top_attractions = df['Attraction'].value_counts().head(10).reset_index()
                top_attractions.columns = ['Attraction', 'Visits']
                fig = px.bar(top_attractions, x='Visits', y='Attraction',
                            orientation='h', title='Top 10 Most Visited Attractions',
                            color='Visits', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
        
        with a2:
            if 'User_CityName' in df.columns:
                top_cities = df['User_CityName'].value_counts().head(10).reset_index()
                top_cities.columns = ['City', 'Visits']
                fig = px.bar(top_cities, x='Visits', y='City',
                            orientation='h', title='Top 10 Cities by User Base',
                            color='Visits', color_continuous_scale='Plasma')
                st.plotly_chart(fig, use_container_width=True)
        
        # Row 2: Visit mode analysis and rating trends
        a3, a4 = st.columns(2)
        with a3:
            if 'VisitMode' in df.columns:
                mode_dist = df['VisitMode'].value_counts().reset_index()
                mode_dist.columns = ['Visit Mode', 'Count']
                fig = px.pie(mode_dist, values='Count', names='Visit Mode',
                            title='Visit Mode Distribution', hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
        
        with a4:
            if 'Rating' in df.columns and 'AttractionType' in df.columns:
                rating_by_type = df.groupby('AttractionType')['Rating'].mean().sort_values(ascending=False).head(10).reset_index()
                fig = px.bar(rating_by_type, x='Rating', y='AttractionType',
                            orientation='h', title='Top 10 Attraction Types by Rating',
                            color='Rating', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
        
        # Data export
        st.markdown("---")
        st.subheader("📥 Export Data")
        
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        
        with exp_col1:
            if st.button("Export Sample Data (100 rows)"):
                sample_csv = df.head(100).to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Sample CSV",
                    sample_csv,
                    f"tourism_sample_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    key='sample_csv'
                )
        
        with exp_col2:
            if st.button("Export Statistics"):
                stats = pd.DataFrame({
                    'Metric': ['Total Records', 'Unique Users', 'Unique Attractions', 'Avg Rating'],
                    'Value': [len(df), df['UserId'].nunique(), df['Attraction'].nunique(), 
                             df['Rating'].mean() if 'Rating' in df.columns else 0]
                })
                stats_csv = stats.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Stats CSV",
                    stats_csv,
                    f"tourism_stats_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    key='stats_csv'
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p><strong>Tourism Analytics Platform v2.0</strong> | Powered by Machine Learning</p>
        <small>Models: Random Forest Regression (Rating) | XGBoost Classification (Visit Mode) | Collaborative Filtering (Recommendations)</small>
    </div>
    """, unsafe_allow_html=True)

# Run app
if __name__ == "__main__":
    main()