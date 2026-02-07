import streamlit as st
import pandas as pd
import numpy as np
import os

# Page setup
st.set_page_config(
    page_title="Tourism Analytics Platform", 
    page_icon="🌍", 
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .main {padding: 20px;}
    h1 {color: #1e3a8a; border-bottom: 2px solid #3b82f6; padding-bottom: 10px;}
    .stMetric {background: white; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0;}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🌍 Tourism Analytics Platform")
st.markdown("**ML-Powered Tourism Insights & Recommendations**")
st.markdown("---")

# Function to check YOUR actual files
def check_data_files():
    DATA_PATH = "Dataset"
    
    # YOUR ACTUAL FILE NAMES
    your_files = [
        "City.xlsx",
        "Continent.xlsx", 
        "Country.xlsx",
        "Item.xlsx",
        "Mode.xlsx",
        "Region.xlsx",
        "Transaction.xlsx",
        "Type.xlsx",
        "User.xlsx"
    ]
    
    st.subheader("📁 Your Data Files")
    
    # Show in 3 columns
    cols = st.columns(3)
    found_count = 0
    
    for i, file in enumerate(your_files):
        col_idx = i % 3
        path = os.path.join(DATA_PATH, file)
        
        with cols[col_idx]:
            if os.path.exists(path):
                file_size = os.path.getsize(path) / 1024  # KB
                st.success(f"✅ {file}")
                st.caption(f"Size: {file_size:.1f} KB")
                found_count += 1
            else:
                st.error(f"❌ {file}")
                st.caption("File missing")
    
    return found_count, len(your_files)

# Function to load and merge YOUR data
def load_your_data():
    try:
        DATA_PATH = "Dataset"
        
        # Load your files
        city = pd.read_excel(os.path.join(DATA_PATH, "City.xlsx"))
        user = pd.read_excel(os.path.join(DATA_PATH, "User.xlsx"))
        transaction = pd.read_excel(os.path.join(DATA_PATH, "Transaction.xlsx"))
        item = pd.read_excel(os.path.join(DATA_PATH, "Item.xlsx"))
        type_df = pd.read_excel(os.path.join(DATA_PATH, "Type.xlsx"))
        continent = pd.read_excel(os.path.join(DATA_PATH, "Continent.xlsx"))
        country = pd.read_excel(os.path.join(DATA_PATH, "Country.xlsx"))
        mode = pd.read_excel(os.path.join(DATA_PATH, "Mode.xlsx"))
        region = pd.read_excel(os.path.join(DATA_PATH, "Region.xlsx"))
        
        # Merge based on YOUR data structure
        # First merge Transaction with User
        df = transaction.merge(user, on="UserId", how="left")
        
        # Merge with Item
        df = df.merge(item, on="AttractionId", how="left")
        
        # Merge with Type
        df = df.merge(type_df, on="AttractionTypeId", how="left")
        
        # Merge with City
        df = df.merge(city.add_prefix("User_"), left_on="CityId", right_on="User_CityId", how="left")
        
        # Drop any missing values
        df = df.dropna()
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)[:200]}")
        return pd.DataFrame()

# Main app
def main():
    # Check files
    found, total = check_data_files()
    
    st.markdown("---")
    
    # Status dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Files Found", f"{found}/{total}")
    
    with col2:
        if found == total:
            st.metric("Status", "READY", "✓")
        else:
            st.metric("Status", "INCOMPLETE", "↓")
    
    with col3:
        st.metric("System", "ONLINE", "✓")
    
    # Load and preview data
    if found == total:
        st.markdown("---")
        st.subheader("📊 Data Preview")
        
        with st.spinner("Loading your data..."):
            df = load_your_data()
        
        if not df.empty:
            st.success(f"✅ Successfully loaded {len(df):,} records")
            
            # Show first few rows
            st.write("**First 5 rows:**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Show columns
            st.write(f"**Columns ({len(df.columns)} total):**")
            cols_per_row = 5
            for i in range(0, len(df.columns), cols_per_row):
                cols = df.columns[i:i+cols_per_row]
                st.code(" | ".join(cols))
            
            # Basic stats
            st.write("**Basic Statistics:**")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("Total Records", f"{len(df):,}")
            
            with stat_col2:
                st.metric("Unique Users", f"{df['UserId'].nunique():,}")
            
            with stat_col3:
                if 'Rating' in df.columns:
                    st.metric("Avg Rating", f"{df['Rating'].mean():.2f}")
                else:
                    st.metric("Columns", len(df.columns))
            
            with stat_col4:
                if 'Attraction' in df.columns:
                    st.metric("Attractions", f"{df['Attraction'].nunique():,}")
                else:
                    st.metric("File Size", f"{os.path.getsize('Dataset/Transaction.xlsx')/1024/1024:.1f} MB")
        else:
            st.warning("Data loaded but empty or merge failed")
    else:
        st.warning(f"⚠️ Missing {total - found} files. All 9 files must be present.")
    
    # Test ML Libraries
    st.markdown("---")
    st.subheader("🧪 System Check")
    
    test_col1, test_col2 = st.columns(2)
    
    with test_col1:
        if st.button("Test ML Libraries", type="primary"):
            try:
                from sklearn.ensemble import RandomForestRegressor
                from xgboost import XGBClassifier
                import plotly.express as px
                
                st.success("✅ All ML libraries work!")
                st.write("Ready for: Rating Prediction & Recommendations")
                
                # Quick model test
                X = np.random.rand(100, 5)
                y = np.random.rand(100)
                model = RandomForestRegressor(n_estimators=10)
                model.fit(X, y)
                st.success("✅ Can train ML models")
                
            except Exception as e:
                st.error(f"❌ ML test failed: {e}")
    
    with test_col2:
        if st.button("Show System Info"):
            st.write(f"Python: {pd.__version__}")
            st.write(f"Pandas: {pd.__version__}")
            st.write(f"Numpy: {np.__version__}")
            st.write(f"Streamlit: {st.__version__}")
    
    # Next Steps
    st.markdown("---")
    st.subheader("🚀 Ready for Full Features")
    
    st.info("""
    **Once this basic version works, we'll add:**
    1. Rating Prediction (ML Model)
    2. Visit Mode Prediction  
    3. Attraction Recommendations
    4. Interactive Analytics Dashboards
    5. Download Reports
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p><strong>Tourism Analytics Platform</strong> | Customized for Your Dataset</p>
        <small>9 Excel files detected | All systems checking...</small>
    </div>
    """, unsafe_allow_html=True)

# Run app
if __name__ == "__main__":
    main()