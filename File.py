"""
Tourism Experience Analytics: Classification, Prediction, and Recommendation System
==================================================================================

This script performs the complete data pipeline:
1. Data Loading from Excel files
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA) with Visualizations
4. Feature Engineering
5. Model Training (Regression, Classification, Recommendation)
6. Model Evaluation and Saving

Author: Tourism Analytics Team
Date: February 2026
"""

# ============================================================================
# IMPORT REQUIRED LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, r2_score, mean_absolute_error
)

# Recommendation System
from sklearn.metrics.pairwise import cosine_similarity

# Model Persistence
import pickle
import os

# Set pandas display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Set matplotlib style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("TOURISM EXPERIENCE ANALYTICS - COMPLETE DATA PIPELINE")
print("="*80)


# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================

def load_data():
    """
    Load all datasets from Excel files in the Dataset folder
    Uses your exact file names: City.xlsx, Continent.xlsx, etc.
    
    Returns:
        dict: Dictionary containing all dataframes
    """
    print("\n[STEP 1] Loading Data from Dataset folder...")
    print("-" * 80)
    
    try:
        # Define file paths - matching YOUR exact file names
        data_files = {
            'transactions': 'Dataset/Transaction.xlsx',
            'users': 'Dataset/User.xlsx',
            'cities': 'Dataset/City.xlsx',
            'attractions': 'Dataset/Item.xlsx',
            'attraction_types': 'Dataset/Type.xlsx',
            'visit_modes': 'Dataset/Mode.xlsx',
            'continents': 'Dataset/Continent.xlsx',
            'countries': 'Dataset/Country.xlsx',
            'regions': 'Dataset/Region.xlsx'
        }
        
        data = {}
        for key, filename in data_files.items():
            try:
                data[key] = pd.read_excel(filename)
                print(f"✓ Loaded {key}: {data[key].shape}")
            except FileNotFoundError:
                print(f"✗ Warning: {filename} not found!")
                data[key] = None
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
                data[key] = None
        
        successful_loads = sum(1 for v in data.values() if v is not None)
        print(f"\n✓ Successfully loaded {successful_loads}/9 datasets")
        
        if successful_loads == 0:
            print("\n" + "="*80)
            print("ERROR: No datasets loaded!")
            print("="*80)
            return None
        
        return data
    
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None


# ============================================================================
# STEP 2: DATA CLEANING
# ============================================================================

def clean_data(data):
    """
    Clean and preprocess all datasets
    - Handle missing values
    - Remove duplicates
    - Handle outliers
    - Standardize data formats
    
    Args:
        data (dict): Dictionary containing all dataframes
    
    Returns:
        dict: Cleaned dataframes
    """
    print("\n[STEP 2] Data Cleaning and Preprocessing...")
    print("-" * 80)
    
    cleaned_data = {}
    
    # -----------------------------------------------------------------------
    # 2.1 Clean Transaction Data
    # -----------------------------------------------------------------------
    if data['transactions'] is not None:
        df_trans = data['transactions'].copy()
        
        print("\n2.1 Cleaning Transaction Data:")
        print(f"   Initial shape: {df_trans.shape}")
        print(f"   Missing values:")
        print(f"{df_trans.isnull().sum()}")
        
        # Handle missing values in Rating (fill with median)
        if 'Rating' in df_trans.columns:
            median_rating = df_trans['Rating'].median()
            missing_ratings = df_trans['Rating'].isnull().sum()
            df_trans['Rating'].fillna(median_rating, inplace=True)
            print(f"   ✓ Filled {missing_ratings} missing ratings with median: {median_rating:.2f}")
        
        # Handle missing values in VisitMode (fill with mode)
        if 'VisitMode' in df_trans.columns:
            mode_visit = df_trans['VisitMode'].mode()[0] if len(df_trans['VisitMode'].mode()) > 0 else 'Unknown'
            missing_modes = df_trans['VisitMode'].isnull().sum()
            df_trans['VisitMode'].fillna(mode_visit, inplace=True)
            print(f"   ✓ Filled {missing_modes} missing visit modes with: {mode_visit}")
        
        # Remove duplicate transactions
        initial_rows = len(df_trans)
        df_trans.drop_duplicates(inplace=True)
        print(f"   ✓ Removed {initial_rows - len(df_trans)} duplicate rows")
        
        # Handle outliers in Rating (valid range: 1-5)
        if 'Rating' in df_trans.columns:
            before_outlier = len(df_trans)
            df_trans = df_trans[(df_trans['Rating'] >= 1) & (df_trans['Rating'] <= 5)]
            print(f"   ✓ Removed {before_outlier - len(df_trans)} invalid ratings (outside 1-5 range)")
        
        # Validate year and month ranges
        if 'VisitYear' in df_trans.columns:
            df_trans = df_trans[(df_trans['VisitYear'] >= 2000) & (df_trans['VisitYear'] <= 2026)]
        if 'VisitMonth' in df_trans.columns:
            df_trans = df_trans[(df_trans['VisitMonth'] >= 1) & (df_trans['VisitMonth'] <= 12)]
        
        print(f"   Final shape: {df_trans.shape}")
        cleaned_data['transactions'] = df_trans
    
    # -----------------------------------------------------------------------
    # 2.2 Clean User Data
    # -----------------------------------------------------------------------
    if data['users'] is not None:
        df_users = data['users'].copy()
        print("\n2.2 Cleaning User Data:")
        print(f"   Initial shape: {df_users.shape}")
        
        # Remove duplicate users
        df_users.drop_duplicates(subset=['UserId'], keep='first', inplace=True)
        
        # Fill missing location IDs with mode (most common value)
        for col in ['ContinentId', 'RegionId', 'CountryId', 'CityId']:
            if col in df_users.columns and df_users[col].isnull().sum() > 0:
                mode_val = df_users[col].mode()[0] if len(df_users[col].mode()) > 0 else 0
                missing_count = df_users[col].isnull().sum()
                df_users[col].fillna(mode_val, inplace=True)
                print(f"   ✓ Filled {missing_count} missing {col} with mode: {mode_val}")
        
        print(f"   Final shape: {df_users.shape}")
        cleaned_data['users'] = df_users
    
    # -----------------------------------------------------------------------
    # 2.3 Clean City Data
    # -----------------------------------------------------------------------
    if data['cities'] is not None:
        df_cities = data['cities'].copy()
        print("\n2.3 Cleaning City Data:")
        print(f"   Initial shape: {df_cities.shape}")
        
        # Standardize city names (strip whitespace, title case)
        if 'CityName' in df_cities.columns:
            df_cities['CityName'] = df_cities['CityName'].astype(str).str.strip().str.title()
            print(f"   ✓ Standardized city names to title case")
        
        # Remove duplicates
        df_cities.drop_duplicates(subset=['CityId'], keep='first', inplace=True)
        
        print(f"   Final shape: {df_cities.shape}")
        cleaned_data['cities'] = df_cities
    
    # -----------------------------------------------------------------------
    # 2.4 Clean Attraction Data (Item Data)
    # -----------------------------------------------------------------------
    if data['attractions'] is not None:
        df_attractions = data['attractions'].copy()
        print("\n2.4 Cleaning Attraction Data:")
        print(f"   Initial shape: {df_attractions.shape}")
        
        # Standardize attraction names
        if 'Attraction' in df_attractions.columns:
            df_attractions['Attraction'] = df_attractions['Attraction'].astype(str).str.strip()
            print(f"   ✓ Standardized attraction names")
        
        # Remove duplicates
        df_attractions.drop_duplicates(subset=['AttractionId'], keep='first', inplace=True)
        
        # Handle missing addresses
        if 'AttractionAddress' in df_attractions.columns:
            missing_addr = df_attractions['AttractionAddress'].isnull().sum()
            df_attractions['AttractionAddress'].fillna('Unknown', inplace=True)
            print(f"   ✓ Filled {missing_addr} missing addresses with 'Unknown'")
        
        print(f"   Final shape: {df_attractions.shape}")
        cleaned_data['attractions'] = df_attractions
    
    # -----------------------------------------------------------------------
    # 2.5 Clean Reference Tables (Types, Modes, Locations)
    # -----------------------------------------------------------------------
    for key in ['attraction_types', 'visit_modes', 'continents', 'countries', 'regions']:
        if data[key] is not None:
            df = data[key].copy()
            initial_shape = df.shape
            df.drop_duplicates(inplace=True)
            cleaned_data[key] = df
            print(f"\n2.5 Cleaned {key}: {initial_shape} → {df.shape}")
    
    print("\n" + "="*80)
    print("✓ DATA CLEANING COMPLETE!")
    print("="*80)
    
    return cleaned_data


# ============================================================================
# STEP 3: FEATURE ENGINEERING & DATA INTEGRATION (FIXED VERSION)
# ============================================================================

def create_master_dataset(data):
    """
    Merge all datasets and create engineered features
    - Join transaction, user, attraction, and location data
    - Create aggregated features (user stats, attraction stats)
    - Create temporal features (season)
    
    Args:
        data (dict): Cleaned dataframes
    
    Returns:
        pd.DataFrame: Integrated master dataset
    """
    print("\n[STEP 3] Feature Engineering & Data Integration...")
    print("-" * 80)
    
    # Start with transaction data as the base
    df = data['transactions'].copy()
    print(f"\nBase dataset (transactions): {df.shape}")
    
    # -----------------------------------------------------------------------
    # 3.1 Merge User Information
    # -----------------------------------------------------------------------
    if data['users'] is not None:
        df = df.merge(data['users'], on='UserId', how='left', suffixes=('', '_user'))
        print(f"After merging users: {df.shape}")
    
    # -----------------------------------------------------------------------
    # 3.2 Merge Attraction Information
    # -----------------------------------------------------------------------
    if data['attractions'] is not None:
        df = df.merge(data['attractions'], on='AttractionId', how='left', suffixes=('', '_attraction'))
        print(f"After merging attractions: {df.shape}")
    
    # -----------------------------------------------------------------------
    # 3.3 Merge City Data for User Location
    # -----------------------------------------------------------------------
    if data['cities'] is not None:
        user_cities = data['cities'].copy()
        user_cities.columns = ['CityId', 'UserCityName', 'UserCityCountryId']
        # Convert CityId to same type before merge
        if 'CityId' in df.columns:
            df['CityId'] = df['CityId'].astype(str)
            user_cities['CityId'] = user_cities['CityId'].astype(str)
        df = df.merge(user_cities, on='CityId', how='left')
        print(f"After merging user cities: {df.shape}")
    
    # -----------------------------------------------------------------------
    # 3.4 Merge City Data for Attraction Location
    # -----------------------------------------------------------------------
    if data['cities'] is not None:
        attraction_cities = data['cities'].copy()
        attraction_cities.columns = ['AttractionCityId', 'AttractionCityName', 'AttractionCityCountryId']
        # Convert AttractionCityId to same type before merge
        if 'AttractionCityId' in df.columns:
            df['AttractionCityId'] = df['AttractionCityId'].astype(str)
            attraction_cities['AttractionCityId'] = attraction_cities['AttractionCityId'].astype(str)
        df = df.merge(attraction_cities, on='AttractionCityId', how='left')
        print(f"After merging attraction cities: {df.shape}")
    
    # -----------------------------------------------------------------------
    # 3.5 Merge Continent Data
    # -----------------------------------------------------------------------
    if data['continents'] is not None:
        continents = data['continents'].copy()
        continents.columns = ['ContinentId', 'ContinentName']
        # Convert ContinentId to same type before merge
        if 'ContinentId' in df.columns:
            df['ContinentId'] = df['ContinentId'].astype(str)
            continents['ContinentId'] = continents['ContinentId'].astype(str)
        df = df.merge(continents, on='ContinentId', how='left')
        print(f"After merging continents: {df.shape}")
    
    # -----------------------------------------------------------------------
    # 3.6 Merge Region Data (FIXED)
    # -----------------------------------------------------------------------
    if data['regions'] is not None:
        regions = data['regions'].copy()
        regions.columns = ['RegionId', 'RegionName', 'RegionContinentId']
        # Convert RegionId to same type before merge - THIS IS THE FIX
        if 'RegionId' in df.columns:
            df['RegionId'] = df['RegionId'].astype(str)
            regions['RegionId'] = regions['RegionId'].astype(str)
        df = df.merge(regions, on='RegionId', how='left')
        print(f"After merging regions: {df.shape}")
    
    # -----------------------------------------------------------------------
    # 3.7 Merge Country Data
    # -----------------------------------------------------------------------
    if data['countries'] is not None:
        countries = data['countries'].copy()
        countries.columns = ['CountryId', 'CountryName', 'CountryRegionId']
        # Convert CountryId to same type before merge
        if 'CountryId' in df.columns:
            df['CountryId'] = df['CountryId'].astype(str)
            countries['CountryId'] = countries['CountryId'].astype(str)
        df = df.merge(countries, on='CountryId', how='left')
        print(f"After merging countries: {df.shape}")
    
    # -----------------------------------------------------------------------
    # 3.8 Merge Attraction Type Data
    # -----------------------------------------------------------------------
    if data['attraction_types'] is not None:
        attraction_types = data['attraction_types'].copy()
        # Convert AttractionTypeId to same type before merge
        if 'AttractionTypeId' in df.columns:
            df['AttractionTypeId'] = df['AttractionTypeId'].astype(str)
            attraction_types['AttractionTypeId'] = attraction_types['AttractionTypeId'].astype(str)
        df = df.merge(attraction_types, on='AttractionTypeId', how='left')
        print(f"After merging attraction types: {df.shape}")
    
    # -----------------------------------------------------------------------
    # 3.9 CREATE ENGINEERED FEATURES
    # -----------------------------------------------------------------------
    print("\n3.9 Creating Engineered Features...")
    
    # Feature 1: Season from Month
    if 'VisitMonth' in df.columns:
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Fall'
        
        df['Season'] = df['VisitMonth'].apply(get_season)
        print("   ✓ Created 'Season' feature from VisitMonth")
    
    # Feature 2: User-Level Aggregated Features
    if 'UserId' in df.columns and 'Rating' in df.columns:
        # Calculate average rating and visit count per user
        user_stats = df.groupby('UserId')['Rating'].agg(['mean', 'count']).reset_index()
        user_stats.columns = ['UserId', 'UserAvgRating', 'UserVisitCount']
        df = df.merge(user_stats, on='UserId', how='left')
        print("   ✓ Created 'UserAvgRating' and 'UserVisitCount' features")
    
    # Feature 3: Attraction-Level Aggregated Features
    if 'AttractionId' in df.columns and 'Rating' in df.columns:
        # Calculate average rating and visit count per attraction
        attraction_stats = df.groupby('AttractionId')['Rating'].agg(['mean', 'count']).reset_index()
        attraction_stats.columns = ['AttractionId', 'AttractionAvgRating', 'AttractionVisitCount']
        df = df.merge(attraction_stats, on='AttractionId', how='left')
        print("   ✓ Created 'AttractionAvgRating' and 'AttractionVisitCount' features")
    
    # Feature 4: Visit Mode Popularity
    if 'VisitMode' in df.columns:
        visit_mode_counts = df.groupby('VisitMode').size().reset_index(name='VisitModePopularity')
        df = df.merge(visit_mode_counts, on='VisitMode', how='left')
        print("   ✓ Created 'VisitModePopularity' feature")
    
    # Feature 5: Attraction Type Popularity
    if 'AttractionType' in df.columns:
        type_counts = df.groupby('AttractionType').size().reset_index(name='AttractionTypePopularity')
        df = df.merge(type_counts, on='AttractionType', how='left')
        print("   ✓ Created 'AttractionTypePopularity' feature")
    
    print(f"\n✓ Master dataset created: {df.shape}")
    print(f"✓ Total features: {len(df.columns)}")
    
    return df


# ============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

def perform_eda(df):
    """
    Perform comprehensive exploratory data analysis
    - Generate descriptive statistics
    - Create visualizations for all key features
    - Analyze distributions, correlations, and patterns
    
    Args:
        df (pd.DataFrame): Master dataset
    """
    print("\n[STEP 4] Exploratory Data Analysis (EDA)...")
    print("-" * 80)
    
    # Create output directory for plots
    os.makedirs('eda_plots', exist_ok=True)
    
    # -----------------------------------------------------------------------
    # 4.1 Dataset Overview
    # -----------------------------------------------------------------------
    print("\n4.1 Dataset Overview:")
    print(f"   Total Records: {len(df):,}")
    print(f"   Total Features: {len(df.columns)}")
    print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # -----------------------------------------------------------------------
    # 4.2 Descriptive Statistics
    # -----------------------------------------------------------------------
    print("\n4.2 Descriptive Statistics (Numerical Features):")
    print(df.describe())
    
    # -----------------------------------------------------------------------
    # 4.3 Missing Values Check
    # -----------------------------------------------------------------------
    print("\n4.3 Missing Values Summary:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing[missing > 0],
            'Percentage': missing_pct[missing > 0]
        }).sort_values('Percentage', ascending=False)
        print(missing_df)
    else:
        print("   ✓ No missing values in the dataset!")
    
    # -----------------------------------------------------------------------
    # 4.4 Create Visualizations
    # -----------------------------------------------------------------------
    print("\n4.4 Creating Visualizations (12 plots)...")
    
    # PLOT 1: Rating Distribution
    if 'Rating' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Rating'], bins=20, kde=True, color='skyblue', edgecolor='black')
        plt.title('Distribution of Attraction Ratings', fontsize=16, fontweight='bold')
        plt.xlabel('Rating (1-5)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig('eda_plots/01_rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ [1/12] Rating distribution plot saved")
    
    # PLOT 2: Visit Mode Distribution
    if 'VisitMode' in df.columns:
        plt.figure(figsize=(10, 6))
        visit_mode_counts = df['VisitMode'].value_counts()
        colors = sns.color_palette('viridis', len(visit_mode_counts))
        bars = plt.bar(range(len(visit_mode_counts)), visit_mode_counts.values, 
                      color=colors, edgecolor='black')
        plt.title('Distribution of Visit Modes', fontsize=16, fontweight='bold')
        plt.xlabel('Visit Mode', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(range(len(visit_mode_counts)), visit_mode_counts.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('eda_plots/02_visit_mode_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ [2/12] Visit mode distribution plot saved")
    
    # PLOT 3: Top 15 Attractions by Average Rating
    if 'Attraction' in df.columns and 'Rating' in df.columns:
        plt.figure(figsize=(12, 8))
        top_attractions = df.groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(15)
        colors = sns.color_palette('coolwarm', len(top_attractions))
        bars = plt.barh(range(len(top_attractions)), top_attractions.values, 
                       color=colors, edgecolor='black')
        plt.yticks(range(len(top_attractions)), top_attractions.index, fontsize=10)
        plt.xlabel('Average Rating', fontsize=12)
        plt.title('Top 15 Attractions by Average Rating', fontsize=16, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('eda_plots/03_top_attractions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ [3/12] Top attractions plot saved")
    
    # PLOT 4: Continent-wise Distribution
    if 'ContinentName' in df.columns:
        plt.figure(figsize=(10, 6))
        continent_counts = df['ContinentName'].value_counts()
        colors = sns.color_palette('Set2', len(continent_counts))
        bars = plt.bar(range(len(continent_counts)), continent_counts.values, 
                      color=colors, edgecolor='black')
        plt.title('User Distribution by Continent', fontsize=16, fontweight='bold')
        plt.xlabel('Continent', fontsize=12)
        plt.ylabel('Number of Visits', fontsize=12)
        plt.xticks(range(len(continent_counts)), continent_counts.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('eda_plots/04_continent_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ [4/12] Continent distribution plot saved")
    
    # PLOT 5: Attraction Type Distribution
    if 'AttractionType' in df.columns:
        plt.figure(figsize=(12, 8))
        type_counts = df['AttractionType'].value_counts().head(10)
        colors = sns.color_palette('magma', len(type_counts))
        bars = plt.barh(range(len(type_counts)), type_counts.values, 
                       color=colors, edgecolor='black')
        plt.yticks(range(len(type_counts)), type_counts.index, fontsize=10)
        plt.xlabel('Count', fontsize=12)
        plt.title('Top 10 Attraction Types', fontsize=16, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('eda_plots/05_attraction_types.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ [5/12] Attraction types plot saved")
    
    # PLOT 6: Seasonal Ratings
    if 'Season' in df.columns and 'Rating' in df.columns:
        plt.figure(figsize=(10, 6))
        season_rating = df.groupby('Season')['Rating'].mean().reindex(['Winter', 'Spring', 'Summer', 'Fall'])
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        bars = plt.bar(range(len(season_rating)), season_rating.values, 
                      color=colors, edgecolor='black')
        plt.title('Average Rating by Season', fontsize=16, fontweight='bold')
        plt.xlabel('Season', fontsize=12)
        plt.ylabel('Average Rating', fontsize=12)
        plt.xticks(range(len(season_rating)), season_rating.index)
        plt.ylim([season_rating.min() - 0.2, season_rating.max() + 0.2])
        plt.grid(axis='y', alpha=0.3)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('eda_plots/06_seasonal_ratings.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ [6/12] Seasonal ratings plot saved")
    
    # PLOT 7: Visit Mode vs Rating Box Plot
    if 'VisitMode' in df.columns and 'Rating' in df.columns:
        plt.figure(figsize=(12, 6))
        visit_modes = df['VisitMode'].value_counts().head(6).index
        data_to_plot = [df[df['VisitMode'] == mode]['Rating'].values for mode in visit_modes]
        bp = plt.boxplot(data_to_plot, labels=visit_modes, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', edgecolor='black'),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'))
        plt.title('Rating Distribution by Visit Mode', fontsize=16, fontweight='bold')
        plt.xlabel('Visit Mode', fontsize=12)
        plt.ylabel('Rating', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('eda_plots/07_visitmode_rating_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ [7/12] Visit mode vs rating box plot saved")
    
    # PLOT 8: Yearly Trends
    if 'VisitYear' in df.columns:
        plt.figure(figsize=(12, 6))
        year_counts = df['VisitYear'].value_counts().sort_index()
        plt.plot(year_counts.index, year_counts.values, marker='o', linewidth=2, 
                markersize=8, color='#e74c3c')
        plt.fill_between(year_counts.index, year_counts.values, alpha=0.3, color='#e74c3c')
        plt.title('Number of Visits by Year', fontsize=16, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of Visits', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('eda_plots/08_yearly_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ [8/12] Yearly trends plot saved")
    
    # PLOT 9: Monthly Patterns
    if 'VisitMonth' in df.columns:
        plt.figure(figsize=(12, 6))
        month_counts = df['VisitMonth'].value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        colors = sns.color_palette('husl', 12)
        bars = plt.bar(range(1, 13), [month_counts.get(i, 0) for i in range(1, 13)], 
                      color=colors, edgecolor='black')
        plt.title('Number of Visits by Month', fontsize=16, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Number of Visits', fontsize=12)
        plt.xticks(range(1, 13), month_names)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('eda_plots/09_monthly_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ [9/12] Monthly patterns plot saved")
    
    # PLOT 10: Top Countries
    if 'CountryName' in df.columns:
        plt.figure(figsize=(12, 8))
        country_counts = df['CountryName'].value_counts().head(15)
        colors = sns.color_palette('rocket', len(country_counts))
        bars = plt.barh(range(len(country_counts)), country_counts.values, 
                       color=colors, edgecolor='black')
        plt.yticks(range(len(country_counts)), country_counts.index, fontsize=10)
        plt.xlabel('Number of Visits', fontsize=12)
        plt.title('Top 15 Countries by Visit Volume', fontsize=16, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('eda_plots/10_top_countries.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ [10/12] Top countries plot saved")
    
    # PLOT 11: Rating vs Visit Count Scatter
    if 'UserVisitCount' in df.columns and 'UserAvgRating' in df.columns:
        plt.figure(figsize=(10, 6))
        sample_size = min(1000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        plt.scatter(sample_df['UserVisitCount'], sample_df['UserAvgRating'], 
                   alpha=0.5, s=30, color='purple', edgecolor='black')
        plt.title('User Visit Count vs Average Rating', fontsize=16, fontweight='bold')
        plt.xlabel('Visit Count', fontsize=12)
        plt.ylabel('Average Rating', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('eda_plots/11_visitcount_vs_rating.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ [11/12] Visit count vs rating scatter plot saved")
    
    # PLOT 12: Correlation Heatmap
    numerical_cols = ['Rating', 'VisitYear', 'VisitMonth', 'UserAvgRating', 
                     'UserVisitCount', 'AttractionAvgRating', 'AttractionVisitCount']
    available_numerical = [col for col in numerical_cols if col in df.columns]
    
    if len(available_numerical) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[available_numerical].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Numerical Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('eda_plots/12_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ [12/12] Correlation heatmap saved")
    
    print("\n" + "="*80)
    print("✓ EDA COMPLETE! All visualizations saved in 'eda_plots/' folder")
    print("="*80)


# ============================================================================
# STEP 5: DATA PREPROCESSING FOR MACHINE LEARNING
# ============================================================================

def preprocess_for_ml(df):
    """
    Prepare data for machine learning models
    - Handle remaining missing values
    - Encode categorical variables
    - Create label encoders for future use
    
    Args:
        df (pd.DataFrame): Master dataset
    
    Returns:
        tuple: (Processed dataframe, Label encoders dictionary)
    """
    print("\n[STEP 5] Preprocessing for Machine Learning...")
    print("-" * 80)
    
    # Create a copy for processing
    df_ml = df.copy()
    
    # -----------------------------------------------------------------------
    # 5.1 Handle Missing Values
    # -----------------------------------------------------------------------
    print("\n5.1 Handling Missing Values...")
    
    # Fill numerical columns with median
    numerical_cols = df_ml.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_ml[col].isnull().sum() > 0:
            median_val = df_ml[col].median()
            missing_count = df_ml[col].isnull().sum()
            df_ml[col].fillna(median_val, inplace=True)
            print(f"   ✓ Filled {missing_count} missing values in {col} with median: {median_val:.2f}")
    
    # Fill categorical columns with mode
    categorical_cols = df_ml.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_ml[col].isnull().sum() > 0:
            mode_val = df_ml[col].mode()[0] if len(df_ml[col].mode()) > 0 else 'Unknown'
            missing_count = df_ml[col].isnull().sum()
            df_ml[col].fillna(mode_val, inplace=True)
            print(f"   ✓ Filled {missing_count} missing values in {col} with mode: {mode_val}")
    
    # -----------------------------------------------------------------------
    # 5.2 Label Encoding for Categorical Variables
    # -----------------------------------------------------------------------
    print("\n5.2 Encoding Categorical Variables...")
    
    label_encoders = {}
    categorical_features = ['VisitMode', 'ContinentName', 'RegionName', 'CountryName', 
                           'AttractionType', 'Season', 'UserCityName', 'AttractionCityName']
    
    for col in categorical_features:
        if col in df_ml.columns:
            le = LabelEncoder()
            # Convert to string to handle any data type
            df_ml[f'{col}_Encoded'] = le.fit_transform(df_ml[col].astype(str))
            label_encoders[col] = le
            print(f"   ✓ Encoded {col}: {len(le.classes_)} unique values")
    
    # -----------------------------------------------------------------------
    # 5.3 Save Label Encoders
    # -----------------------------------------------------------------------
    os.makedirs('models', exist_ok=True)
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("\n   ✓ Label encoders saved to 'models/label_encoders.pkl'")
    
    print("\n" + "="*80)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*80)
    
    return df_ml, label_encoders


# ============================================================================
# STEP 6: REGRESSION MODEL - RATING PREDICTION
# ============================================================================

def train_regression_model(df):
    """
    Train regression models to predict attraction ratings
    - Compare Linear Regression and Random Forest
    - Evaluate using R2, RMSE, MAE
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
    
    Returns:
        tuple: (Best model data, All results)
    """
    print("\n[STEP 6] Training Regression Model (Rating Prediction)...")
    print("-" * 80)
    
    # -----------------------------------------------------------------------
    # 6.1 Select Features for Regression
    # -----------------------------------------------------------------------
    feature_cols = [
        'VisitYear', 'VisitMonth', 'UserVisitCount', 'AttractionVisitCount',
        'VisitModePopularity', 'AttractionTypePopularity'
    ]
    
    # Add encoded features
    encoded_features = [col for col in df.columns if col.endswith('_Encoded') and 'VisitMode' not in col]
    feature_cols.extend(encoded_features)
    
    # Keep only available features
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"\nFeatures used for regression ({len(feature_cols)} features):")
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i}. {col}")
    
    # -----------------------------------------------------------------------
    # 6.2 Prepare X and y
    # -----------------------------------------------------------------------
    X = df[feature_cols].copy()
    y = df['Rating'].copy()
    
    # Handle any remaining missing values
    X.fillna(X.median(), inplace=True)
    
    # -----------------------------------------------------------------------
    # 6.3 Train-Test Split
    # -----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"   Training set: {X_train.shape}")
    print(f"   Testing set: {X_test.shape}")
    
    # -----------------------------------------------------------------------
    # 6.4 Train Multiple Models
    # -----------------------------------------------------------------------
    print("\n6.4 Training and Comparing Models...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, 
                                              random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n   Training {model_name}...")
        
        # Scale features for Linear Regression
        if model_name == 'Linear Regression':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            scaler = None
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[model_name] = {
            'model': model,
            'scaler': scaler,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'predictions': y_pred
        }
        
        print(f"   ✓ {model_name} Results:")
        print(f"      - R² Score: {r2:.4f}")
        print(f"      - RMSE: {rmse:.4f}")
        print(f"      - MAE: {mae:.4f}")
        print(f"      - MSE: {mse:.4f}")
    
    # -----------------------------------------------------------------------
    # 6.5 Select Best Model
    # -----------------------------------------------------------------------
    best_model_name = max(results, key=lambda x: results[x]['R2'])
    best_model_data = results[best_model_name]
    
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"   R² Score: {best_model_data['R2']:.4f}")
    
    # -----------------------------------------------------------------------
    # 6.6 Save Best Model
    # -----------------------------------------------------------------------
    model_package = {
        'model': best_model_data['model'],
        'scaler': best_model_data['scaler'],
        'feature_columns': feature_cols,
        'model_name': best_model_name,
        'metrics': {
            'R2': best_model_data['R2'],
            'RMSE': best_model_data['RMSE'],
            'MAE': best_model_data['MAE'],
            'MSE': best_model_data['MSE']
        }
    }
    
    with open('models/regression_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    print("   ✓ Model saved to 'models/regression_model.pkl'")
    
    # -----------------------------------------------------------------------
    # 6.7 Create Visualization: Actual vs Predicted
    # -----------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, best_model_data['predictions'], alpha=0.5, s=30, color='blue', edgecolor='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Rating', fontsize=12)
    plt.ylabel('Predicted Rating', fontsize=12)
    plt.title(f'Actual vs Predicted Ratings\n{best_model_name} (R² = {best_model_data["R2"]:.4f})', 
             fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_plots/regression_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Actual vs Predicted plot saved")
    
    print("\n" + "="*80)
    print("✓ REGRESSION MODEL TRAINING COMPLETE!")
    print("="*80)
    
    return model_package, results


# ============================================================================
# STEP 7: CLASSIFICATION MODEL - VISIT MODE PREDICTION
# ============================================================================

def train_classification_model(df):
    """
    Train classification model to predict visit mode
    - Use Random Forest Classifier
    - Evaluate using accuracy, precision, recall, F1-score
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
    
    Returns:
        dict: Model data and metrics
    """
    print("\n[STEP 7] Training Classification Model (Visit Mode Prediction)...")
    print("-" * 80)
    
    # -----------------------------------------------------------------------
    # 7.1 Select Features for Classification
    # -----------------------------------------------------------------------
    feature_cols = [
        'VisitYear', 'VisitMonth', 'Rating', 'UserVisitCount', 'AttractionVisitCount',
        'UserAvgRating', 'AttractionAvgRating', 'AttractionTypePopularity'
    ]
    
    # Add encoded features (exclude VisitMode encoding as it's the target)
    encoded_features = [col for col in df.columns if col.endswith('_Encoded') and 'VisitMode' not in col]
    feature_cols.extend(encoded_features)
    
    # Keep only available features
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"\nFeatures used for classification ({len(feature_cols)} features):")
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i}. {col}")
    
    # -----------------------------------------------------------------------
    # 7.2 Prepare X and y
    # -----------------------------------------------------------------------
    X = df[feature_cols].copy()
    y = df['VisitMode'].copy()
    
    # Handle missing values
    X.fillna(X.median(), inplace=True)
    
    # -----------------------------------------------------------------------
    # 7.3 Train-Test Split (Stratified)
    # -----------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"   Training set: {X_train.shape}")
    print(f"   Testing set: {X_test.shape}")
    
    print(f"\nClass distribution in training set:")
    print(y_train.value_counts())
    
    # -----------------------------------------------------------------------
    # 7.4 Train Random Forest Classifier
    # -----------------------------------------------------------------------
    print("\n7.4 Training Random Forest Classifier...")
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle imbalanced classes
    )
    
    clf.fit(X_train, y_train)
    print("   ✓ Model training complete")
    
    # -----------------------------------------------------------------------
    # 7.5 Make Predictions
    # -----------------------------------------------------------------------
    y_pred = clf.predict(X_test)
    
    # -----------------------------------------------------------------------
    # 7.6 Calculate Metrics
    # -----------------------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n✓ Model Performance Metrics:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # -----------------------------------------------------------------------
    # 7.7 Classification Report
    # -----------------------------------------------------------------------
    print(f"\n7.7 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # -----------------------------------------------------------------------
    # 7.8 Confusion Matrix
    # -----------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=clf.classes_, yticklabels=clf.classes_,
               cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Visit Mode Prediction', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Visit Mode', fontsize=12)
    plt.xlabel('Predicted Visit Mode', fontsize=12)
    plt.tight_layout()
    plt.savefig('eda_plots/classification_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Confusion matrix plot saved")
    
    # -----------------------------------------------------------------------
    # 7.9 Feature Importance
    # -----------------------------------------------------------------------
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    colors = sns.color_palette('viridis', len(top_features))
    bars = plt.barh(range(len(top_features)), top_features['importance'].values, 
                   color=colors, edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['feature'].values, fontsize=10)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Top 15 Feature Importances - Visit Mode Classification', 
             fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_plots/classification_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Feature importance plot saved")
    
    # -----------------------------------------------------------------------
    # 7.10 Save Model
    # -----------------------------------------------------------------------
    model_data = {
        'model': clf,
        'feature_columns': feature_cols,
        'classes': clf.classes_.tolist(),
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    }
    
    with open('models/classification_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("   ✓ Model saved to 'models/classification_model.pkl'")
    
    print("\n" + "="*80)
    print("✓ CLASSIFICATION MODEL TRAINING COMPLETE!")
    print("="*80)
    
    return model_data


# ============================================================================
# STEP 8: RECOMMENDATION SYSTEM
# ============================================================================

def build_recommendation_system(df):
    """
    Build collaborative filtering and content-based recommendation systems
    - User-based collaborative filtering
    - Content-based filtering (attraction similarity)
    
    Args:
        df (pd.DataFrame): Master dataset
    
    Returns:
        dict: Recommendation system components
    """
    print("\n[STEP 8] Building Recommendation System...")
    print("-" * 80)
    
    # -----------------------------------------------------------------------
    # 8.1 Collaborative Filtering (User-Based)
    # -----------------------------------------------------------------------
    print("\n8.1 Building Collaborative Filtering Model...")
    
    # Create user-item rating matrix
    user_item_matrix = df.pivot_table(
        index='UserId',
        columns='AttractionId',
        values='Rating',
        aggfunc='mean'
    ).fillna(0)
    
    print(f"   User-Item Matrix shape: {user_item_matrix.shape}")
    print(f"   Total users: {user_item_matrix.shape[0]}")
    print(f"   Total attractions: {user_item_matrix.shape[1]}")
    
    # Compute user similarity matrix using cosine similarity
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    print("   ✓ User similarity matrix computed")
    
    # -----------------------------------------------------------------------
    # 8.2 Content-Based Filtering (Attraction Similarity)
    # -----------------------------------------------------------------------
    print("\n8.2 Building Content-Based Filtering Model...")
    
    # Get unique attractions with their features - convert IDs to strings
    attraction_features = df[['AttractionId', 'AttractionTypeId', 'AttractionCityId']].drop_duplicates().copy()
    attraction_features['AttractionTypeId'] = attraction_features['AttractionTypeId'].astype(str)
    attraction_features['AttractionCityId'] = attraction_features['AttractionCityId'].astype(str)
    
    # One-hot encode attraction features
    attraction_features_encoded = pd.get_dummies(
        attraction_features,
        columns=['AttractionTypeId', 'AttractionCityId'],
        prefix=['Type', 'City']
    )
    
    attraction_features_encoded.set_index('AttractionId', inplace=True)
    
    # Compute attraction similarity
    attraction_similarity = cosine_similarity(attraction_features_encoded)
    attraction_similarity_df = pd.DataFrame(
        attraction_similarity,
        index=attraction_features_encoded.index,
        columns=attraction_features_encoded.index
    )
    
    print("   ✓ Attraction similarity matrix computed")
    
    # -----------------------------------------------------------------------
    # 8.3 Prepare Attraction Information for Recommendations
    # -----------------------------------------------------------------------
    attraction_info = df[[
        'AttractionId', 'Attraction', 'AttractionType', 
        'AttractionCityName', 'AttractionAvgRating'
    ]].drop_duplicates()
    
    print(f"   ✓ Prepared information for {len(attraction_info)} unique attractions")
    
    # -----------------------------------------------------------------------
    # 8.4 Save Recommendation System
    # -----------------------------------------------------------------------
    recommendation_data = {
        'user_item_matrix': user_item_matrix,
        'user_similarity': user_similarity_df,
        'attraction_similarity': attraction_similarity_df,
        'attraction_info': attraction_info
    }
    
    with open('models/recommendation_system.pkl', 'wb') as f:
        pickle.dump(recommendation_data, f)
    
    print("   ✓ Recommendation system saved to 'models/recommendation_system.pkl'")
    
    print("\n" + "="*80)
    print("✓ RECOMMENDATION SYSTEM BUILD COMPLETE!")
    print("="*80)
    
    return recommendation_data


# ============================================================================
# STEP 9: GENERATE SUMMARY REPORT
# ============================================================================

def generate_summary_report(df, regression_results, classification_results):
    """
    Generate a comprehensive summary report
    
    Args:
        df (pd.DataFrame): Master dataset
        regression_results (dict): Regression model results
        classification_results (dict): Classification model results
    """
    print("\n[STEP 9] Generating Summary Report...")
    print("-" * 80)
    
    report = []
    report.append("="*80)
    report.append("TOURISM EXPERIENCE ANALYTICS - COMPREHENSIVE SUMMARY REPORT")
    report.append("="*80)
    report.append("")
    report.append(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # -----------------------------------------------------------------------
    # Dataset Summary
    # -----------------------------------------------------------------------
    report.append("1. DATASET SUMMARY")
    report.append("-" * 80)
    report.append(f"Total Records: {len(df):,}")
    report.append(f"Total Features: {len(df.columns)}")
    
    if 'VisitYear' in df.columns:
        report.append(f"Date Range: {int(df['VisitYear'].min())} - {int(df['VisitYear'].max())}")
    
    if 'UserId' in df.columns:
        report.append(f"Unique Users: {df['UserId'].nunique():,}")
    
    if 'AttractionId' in df.columns:
        report.append(f"Unique Attractions: {df['AttractionId'].nunique():,}")
    
    if 'ContinentName' in df.columns:
        report.append(f"Continents Covered: {df['ContinentName'].nunique()}")
    
    if 'CountryName' in df.columns:
        report.append(f"Countries Covered: {df['CountryName'].nunique()}")
    
    report.append("")
    
    # -----------------------------------------------------------------------
    # Key Insights
    # -----------------------------------------------------------------------
    report.append("2. KEY INSIGHTS")
    report.append("-" * 80)
    
    if 'Rating' in df.columns:
        report.append(f"Average Rating: {df['Rating'].mean():.2f} / 5.00")
        report.append(f"Median Rating: {df['Rating'].median():.1f}")
        report.append(f"Most Common Rating: {df['Rating'].mode()[0]:.1f}")
    
    if 'VisitMode' in df.columns:
        most_common_mode = df['VisitMode'].mode()[0]
        mode_percentage = (df['VisitMode'].value_counts().iloc[0] / len(df)) * 100
        report.append(f"Most Common Visit Mode: {most_common_mode} ({mode_percentage:.1f}%)")
    
    if 'AttractionType' in df.columns:
        most_popular_type = df['AttractionType'].mode()[0]
        report.append(f"Most Popular Attraction Type: {most_popular_type}")
    
    if 'ContinentName' in df.columns:
        top_continent = df['ContinentName'].mode()[0]
        report.append(f"Top Continent by Visits: {top_continent}")
    
    if 'Season' in df.columns and 'Rating' in df.columns:
        best_season = df.groupby('Season')['Rating'].mean().idxmax()
        report.append(f"Best Rated Season: {best_season}")
    
    report.append("")
    
    # -----------------------------------------------------------------------
    # Model Performance
    # -----------------------------------------------------------------------
    report.append("3. MODEL PERFORMANCE SUMMARY")
    report.append("-" * 80)
    
    # Regression Results
    report.append("A. REGRESSION MODEL (Rating Prediction):")
    report.append("")
    for model_name, results in regression_results.items():
        report.append(f"   {model_name}:")
        report.append(f"      R2 Score:  {results['R2']:.4f}")
        report.append(f"      RMSE:      {results['RMSE']:.4f}")
        report.append(f"      MAE:       {results['MAE']:.4f}")
        report.append(f"      MSE:       {results['MSE']:.4f}")
        report.append("")
    
    # Classification Results
    report.append("B. CLASSIFICATION MODEL (Visit Mode Prediction):")
    report.append("")
    metrics = classification_results['metrics']
    report.append(f"   Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    report.append(f"   Precision:  {metrics['precision']:.4f}")
    report.append(f"   Recall:     {metrics['recall']:.4f}")
    report.append(f"   F1-Score:   {metrics['f1_score']:.4f}")
    report.append("")
    
    # Recommendation System
    report.append("C. RECOMMENDATION SYSTEM:")
    report.append("")
    report.append("   [OK] Collaborative Filtering: User-based recommendation system")
    report.append("   [OK] Content-Based Filtering: Attraction similarity system")
    report.append("")
    
    # -----------------------------------------------------------------------
    # Business Recommendations
    # -----------------------------------------------------------------------
    report.append("4. BUSINESS RECOMMENDATIONS")
    report.append("-" * 80)
    
    if 'Rating' in df.columns:
        avg_rating = df['Rating'].mean()
        if avg_rating >= 4.0:
            report.append("[+] High user satisfaction overall - maintain service quality")
        elif avg_rating >= 3.0:
            report.append("[!] Moderate satisfaction - identify and improve low-rated attractions")
        else:
            report.append("[!] Low satisfaction - urgent service improvements needed")
    
    if 'VisitMode' in df.columns:
        top_mode = df['VisitMode'].value_counts().index[0]
        report.append(f"[+] Focus marketing on {top_mode} travel packages")
    
    if 'AttractionType' in df.columns:
        top_type = df['AttractionType'].value_counts().index[0]
        report.append(f"[+] Promote {top_type} attractions more prominently")
    
    report.append("[+] Use personalized recommendations to increase user engagement")
    report.append("[+] Target seasonal promotions based on rating patterns")
    report.append("")
    
    # -----------------------------------------------------------------------
    # Files Generated
    # -----------------------------------------------------------------------
    report.append("5. OUTPUT FILES GENERATED")
    report.append("-" * 80)
    report.append("[OK] master_dataset.csv - Integrated and processed dataset")
    report.append("[OK] eda_plots/ - 14+ visualization plots")
    report.append("[OK] models/regression_model.pkl - Rating prediction model")
    report.append("[OK] models/classification_model.pkl - Visit mode prediction model")
    report.append("[OK] models/recommendation_system.pkl - Recommendation engine")
    report.append("[OK] models/label_encoders.pkl - Categorical encoders")
    report.append("")
    
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    # Save report to file with UTF-8 encoding
    with open('SUMMARY_REPORT.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    # Print to console
    print('\n'.join(report))
    print("\n[OK] Summary report saved to 'SUMMARY_REPORT.txt'")


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main execution function - runs the complete pipeline
    """
    print("\n" + "="*80)
    print("STARTING TOURISM ANALYTICS COMPLETE PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Load Data
    data = load_data()
    if data is None or all(v is None for v in data.values()):
        print("\n✗ ERROR: Could not load data. Please check that Excel files are in Dataset folder")
        return
    
    # Step 2: Clean Data
    cleaned_data = clean_data(data)
    
    # Step 3: Create Master Dataset
    df_master = create_master_dataset(cleaned_data)
    
    # Save master dataset
    df_master.to_csv('master_dataset.csv', index=False)
    print(f"\n✓ Master dataset saved to 'master_dataset.csv'")
    
    # Step 4: Perform EDA
    perform_eda(df_master)
    
    # Step 5: Preprocess for ML
    df_ml, encoders = preprocess_for_ml(df_master)
    
    # Step 6: Train Regression Model
    regression_model, regression_results = train_regression_model(df_ml)
    
    # Step 7: Train Classification Model
    classification_model = train_classification_model(df_ml)
    
    # Step 8: Build Recommendation System
    recommendation_system = build_recommendation_system(df_master)
    
    # Step 9: Generate Summary Report
    generate_summary_report(df_master, regression_results, classification_model)
    
    # Final Summary
    print("\n" + "="*80)
    print(" PIPELINE EXECUTION COMPLETE! ")
    print("="*80)
    print("\n OUTPUT FILES GENERATED:")
    print("    master_dataset.csv - Integrated dataset")
    print("    eda_plots/ folder - 14 visualization plots")
    print("    models/regression_model.pkl - Rating prediction model")
    print("    models/classification_model.pkl - Visit mode prediction model")
    print("    models/recommendation_system.pkl - Recommendation system")
    print("    models/label_encoders.pkl - Label encoders")
    print("    SUMMARY_REPORT.txt - Comprehensive analysis report")
    
    print("\n" + "="*80 + "\n")

# ============================================================================
# RUN THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    main()