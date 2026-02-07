import pandas as pd
import os

# Set dataset path
DATA_PATH = os.path.join(os.getcwd(), "Dataset")

print("=" * 50)
print("STEP 1: Loading and Verifying Data")
print("=" * 50)

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
    try:
        filepath = os.path.join(DATA_PATH, file)
        data[name] = pd.read_excel(filepath)
        print(f"\n✓ {name} loaded successfully")
        print(f"  Shape: {data[name].shape}")
        print(f"  Columns: {list(data[name].columns)}")
    except FileNotFoundError:
        print(f"\n✗ ERROR: {file} not found in Dataset folder")
    except Exception as e:
        print(f"\n✗ ERROR loading {name}: {str(e)}")

print("\n" + "=" * 50)
print("Data loading complete!")
print("=" * 50)

import pandas as pd
import os

# Set dataset path
DATA_PATH = os.path.join(os.getcwd(), "Dataset")

print("=" * 50)
print("STEP 2: Data Preprocessing & Merging")
print("=" * 50)

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
    print(f"✓ {name} loaded: {data[name].shape}")

print("\n" + "=" * 50)
print("Starting Data Merge...")
print("=" * 50)

# Extract individual dataframes
city = data["City"]
user = data["User"]
trans = data["Transaction"]
item = data["Item"]
typ = data["Type"]

print(f"\nTransaction columns: {list(trans.columns)}")
print(f"User columns: {list(user.columns)}")
print(f"Item columns: {list(item.columns)}")
print(f"Type columns: {list(typ.columns)}")
print(f"City columns: {list(city.columns)}")

# Merge step by step
print("\n--- Merging Transaction + User ---")
merged = trans.merge(user, on="UserId", how="left")
print(f"After User merge: {merged.shape}")

print("\n--- Merging + Item ---")
merged = merged.merge(item, on="AttractionId", how="left")
print(f"After Item merge: {merged.shape}")

print("\n--- Merging + Type ---")
merged = merged.merge(typ, on="AttractionTypeId", how="left")
print(f"After Type merge: {merged.shape}")

print("\n--- Merging + City ---")
merged = merged.merge(city.add_prefix("User_"), left_on="CityId", right_on="User_CityId", how="left")
print(f"After City merge: {merged.shape}")

print(f"\n--- Columns in merged dataset ---")
print(list(merged.columns))

print("\n--- Checking for missing values ---")
print(merged.isnull().sum())

print("\n--- Dropping rows with missing values ---")
merged_clean = merged.dropna()
print(f"Before dropna: {merged.shape}")
print(f"After dropna: {merged_clean.shape}")

print("\n" + "=" * 50)
print("Data preprocessing complete!")
print("=" * 50)

# Save sample to verify
print("\n--- First 5 rows of clean data ---")
print(merged_clean.head())



import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set dataset path
DATA_PATH = os.path.join(os.getcwd(), "Dataset")

print("=" * 50)
print("STEP 3: Regression Model - Rating Prediction")
print("=" * 50)

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

# Preprocess data
city = data["City"]
user = data["User"]
trans = data["Transaction"]
item = data["Item"]
typ = data["Type"]

merged = trans.merge(user, on="UserId", how="left") \
              .merge(item, on="AttractionId", how="left") \
              .merge(typ, on="AttractionTypeId", how="left") \
              .merge(city.add_prefix("User_"), left_on="CityId", right_on="User_CityId", how="left")

df = merged.dropna()

print(f"\n✓ Data loaded and merged: {df.shape}")

# Check required columns for regression
required_cols = ['VisitYear', 'VisitMonth', 'AttractionTypeId', 'ContinentId', 'CountryId', 'Rating']
print(f"\n--- Checking required columns ---")
for col in required_cols:
    if col in df.columns:
        print(f"✓ {col} exists")
    else:
        print(f"✗ {col} MISSING!")

# Build regression model
print("\n" + "=" * 50)
print("Training Regression Model...")
print("=" * 50)

X = df[['VisitYear', 'VisitMonth', 'AttractionTypeId', 'ContinentId', 'CountryId']]
y = df['Rating']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target (Rating) range: {y.min()} to {y.max()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

print("\n✓ Model trained!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "=" * 50)
print("Model Performance")
print("=" * 50)
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Test prediction
print("\n" + "=" * 50)
print("Test Prediction")
print("=" * 50)
sample_input = X_test.iloc[0:1]
print(f"Input: {sample_input.values}")
prediction = model.predict(sample_input)[0]
actual = y_test.iloc[0]
print(f"Predicted Rating: {prediction:.2f}")
print(f"Actual Rating: {actual:.2f}")

print("\n" + "=" * 50)
print("Regression Model Complete!")
print("=" * 50)


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report

# Set dataset path
DATA_PATH = os.path.join(os.getcwd(), "Dataset")

print("=" * 50)
print("STEP 4: Classification Model - Visit Mode Prediction")
print("=" * 50)

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

# Preprocess data
city = data["City"]
user = data["User"]
trans = data["Transaction"]
item = data["Item"]
typ = data["Type"]

merged = trans.merge(user, on="UserId", how="left") \
              .merge(item, on="AttractionId", how="left") \
              .merge(typ, on="AttractionTypeId", how="left") \
              .merge(city.add_prefix("User_"), left_on="CityId", right_on="User_CityId", how="left")

df = merged.dropna()

print(f"\n✓ Data loaded and merged: {df.shape}")

# Check required columns for classification
required_cols = ['VisitYear', 'VisitMonth', 'ContinentId', 'CountryId', 'VisitMode']
print(f"\n--- Checking required columns ---")
for col in required_cols:
    if col in df.columns:
        print(f"✓ {col} exists")
    else:
        print(f"✗ {col} MISSING!")

# Check VisitMode values
print(f"\n--- VisitMode Distribution ---")
print(df['VisitMode'].value_counts())

# Build classification model
print("\n" + "=" * 50)
print("Training Classification Model...")
print("=" * 50)

X = df[['VisitYear', 'VisitMonth', 'ContinentId', 'CountryId']]
y = df['VisitMode']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nLabel Encoding:")
for i, label in enumerate(le.classes_):
    print(f"  {label} -> {i}")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

print("\n✓ Model trained!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("Model Performance")
print("=" * 50)
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")

print("\n--- Classification Report ---")
# Convert classes to strings for classification_report
target_names = [str(label) for label in le.classes_]
print(classification_report(y_test, y_pred, target_names=target_names))

# Test prediction
print("\n" + "=" * 50)
print("Test Prediction")
print("=" * 50)
sample_input = X_test.iloc[0:1]
print(f"Input: {sample_input.values}")
prediction = model.predict(sample_input)[0]
predicted_mode = le.inverse_transform([prediction])[0]
actual_mode = le.inverse_transform([y_test[0]])[0]
print(f"Predicted Visit Mode: {predicted_mode}")
print(f"Actual Visit Mode: {actual_mode}")

print("\n" + "=" * 50)
print("Classification Model Complete!")
print("=" * 50)

import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = os.path.join(os.getcwd(), "Dataset")

print("STEP 5: Recommendation System\n" + "="*50)

# Load data
files = {"City": "City.xlsx", "User": "User.xlsx", "Transaction": "Transaction.xlsx", 
         "Item": "Item.xlsx", "Type": "Type.xlsx"}
data = {name: pd.read_excel(os.path.join(DATA_PATH, file)) for name, file in files.items()}

# Merge
df = data["Transaction"].merge(data["User"], on="UserId", how="left") \
    .merge(data["Item"], on="AttractionId", how="left") \
    .merge(data["Type"], on="AttractionTypeId", how="left") \
    .merge(data["City"].add_prefix("User_"), left_on="CityId", right_on="User_CityId", how="left").dropna()

print(f"✓ Data ready: {df.shape}")

# Build recommendation system
pivot = df.pivot_table(index='UserId', columns='Attraction', values='Rating').fillna(0)
sim_df = pd.DataFrame(cosine_similarity(pivot), index=pivot.index, columns=pivot.index)

def recommend(user_id):
    if user_id not in sim_df.index:
        return pd.DataFrame({'Attraction': ["Not enough data"], 'Rating': [None]})
    similar_users = sim_df[user_id].sort_values(ascending=False)[1:4].index
    return df[df['UserId'].isin(similar_users)].groupby('Attraction')['Rating'].mean().sort_values(ascending=False).head(5).reset_index()

# Test
user = df['UserId'].iloc[0]
print(f"\nRecommendations for User {user}:\n{recommend(user)}")
print("\n✓ Recommendation System Complete!")