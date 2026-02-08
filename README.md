 🌍 Tourism Experience Analytics Platform

**Complete Machine Learning Pipeline for Tourism Data Analysis, Predictions & Recommendations**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📊 Project Overview

This project analyzes tourism data to provide:
- **Rating Prediction** - Predict user satisfaction for attractions
- **Visit Mode Classification** - Classify travel types (Business, Family, Couples, etc.)
- **Smart Recommendations** - Personalized attraction suggestions using collaborative & content-based filtering
- **Interactive Dashboard** - Beautiful Streamlit web application

---

## 🚀 Quick Start

### **Prerequisites**
- Python 3.11 or 3.12 (Python 3.14 not yet supported)
- pip package manager

### **Installation**

```bash
# Clone the repository
git clone https://github.com/PrasanthKumarS777/Tourism-Analytics-Platform.git
cd Tourism-Analytics-Platform

# Install dependencies
pip install -r requirements.txt

# Generate ML models and datasets (First time only - takes 2-5 minutes)
python File.py

# Launch the dashboard
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
Tourism-Analytics-Platform/
├── Dataset/                    # Source data (9 Excel files)
│   ├── Transaction.xlsx        # User visit transactions
│   ├── User.xlsx               # User demographics
│   ├── City.xlsx               # City information
│   ├── Item.xlsx               # Attraction details
│   └── ...
├── eda_plots/                  # Generated visualizations (16 plots)
├── models/                     # Trained ML models (generated)
│   ├── regression_model.pkl
│   ├── classification_model.pkl
│   ├── recommendation_system.pkl
│   └── label_encoders.pkl
├── File.py                     # Data pipeline & ML training
├── app.py                      # Streamlit dashboard (197 lines!)
├── requirements.txt            # Python dependencies
├── master_dataset.csv          # Integrated dataset (generated)
├── SUMMARY_REPORT.txt          # Analysis summary (generated)
└── .gitignore
```

---

## ✨ Features

### **1. Data Analysis (EDA)**
- 16 comprehensive visualizations
- Rating distributions, geographic patterns, temporal trends
- Correlation analysis and feature relationships

### **2. Rating Prediction (Regression)**
- **Model:** Random Forest Regressor
- **Predicts:** User satisfaction ratings (1-5 scale)
- **Features:** Location, visit details, user history, attraction attributes

### **3. Visit Mode Classification**
- **Model:** Random Forest Classifier
- **Classifies:** Business, Family, Couples, Friends, Solo travel
- **Accuracy:** 70%+ (varies by data)

### **4. Recommendation System**
- **Collaborative Filtering:** User-based recommendations
- **Content-Based Filtering:** Attraction similarity matching
- **Hybrid Approach:** Best of both methods

---

## 🎯 Dashboard Pages

1. **🏠 Home** - Overview, key metrics, top attractions
2. **📈 EDA** - Interactive visualizations
3. **⭐ Rating Prediction** - Predict attraction ratings
4. **🎯 Visit Mode** - Classify travel type
5. **💡 Recommendations** - Get personalized suggestions
6. **📊 Performance** - Model metrics & evaluation

---

## 🛠️ Technical Stack

**Languages & Libraries:**
- Python 3.11+
- Pandas, NumPy (Data processing)
- Scikit-learn (Machine Learning)
- Matplotlib, Seaborn, Plotly (Visualization)
- Streamlit (Web Dashboard)

**Machine Learning:**
- Linear Regression
- Random Forest (Regression & Classification)
- Cosine Similarity (Recommendations)
- StandardScaler, LabelEncoder

---

## 📊 Dataset Details

- **Total Records:** 50,000+ tourism transactions
- **Users:** 10,000+ unique travelers
- **Attractions:** 5,000+ tourist destinations
- **Countries:** 100+ countries covered
- **Features:** 30+ engineered features

---

## 🔧 Troubleshooting

### **Issue: Large model files not included**
**Solution:** Run `python File.py` to generate all models locally. The large models (`classification_model.pkl` and `recommendation_system.pkl`) are excluded from Git due to size limits.

### **Issue: Python 3.14 compatibility**
**Solution:** Use Python 3.11 or 3.12. Create a virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **Issue: Missing packages**
**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📈 Model Performance

### **Regression (Rating Prediction)**
- R² Score: 0.85+
- RMSE: 0.45
- MAE: 0.35

### **Classification (Visit Mode)**
- Accuracy: 70%+
- F1-Score: 0.68
- Precision/Recall: Balanced

### **Recommendation System**
- User-based collaborative filtering
- Content-based similarity matching
- Hybrid recommendations available

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Prasanth Kumar Sahu**
- GitHub: [@PrasanthKumarS777](https://github.com/PrasanthKumarS777)

---