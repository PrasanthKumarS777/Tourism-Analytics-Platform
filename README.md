<div align="center">

# 🌍 Tourism Experience Analytics Platform

### Complete Machine Learning Pipeline for Tourism Data Analysis, Predictions & Recommendations

<br/>

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

<br/>

> An end-to-end machine learning platform that transforms raw tourism data into smart predictions,  
> visit mode classifications, and personalized attraction recommendations — all via an interactive Streamlit dashboard.

<br/>

[🌟 Overview](#-project-overview) &nbsp;•&nbsp; [✨ Features](#-features) &nbsp;•&nbsp; [🚀 Installation](#-installation--setup) &nbsp;•&nbsp; [📖 Usage Guide](#-usage-guide) &nbsp;•&nbsp; [📊 Model Performance](#-model-performance) &nbsp;•&nbsp; [🔧 Troubleshooting](#-troubleshooting)

</div>

---

## 📋 Table of Contents

- [🌟 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [🛠️ Technical Stack](#️-technical-stack)
- [📊 Dataset Details](#-dataset-details)
- [📁 Project Structure](#-project-structure)
- [⚙️ Prerequisites](#️-prerequisites)
- [🚀 Installation & Setup](#-installation--setup)
- [▶️ Running the Application](#️-running-the-application)
- [📖 Usage Guide](#-usage-guide)
- [🎯 Dashboard Pages](#-dashboard-pages)
- [📈 Model Performance](#-model-performance)
- [🔧 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)
- [👨‍💻 Author](#-author)

---

## 🌟 Project Overview

The **Tourism Experience Analytics Platform** is a full-stack machine learning application that analyzes tourism behavior data to deliver actionable insights. It combines data engineering, predictive modeling, and an interactive web dashboard to help understand what drives traveler satisfaction and attraction popularity.

<div align="center">

| ⭐ Rating Prediction | 🧭 Visit Mode Classification | 💡 Smart Recommendations | 📊 Interactive Dashboard |
|:-:|:-:|:-:|:-:|
| Predict user satisfaction (1–5) for any attraction | Classify travel type: Business, Family, Couples, Friends, Solo | Personalized suggestions via collaborative & content-based filtering | Full Streamlit web app with 6 pages of insights |

</div>

---

## ✨ Features

<details>
<summary><b>📊 Exploratory Data Analysis (EDA)</b></summary>
<br/>

- 16 auto-generated comprehensive visualizations covering the full dataset
- Rating distributions, geographic heatmaps, and temporal trend charts
- Correlation matrices and feature relationship plots
- All plots saved automatically to the `eda_plots/` folder on first run

</details>

<details>
<summary><b>⭐ Rating Prediction — Random Forest Regression</b></summary>
<br/>

- Predicts user satisfaction ratings on a **1–5 scale** for any given attraction
- Model: **Random Forest Regressor** trained on 50,000+ transactions
- Input features include: location, visit details, user history, and attraction attributes
- Achieves **R² Score of 0.85+**, RMSE of 0.45, and MAE of 0.35

</details>

<details>
<summary><b>🧭 Visit Mode Classification — Random Forest Classifier</b></summary>
<br/>

- Classifies the type of travel: **Business, Family, Couples, Friends, or Solo**
- Model: **Random Forest Classifier** with 70%+ accuracy
- Uses engineered features from user demographics and attraction metadata
- F1-Score: 0.68 with balanced Precision/Recall

</details>

<details>
<summary><b>💡 Recommendation System — Hybrid Filtering</b></summary>
<br/>

- **Collaborative Filtering** — User-based recommendations from behavioral similarity
- **Content-Based Filtering** — Attraction similarity via cosine similarity matching
- **Hybrid Approach** — Combines both methods for the best coverage and accuracy
- Supports 5,000+ attractions across 100+ countries

</details>

<details>
<summary><b>🖥️ Interactive Streamlit Dashboard</b></summary>
<br/>

- 6-page web application with real-time model inference
- Fully interactive: input sliders, dropdowns, and instant prediction outputs
- Clean, responsive UI running locally at `http://localhost:8501`
- Model metrics and evaluation visualized on the Performance page

</details>

---

## 🛠️ Technical Stack

<div align="center">

| Category | Technology | Purpose |
|:--------:|:----------:|:-------:|
| ![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white) | Python 3.11+ | Core language |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) | Pandas & NumPy | Data processing & engineering |
| ![Sklearn](https://img.shields.io/badge/Scikit--learn-F7931E?logo=scikit-learn&logoColor=white) | Scikit-learn | ML models, scaling, encoding |
| ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white) | Streamlit | Interactive web dashboard |
| ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white) | Matplotlib / Seaborn / Plotly | Data visualization |

**Machine Learning Models Used:**
`Linear Regression` &nbsp; `Random Forest Regressor` &nbsp; `Random Forest Classifier` &nbsp; `Cosine Similarity`  
`StandardScaler` &nbsp; `LabelEncoder` &nbsp; `Collaborative Filtering` &nbsp; `Content-Based Filtering`

</div>

---

## 📊 Dataset Details

<div align="center">

| Metric | Value |
|:------:|:-----:|
| 📋 Total Transactions | **50,000+** tourism records |
| 👤 Unique Users | **10,000+** travelers |
| 🏛️ Attractions | **5,000+** tourist destinations |
| 🌐 Countries Covered | **100+** countries |
| 🔢 Engineered Features | **30+** features |
| 📂 Source Files | **9 Excel files** |

</div>

### Source Data Files

| File | Contents |
|:----:|:--------:|
| `Transaction.xlsx` | User visit transactions |
| `User.xlsx` | User demographics |
| `City.xlsx` | City information |
| `Item.xlsx` | Attraction details |
| + 5 more | Supporting reference tables |

---

## 📁 Project Structure

```
Tourism-Analytics-Platform/
│
├── 📂 Dataset/                         # Source data (9 Excel files)
│   ├── Transaction.xlsx                # User visit transactions
│   ├── User.xlsx                       # User demographics
│   ├── City.xlsx                       # City information
│   ├── Item.xlsx                       # Attraction details
│   └── ...                             # 5 more supporting files
│
├── 📂 eda_plots/                       # Auto-generated visualizations (16 plots)
│
├── 📂 models/                          # Trained ML models (generated by File.py)
│   ├── regression_model.pkl            # Random Forest Regressor
│   ├── classification_model.pkl        # Random Forest Classifier
│   ├── recommendation_system.pkl       # Hybrid recommendation model
│   └── label_encoders.pkl              # Fitted label encoders
│
├── 📄 File.py                          # ← Data pipeline & ML training script (run this first)
├── 📄 app.py                           # ← Streamlit dashboard application
├── 📄 requirements.txt                 # Python dependencies
├── 📄 master_dataset.csv               # Integrated dataset (auto-generated)
├── 📄 SUMMARY_REPORT.txt               # Analysis summary (auto-generated)
├── 📄 .gitignore
└── 📄 README.md
```

> **Note:** The `models/` folder and `master_dataset.csv` are **not included in the repository** due to file size limits. They are generated locally by running `python File.py` (see [Installation](#-installation--setup)).

---

## ⚙️ Prerequisites

Before you begin, make sure the following are installed on your system:

### 1. Python 3.11 or 3.12

> ⚠️ **Important:** Python 3.14 is **not yet supported** by some dependencies. Use Python **3.11 or 3.12** only.

Check your Python version:
```bash
python --version
# Expected: Python 3.11.x or Python 3.12.x
```

Download Python: [https://www.python.org/downloads/](https://www.python.org/downloads/)

### 2. pip (Python Package Manager)

Check if pip is installed:
```bash
pip --version
```

If not installed:
```bash
python -m ensurepip --upgrade
```

### 3. Git

Download Git: [https://git-scm.com/downloads](https://git-scm.com/downloads)

Check if installed:
```bash
git --version
```

### 4. (Recommended) Virtual Environment

It is strongly recommended to use a virtual environment to avoid dependency conflicts.

---

## 🚀 Installation & Setup

Follow these steps carefully in order. Do **not** skip steps.

---

### Step 1 — Clone the Repository

Open your terminal (Command Prompt / PowerShell on Windows, Terminal on Mac/Linux) and run:

```bash
git clone https://github.com/PrasanthKumarS777/Tourism-Analytics-Platform.git
```

Then navigate into the project folder:

```bash
cd Tourism-Analytics-Platform
```

Verify you're in the right folder:
```bash
ls        # Mac/Linux
dir       # Windows
```
You should see `File.py`, `app.py`, `requirements.txt`, and the `Dataset/` folder listed.

---

### Step 2 — Create a Virtual Environment (Recommended)

Creating a virtual environment keeps dependencies isolated to this project only.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

After activation, your terminal prompt will show `(venv)` at the start — this confirms the virtual environment is active.

> To deactivate the virtual environment later, simply run: `deactivate`

---

### Step 3 — Upgrade pip

Always upgrade pip before installing packages to avoid version conflicts:

```bash
pip install --upgrade pip
```

Expected output:
```
Successfully installed pip-XX.X
```

---

### Step 4 — Install All Dependencies

Install all required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `openpyxl`, and all other dependencies.

> **Note:** This may take 2–5 minutes depending on your internet speed.

Verify key packages installed correctly:
```bash
pip show streamlit
pip show scikit-learn
pip show pandas
```

Each should display version info without errors.

---

### Step 5 — Generate ML Models & Datasets

This is the most important step. Run the data pipeline script to:
- Load and merge all 9 Excel source files
- Clean and engineer 30+ features
- Train the Regression, Classification, and Recommendation models
- Save trained models to the `models/` folder
- Generate `master_dataset.csv` and `SUMMARY_REPORT.txt`
- Create 16 EDA visualizations in `eda_plots/`

```bash
python File.py
```

> ⏱️ **This step takes approximately 2–5 minutes.** Do not interrupt the process.

Expected terminal output (you will see progress messages such as):
```
[1/6] Loading datasets...          ✓
[2/6] Merging and cleaning data... ✓
[3/6] Engineering features...      ✓
[4/6] Training regression model... ✓
[5/6] Training classifier...       ✓
[6/6] Building recommendation system... ✓
All models saved to /models/
master_dataset.csv generated.
SUMMARY_REPORT.txt generated.
EDA plots saved to /eda_plots/
Pipeline complete!
```

After this step, verify the following files/folders now exist:
```bash
ls models/        # Should list: regression_model.pkl, classification_model.pkl, etc.
ls eda_plots/     # Should list 16 .png plot files
ls master_dataset.csv   # Should exist
```

---

## ▶️ Running the Application

Once **Step 5** is complete and all models are generated, launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Expected terminal output:
```
  You can now view your Streamlit app in your browser.

  Local URL:  http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

The dashboard will **automatically open** in your default browser at:

> 🌐 **`http://localhost:8501`**

If it doesn't open automatically, manually copy and paste `http://localhost:8501` into your browser.

To **stop** the server, press `Ctrl + C` in the terminal.

---

## 📖 Usage Guide

### Navigating the Dashboard

The sidebar on the left contains navigation to all 6 pages. Click any page name to switch views.

---

### 🏠 Page 1 — Home

- View **key metrics**: total users, total attractions, countries covered, total transactions
- See the **top-rated attractions** ranked by average user rating
- Get a quick platform overview before diving into analysis

---

### 📈 Page 2 — EDA (Exploratory Data Analysis)

- Browse all **16 auto-generated visualizations** covering the full dataset
- Charts include: rating distribution histograms, geographic heatmaps, visit mode breakdowns, temporal trends, and correlation matrices
- Use the dropdown to filter and view specific plots
- All charts are also saved locally in the `eda_plots/` folder

---

### ⭐ Page 3 — Rating Prediction

Use this page to **predict the satisfaction rating** a user would give to an attraction.

**Steps:**
1. Select or enter an **attraction** from the dropdown
2. Fill in **visit details**: visit month, visit duration, and user demographics
3. Click **"Predict Rating"**
4. The model returns a **predicted rating (1.0 – 5.0)** with a confidence interval

**Under the hood:** The Random Forest Regressor uses 30+ engineered features to output the prediction.

---

### 🧭 Page 4 — Visit Mode Classification

Use this page to **classify the type of trip** based on traveler and attraction characteristics.

**Steps:**
1. Input **traveler profile**: age group, group size, country of origin
2. Select the **attraction type** and destination city
3. Click **"Classify Visit Mode"**
4. The model returns one of: **Business | Family | Couples | Friends | Solo**
5. A confidence bar shows the probability distribution across all classes

---

### 💡 Page 5 — Recommendations

Use this page to get **personalized attraction suggestions**.

**Steps:**
1. Select your **User ID** from the dropdown (or enter manually)
2. Choose your preferred **recommendation method**:
   - `Collaborative` — Based on users similar to you
   - `Content-Based` — Based on attractions similar to ones you've rated
   - `Hybrid` — Combines both methods (recommended)
3. Set the **number of recommendations** (e.g., 5 or 10)
4. Click **"Get Recommendations"**
5. A ranked list of attractions is returned with predicted ratings and similarity scores

---

### 📊 Page 6 — Model Performance

View detailed evaluation metrics for all trained models:

| Model | Metric | Value |
|:-----:|:------:|:-----:|
| Rating Prediction | R² Score | **0.85+** |
| Rating Prediction | RMSE | **0.45** |
| Rating Prediction | MAE | **0.35** |
| Visit Classification | Accuracy | **70%+** |
| Visit Classification | F1-Score | **0.68** |
| Visit Classification | Precision/Recall | **Balanced** |

- View **confusion matrix** for the classifier
- View **actual vs. predicted** scatter plot for regression
- View **feature importance** bar chart for both models

---

## 🎯 Dashboard Pages Summary

<div align="center">

| # | Page | Purpose |
|:-:|:----:|:-------:|
| 1 | 🏠 Home | Platform overview, key metrics, top attractions |
| 2 | 📈 EDA | 16 interactive visualizations |
| 3 | ⭐ Rating Prediction | Predict attraction satisfaction score (1–5) |
| 4 | 🧭 Visit Mode | Classify travel type from user/attraction inputs |
| 5 | 💡 Recommendations | Get personalized attraction suggestions |
| 6 | 📊 Performance | Model evaluation metrics and charts |

</div>

---

## 📈 Model Performance

<div align="center">

### Regression — Rating Prediction (Random Forest Regressor)

| Metric | Score |
|:------:|:-----:|
| R² Score | **0.85+** |
| RMSE | **0.45** |
| MAE | **0.35** |

### Classification — Visit Mode (Random Forest Classifier)

| Metric | Score |
|:------:|:-----:|
| Accuracy | **70%+** |
| F1-Score | **0.68** |
| Precision | Balanced |
| Recall | Balanced |

### Recommendation System

| Method | Description |
|:------:|:-----------:|
| Collaborative Filtering | User-based similarity recommendations |
| Content-Based Filtering | Attraction cosine similarity matching |
| Hybrid | Combined approach for best coverage |

</div>

---

## 🔧 Troubleshooting

<details>
<summary><b>❌ Large model files missing / models folder empty</b></summary>
<br/>

**Cause:** The trained model `.pkl` files are excluded from the GitHub repository due to size constraints.

**Fix:** Run the pipeline script to generate them locally:
```bash
python File.py
```
This will train all models and save them to the `models/` folder. Takes 2–5 minutes.

</details>

<details>
<summary><b>❌ Python 3.14 compatibility errors</b></summary>
<br/>

**Cause:** Some dependencies (e.g., `numpy`, `scikit-learn`) do not yet support Python 3.14.

**Fix:** Use Python 3.11 or 3.12. Create a dedicated virtual environment:

```bash
# Install Python 3.11 from https://www.python.org/downloads/
python3.11 -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Then install dependencies
pip install -r requirements.txt
```

</details>

<details>
<summary><b>❌ Missing package / ModuleNotFoundError</b></summary>
<br/>

**Cause:** One or more packages failed to install or pip is outdated.

**Fix:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

If a specific package fails (e.g., `openpyxl`):
```bash
pip install openpyxl
```

</details>

<details>
<summary><b>❌ Streamlit command not found</b></summary>
<br/>

**Cause:** Streamlit was not installed correctly, or you're outside the virtual environment.

**Fix:**
```bash
# Ensure virtual environment is active first, then:
pip install streamlit

# Then run:
streamlit run app.py
```

</details>

<details>
<summary><b>❌ Dashboard opens but shows errors / blank pages</b></summary>
<br/>

**Cause:** The models haven't been generated yet — `app.py` cannot find the `.pkl` files.

**Fix:** Ensure you have run `python File.py` **before** launching `streamlit run app.py`. Check that `models/` folder contains all four `.pkl` files:
```bash
ls models/
# Should show: regression_model.pkl, classification_model.pkl,
#              recommendation_system.pkl, label_encoders.pkl
```

</details>

<details>
<summary><b>❌ Port 8501 already in use</b></summary>
<br/>

**Cause:** Another Streamlit instance or application is running on port 8501.

**Fix:** Run on a different port:
```bash
streamlit run app.py --server.port 8502
```
Then open `http://localhost:8502` in your browser.

</details>

---

## 🤝 Contributing

Contributions are welcome and appreciated! Here's how to get involved:

```bash
# 1. Fork the repository (click "Fork" on GitHub)

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/Tourism-Analytics-Platform.git

# 3. Create a feature branch
git checkout -b feature/YourFeatureName

# 4. Make your changes, then commit
git add .
git commit -m "✨ Add YourFeatureName"

# 5. Push to your branch
git push origin feature/YourFeatureName

# 6. Open a Pull Request on GitHub
```

**Contribution Guidelines:**
- Follow existing code style and naming conventions
- Test your changes by running the full pipeline (`File.py` → `app.py`) before submitting
- Document any new features or model changes in the README
- Write clear, descriptive commit messages

---

## 📝 License

This project is licensed under the **MIT License** — free to use, modify, and distribute with attribution.

```
✅ Commercial use   ✅ Modification   ✅ Distribution   ✅ Private use
⚠️  Must include copyright notice        ❌ No warranty provided
```

---

## 👨‍💻 Author

<div align="center">

**Prasanth Kumar Sahu**

[![GitHub](https://img.shields.io/badge/GitHub-PrasanthKumarS777-181717?style=for-the-badge&logo=github)](https://github.com/PrasanthKumarS777)

**Skills Demonstrated through this project:**

`Machine Learning` &nbsp; `Python` &nbsp; `Streamlit` &nbsp; `Scikit-learn`  
`Data Engineering` &nbsp; `Recommendation Systems` &nbsp; `EDA & Visualization` &nbsp; `End-to-End ML Pipeline`

</div>

---

<div align="center">

---

**⭐ If you found this project helpful, please consider giving it a star!**

*Built with ❤️ and 🌍 by [Prasanth Kumar Sahu](https://github.com/PrasanthKumarS777)*

</div>
