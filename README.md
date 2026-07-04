# OptiCrop – Smart Agricultural Production Optimization Engine

OptiCrop is a full-stack precision agriculture decision-support system. It utilizes environmental parameters and soil nutrients to recommend compatible crops, diagnose soil suitability levels, and provide advanced analytical reports.

---

## Technical Stack
* **Frontend**: HTML5, CSS3, Bootstrap 5, Javascript, Chart.js (Interactive Charts)
* **Backend**: Python 3.9+, Flask Web Framework, Flask-Login (Session & Security)
* **Machine Learning & Data Analysis**: NumPy, Pandas, Scikit-learn, SciPy, Matplotlib, Seaborn
* **Database**: SQLite (SQL-based prediction logs, security audits, and user credentials)
* **Report Generation**: FPDF2 (PDF compiler)

---

## Project Structure
```
OptiCrop/
│
├── app.py                     # Main Flask Application
├── requirements.txt           # Dependencies configuration
├── model.pkl                  # Serialized champion model (generated after training)
├── scaler.pkl                 # Serialized feature scaler (generated after training)
├── label_encoder.pkl          # Serialized label encoder (generated after training)
├── README.md                  # System Documentation
│
├── dataset/
│   └── Crop_recommendation.csv # Precision agricultural dataset
│
├── database/
│   └── opticrops.db           # SQLite database file (generated on startup)
│   └── db_manager.py          # Database queries & connection managers
│
├── models/
│   ├── train_model.py         # Multi-model fit, validation, & selection pipeline
│   ├── prediction.py          # Inference engine & Z-score suitability calculator
│   └── preprocessing.py       # Data validation & sanity boundaries checks
│
├── static/
│   ├── css/
│   │   └── style.css          # Green-themed stylesheet, layout, & dark mode settings
│   ├── js/
│   │   └── main.js           # Client-side theme handlers, loading spin, validations
│   └── charts/                # Generated Seaborn heatmaps & Matplotlib analytics
│
├── templates/                 # Jinja2 HTML Templates
│   ├── base.html              # Core Layout & Sidebar
│   ├── index.html             # Brand Landing Page
│   ├── login.html             # Login
│   ├── register.html          # Registration
│   ├── dashboard.html         # User Dashboard
│   ├── recommendation.html    # Smart Recommendation System
│   ├── suitability.html       # Suitability Assessment
│   ├── analytics.html         # Research & Analytics Visuals
│   ├── admin_dashboard.html   # System Administration
│   └── reports.html           # Historical logs & filters
│
└── reports/
    └── generated_reports/     # Generated PDF & CSV exports directory
```

---

## Setup & Running Instructions

### 1. Prerequisites
Ensure you have Python 3.x installed and added to your PATH environment. 

### 2. Install Dependencies
Run the following command to download all machine learning and web server packages:
```bash
pip install -r requirements.txt
```

### 3. Initialize Dataset
Retrieve the precision crop dataset. This downloads the standard 2,200 row Crop Recommendation dataset from a public source. If offline, it automatically generates a high-fidelity synthetic agricultural dataset:
```bash
python download_dataset.py
```

### 4. Train Machine Learning Models
Execute the ML pipeline to fit Random Forest, Decision Tree, Naive Bayes, KNN, and SVM models. The script evaluates their cross-validation and test accuracies, selects the highest performing classifier as the "champion model", and exports validation plots to `static/charts/`:
```bash
python models/train_model.py
```

### 5. Start Web Portal
Run the Flask server:
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## Pre-configured User Accounts
The database is initialized with three default credentials mapping to different system roles:

| Username | Password | Access Role | Allowed Actions |
| :--- | :--- | :--- | :--- |
| **admin** | `admin123` | **Admin** | User management, Retraining triggers, Audit logs, full prediction logs, CSV downloads. |
| **researcher** | `researcher123` | **Researcher** | Advanced Seaborn analytics maps, PDF report generation, CSV logs exports, prediction logs. |
| **farmer** | `farmer123` | **Farmer** | Crop recommendations, Soil compatibility assessments, personal prediction logs. |

*(Users can also register custom accounts via the Register tab.)*
