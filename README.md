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
├── database/
├── models/
├── static/
├── templates/
├── wheels/
├── app.py
├── requirements.txt
├── README.md
└── ER_Diagram.png
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/aishwaryabatchu/opticrop.git
```

Go to project directory

```bash
cd opticrop
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

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
