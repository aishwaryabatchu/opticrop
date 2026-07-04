# OptiCrop - Smart Agricultural Production Optimization System

## Overview

OptiCrop is an AI-powered Smart Agricultural Production Optimization System that recommends the most suitable crop based on soil and environmental conditions. The application uses Machine Learning techniques to analyze soil parameters and provide intelligent crop recommendations, helping farmers improve productivity and support sustainable agriculture.

---

## Features

- User-friendly web interface
- Soil data analysis
- Crop recommendation using Machine Learning
- Prediction report generation
- Model accuracy evaluation
- Secure data management

---

## Technologies Used

### Programming Language
- Python 3.x

### Framework
- Flask

### Machine Learning
- Scikit-learn

### Data Processing
- NumPy
- Pandas

### Data Visualization
- Matplotlib
- Seaborn

### Database
- SQLite

### Frontend
- HTML
- CSS
- JavaScript

### IDE
- PyCharm

### Environment
- Anaconda Navigator

---

# Pre-requisites

Before running the project, install the following software:

- Python 3.x
- Anaconda Navigator
- PyCharm IDE
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Flask

Install all required Python libraries:

```bash
pip install flask numpy pandas matplotlib seaborn scikit-learn
```

---

# Project Flow

1. User enters soil information.
2. System validates the input.
3. Soil data is preprocessed.
4. Machine Learning model analyzes the data.
5. Best suitable crop is predicted.
6. Prediction report is generated.
7. Results are displayed to the user.

---

# Entity Relationship Diagram

### Entities

- User
- SoilData
- Crop
- Dataset
- MLModel
- Prediction
- Report

### Relationships

- User → SoilData (One to Many)
- SoilData → Prediction (One to One)
- Crop → Prediction (One to Many)
- Dataset → MLModel (One to Many)
- MLModel → Prediction (One to Many)
- Prediction → Report (One to Many)

---

# Machine Learning Model

Algorithm Used:

- Random Forest Classifier

Libraries:

- Scikit-learn
- Pandas
- NumPy

---

# Dataset

The dataset contains agricultural information including:

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall
- Crop Label

---

# Folder Structure

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

---

# Future Enhancements

- Weather API Integration
- Fertilizer Recommendation
- Disease Prediction
- Mobile Application
- Cloud Deployment

---
