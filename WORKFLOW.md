# OptiCrop Workflow

## Epic 1: Define Problem and Understanding

### Story 1
Identify the agricultural problem:
Farmers often face difficulty in selecting the most suitable crop based on soil nutrients and environmental conditions.

### Story 2
Business Requirements:
- Improve crop productivity.
- Reduce crop failure.
- Provide AI-based crop recommendations.
- Support sustainable farming.

### Story 3
Literature Survey:
Existing crop recommendation systems use Machine Learning algorithms such as:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Means Clustering

### Story 4
Business & Social Impact:
- Higher agricultural productivity.
- Better decision making.
- Increased farmer income.
- Sustainable agriculture.

---

# Epic 2: Data Collection and Analysis

### Story 1
Collected Crop Recommendation Dataset from Kaggle.

### Story 2
Imported Libraries:
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Story 3
Dataset Analysis:
- Read dataset using Pandas.
- Checked rows and columns.
- Identified target variable.

### Story 4
Univariate Analysis:
Analyzed individual features using histograms and boxplots.

### Story 5
Bivariate Analysis:
Studied relationship between soil nutrients and crop labels.

### Story 6
Multivariate Analysis:
Generated correlation heatmap and feature relationships.

---

# Epic 3: Data Preprocessing

### Story 1
Checked missing values.

### Story 2
Detected and removed outliers.

### Story 3
Prepared crop season related features.

### Story 4
Split dataset into:
- Training Data (80%)
- Testing Data (20%)

---

# Epic 4: Model Building

### Story 1
Applied K-Means Clustering.

### Story 2
Built Logistic Regression model.

### Story 3
Evaluated model using:
- Accuracy
- Precision
- Recall
- F1 Score

### Story 4
Saved trained model using Pickle.

---

# Epic 5: Application Building

### Story 1
Designed HTML pages.

### Story 2
Integrated Flask backend with ML model.

### Story 3
Tested the application and verified crop prediction results.