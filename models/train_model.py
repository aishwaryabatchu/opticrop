import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization Style
plt.style.use("fivethirtyeight")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

DATASET_PATH = os.path.join("dataset", "Crop_recommendation.csv")
CHARTS_DIR = os.path.join("static", "charts")
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
METRICS_PATH = "models/model_metrics.json"

def run_training_pipeline():
    print("Starting ML Model Training Pipeline...")
    
    # 1. Load Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please run download_dataset.py first.")
        
    df = pd.read_csv(DATASET_PATH)
    
    print("\nFirst 5 Records:")
    print(df.head())

    
    # Check for missing values and fill/drop
    df = df.dropna()
    
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    # 2. Preprocessing
    # Label Encoding for targets
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # 3. Initialize Models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True, kernel='rbf', random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    # Create charts directory if it does not exist
    os.makedirs(CHARTS_DIR, exist_ok=True)
    
    # 4. Train and Evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        # Train
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Accuracy & CV
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
        cv_mean = cv_scores.mean()
        
        results[name] = {
            'accuracy': float(acc),
            'cv_mean': float(cv_mean),
            'classification_report': classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        }
        print(f"{name} - Test Accuracy: {acc:.4f}, 5-Fold CV Accuracy: {cv_mean:.4f}")
        
    # 5. Select the best performing model based on test accuracy
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = trained_models[best_model_name]
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\nWinner Model: {best_model_name} with Accuracy {best_accuracy:.4f}")
    
    # Save the selected model, scaler, and label encoder
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
        
    print(f"Saved: {MODEL_PATH}, {SCALER_PATH}, and {LABEL_ENCODER_PATH}")
    
    # Save training metrics to JSON for the Admin panel to inspect
    os.makedirs("models", exist_ok=True)
    metrics_summary = {
        'best_model': best_model_name,
        'accuracy': best_accuracy,
        'cv_accuracy': results[best_model_name]['cv_mean'],
        'all_models': {name: {'accuracy': results[name]['accuracy'], 'cv_mean': results[name]['cv_mean']} for name in results}
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
        
    # 6. Generate Comparison Plots
    # Model Comparison Chart
    plt.figure(figsize=(10, 5))
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in model_names]
    cv_means = [results[m]['cv_mean'] * 100 for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Test Accuracy', color='#40916c')
    rects2 = ax.bar(x + width/2, cv_means, width, label='5-Fold CV Accuracy', color='#1b4332')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Crop Recommendation Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 110)
    ax.legend()
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "model_comparison.png"), dpi=300)
    plt.close()
    
    # Confusion Matrix for Best Model
    plt.figure(figsize=(12, 10))
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.ylabel('Actual Crop')
    plt.xlabel('Predicted Crop')
    plt.title(f'Confusion Matrix - {best_model_name} (Accuracy: {best_accuracy*100:.2f}%)')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()
    
    # 7. Generate Feature Importance Plot if the best model is Random Forest or Decision Tree
    if hasattr(best_model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importances = best_model.feature_importances_
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        indices = np.argsort(importances)[::-1]
        
        sns.barplot(x=[importances[i] for i in indices], y=[features[i] for i in indices], palette='Greens_r')
        plt.title(f'Feature Importance - {best_model_name}')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, "feature_importance.png"), dpi=300)
        plt.close()
    else:
        # If best model doesn't support feature importances (e.g. KNN/NB/SVM), train a temporary DecisionTree to get a feature importance chart
        plt.figure(figsize=(10, 6))
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train, y_train)
        importances = dt.feature_importances_
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        indices = np.argsort(importances)[::-1]
        
        sns.barplot(x=[importances[i] for i in indices], y=[features[i] for i in indices], palette='Greens_r')
        plt.title('Feature Importance (via Decision Tree Estimator)')
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(CHARTS_DIR, "feature_importance.png"), dpi=300)
        plt.close()

    print("Pipeline completed successfully.")
    return metrics_summary

if __name__ == "__main__":
    run_training_pipeline()
