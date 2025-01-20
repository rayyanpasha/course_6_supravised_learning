import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('inliers.csv')

# Select relevant columns and target variable
X = data[['Gender', 'Age', 'Academic Pressure', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress']]
y = data['Depression']  # Target variable

# Handle categorical variables (if any)
X = pd.get_dummies(X, drop_first=True)

# Normalize/Scale the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Streamlit Dashboard ---
st.title("Depression Prediction Dashboard")

# Sidebar Input Fields for Feature Values
st.sidebar.header("Input Features")
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
age = st.sidebar.slider("Age", min_value=18, max_value=40, value=22)
academic_pressure = st.sidebar.slider("Academic Pressure", min_value=1, max_value=5, value=3)
study_satisfaction = st.sidebar.slider("Study Satisfaction", min_value=1, max_value=5, value=3)
work_study_hours = st.sidebar.slider("Work/Study Hours", min_value=1, max_value=24, value=6)
financial_stress = st.sidebar.slider("Financial Stress", min_value=1, max_value=5, value=3)

# Convert categorical 'Gender' to numerical value (Male = 1, Female = 0)
gender = 1 if gender == "Male" else 0

# Create a DataFrame for user input
user_input = pd.DataFrame({
    'Gender': [gender],
    'Age': [age],
    'Academic Pressure': [academic_pressure],
    'Study Satisfaction': [study_satisfaction],
    'Work/Study Hours': [work_study_hours],
    'Financial Stress': [financial_stress]
})

# Handle categorical variables for input (if any)
user_input = pd.get_dummies(user_input, drop_first=True)

# Normalize/Scale the user input
user_input_scaled = scaler.transform(user_input)

# --- Model Predictions ---

# Logistic Regression with best hyperparameters
log_reg = LogisticRegression(solver='saga', C=0.01, max_iter=2000, random_state=42)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(user_input_scaled)

# Decision Tree Classifier with best hyperparameters
decision_tree = DecisionTreeClassifier(max_depth=50, min_samples_split=10, min_samples_leaf=30, max_features=None, random_state=42)
decision_tree.fit(X_train, y_train)
dt_pred = decision_tree.predict(user_input_scaled)

# K-Nearest Neighbors (KNN) with best hyperparameters
knn = KNeighborsClassifier(weights='uniform', n_neighbors=50, metric='manhattan', leaf_size=10)
knn.fit(X_train, y_train)
knn_pred = knn.predict(user_input_scaled)

# Display predictions
st.subheader("Model Predictions")
st.write(f"Prediction from Logistic Regression: {'Depressed' if log_reg_pred[0] == 1 else 'Not Depressed'}")
st.write(f"Prediction from Decision Tree: {'Depressed' if dt_pred[0] == 1 else 'Not Depressed'}")
st.write(f"Prediction from K-Nearest Neighbors: {'Depressed' if knn_pred[0] == 1 else 'Not Depressed'}")

# --- Model Evaluation Metrics (Precision & Recall) ---

# Logistic Regression Evaluation
log_reg_precision = precision_score(y_test, log_reg.predict(X_test))
log_reg_recall = recall_score(y_test, log_reg.predict(X_test))

# Decision Tree Evaluation
dt_precision = precision_score(y_test, dt_best_model.predict(X_test))
dt_recall = recall_score(y_test, dt_best_model.predict(X_test))

# KNN Evaluation
knn_precision = precision_score(y_test, knn_best_model.predict(X_test))
knn_recall = recall_score(y_test, knn_best_model.predict(X_test))

# --- Show Model Comparison (Precision & Recall) ---
model_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Tuned Decision Tree', 'Tuned KNN'],
    'Precision': [log_reg_precision, dt_precision, knn_precision],
    'Recall': [log_reg_recall, dt_recall, knn_recall]
})

st.subheader("Model Comparison (Precision & Recall)")
st.write(model_comparison)
