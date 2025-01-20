import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Initialize session state if not exists
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()
    st.session_state.scaler_lr = StandardScaler()

@st.cache_data
def load_depression_data():
    data = pd.read_csv('inliers.csv')
    return data

@st.cache_data
def load_cgpa_data():
    return pd.read_csv("data.csv")

def prepare_depression_data(data):
    X = data[['Gender', 'Age', 'Academic Pressure', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress']]
    y = data['Depression']
    X = pd.get_dummies(X, drop_first=True)
    X_scaled = st.session_state.scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def prepare_user_input(user_input):
    user_input = pd.get_dummies(user_input, drop_first=True)
    return st.session_state.scaler.transform(user_input)

def bound_cgpa(cgpa_value):
    """Ensure CGPA stays within 0-10 range"""
    return max(0, min(10, cgpa_value))

# --- Streamlit Dashboard ---
st.title("Student Performance and Wellbeing Prediction Dashboard")

# Sidebar Selection for Prediction Type
prediction_type = st.sidebar.selectbox("Select Prediction Type", 
                                     options=["Depression Prediction", "CGPA Prediction"])

try:
    if prediction_type == "Depression Prediction":
        st.sidebar.header("Input Features for Depression Prediction")
        
        # User input collection
        gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
        age = st.sidebar.slider("Age", min_value=18, max_value=40, value=22)
        academic_pressure = st.sidebar.slider("Academic Pressure", min_value=1, max_value=5, value=3)
        study_satisfaction = st.sidebar.slider("Study Satisfaction", min_value=1, max_value=5, value=3)
        work_study_hours = st.sidebar.slider("Work/Study Hours", min_value=1, max_value=24, value=6)
        financial_stress = st.sidebar.slider("Financial Stress", min_value=1, max_value=5, value=3)

        # Load and prepare data
        data = load_depression_data()
        X_train, X_test, y_train, y_test = prepare_depression_data(data)

        # Prepare user input
        user_input = pd.DataFrame({
            'Gender': [1 if gender == "Male" else 0],
            'Age': [age],
            'Academic Pressure': [academic_pressure],
            'Study Satisfaction': [study_satisfaction],
            'Work/Study Hours': [work_study_hours],
            'Financial Stress': [financial_stress]
        })
        user_input_scaled = prepare_user_input(user_input)

        # Model training and predictions
        with st.spinner('Training models and making predictions...'):
            # Logistic Regression
            log_reg = LogisticRegression(solver='saga', C=0.01, max_iter=2000, random_state=42)
            log_reg.fit(X_train, y_train)
            log_reg_pred = log_reg.predict(user_input_scaled)

            # Decision Tree
            decision_tree = DecisionTreeClassifier(max_depth=50, min_samples_split=10, 
                                                 min_samples_leaf=30, random_state=42)
            decision_tree.fit(X_train, y_train)
            dt_pred = decision_tree.predict(user_input_scaled)

            # KNN
            knn = KNeighborsClassifier(weights='uniform', n_neighbors=50, metric='manhattan')
            knn.fit(X_train, y_train)
            knn_pred = knn.predict(user_input_scaled)

        # Display predictions
        st.subheader("Model Predictions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Logistic Regression", 
                     "Depressed" if log_reg_pred[0] == 1 else "Not Depressed")
        with col2:
            st.metric("Decision Tree", 
                     "Depressed" if dt_pred[0] == 1 else "Not Depressed")
        with col3:
            st.metric("KNN", 
                     "Depressed" if knn_pred[0] == 1 else "Not Depressed")

        # Model evaluation metrics in expandable section
        with st.expander("View Model Evaluation Metrics"):
            for model_name, model in [("Logistic Regression", log_reg), 
                                    ("Decision Tree", decision_tree), 
                                    ("KNN", knn)]:
                st.subheader(model_name)
                y_pred = model.predict(X_test)
                st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.write("Confusion Matrix:")
                st.write(confusion_matrix(y_test, y_pred))
                st.write("Classification Report:")
                st.text(classification_report(y_test, y_pred))

    else:  # CGPA Prediction
        st.sidebar.header("Input Features for CGPA Prediction")
        
        # User input collection
        study_hours = st.sidebar.slider("Study Hours", min_value=1, max_value=24, value=6,
                                      help="Number of hours spent studying per day")
        academic_pressure = st.sidebar.slider("Academic Pressure", min_value=1, max_value=5, value=3,
                                           help="Level of academic pressure (1: Very Low, 5: Very High)")
        financial_stress = st.sidebar.slider("Financial Stress", min_value=1, max_value=5, value=3,
                                          help="Level of financial stress (1: Very Low, 5: Very High)")
        gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
        study_satisfaction = st.sidebar.slider("Study Satisfaction", min_value=1, max_value=5, value=3,
                                            help="Level of satisfaction with studies (1: Very Low, 5: Very High)")

        # Load and prepare CGPA data
        data_lr = load_cgpa_data()
        X_lr = data_lr[['Study Hours', 'Academic Pressure', 
                        'Financial Stress', 'Gender', 'Study Satisfaction']]  # Removed Age
        y_lr = data_lr['CGPA']
        X_lr_scaled = st.session_state.scaler_lr.fit_transform(X_lr)

        # Prepare user input
        user_input_cgpa = pd.DataFrame({
            'Study Hours': [study_hours],
            'Academic Pressure': [academic_pressure],
            'Financial Stress': [financial_stress],
            'Gender': [1 if gender == "Male" else 0],
            'Study Satisfaction': [study_satisfaction]
        })
        user_input_cgpa_scaled = st.session_state.scaler_lr.transform(user_input_cgpa)

        # Train and predict
        with st.spinner('Calculating CGPA prediction...'):
            lr_model = LinearRegression()
            lr_model.fit(X_lr_scaled, y_lr)
            cgpa_prediction = lr_model.predict(user_input_cgpa_scaled)
            bounded_cgpa = bound_cgpa(cgpa_prediction[0])

        # Display prediction
        st.subheader("CGPA Prediction")
        st.metric("Predicted CGPA", f"{bounded_cgpa:.2f}")
        
        # Add interpretation
        st.info("""
        Note: The predicted CGPA is bounded between 0 and 10, which is the standard CGPA scale.
        - Higher study hours and study satisfaction generally correlate with higher CGPA
        - Academic pressure and financial stress may negatively impact CGPA
        """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please check your input data and try again.")