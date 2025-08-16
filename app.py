import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, RocCurveDisplay
import streamlit as st
import joblib
import os


def load_and_preprocess_data(train_path, test_path):
    """
    Loads the Titanic datasets, combines them for consistent preprocessing,
    and performs feature engineering and cleaning.
    """
    print("--- Loading and Combining Datasets ---")
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except FileNotFoundError:
        print("Error: Make sure 'Titanic_train.csv' and 'Titanic_test.csv' are in the same directory.")
        return None, None

    
    train_df['source'] = 'train'
    test_df['source'] = 'test'
    test_df['Survived'] = np.nan

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print("\n--- Initial Data Exploration ---")
    print("Combined dataset info:")
    combined_df.info()

    print("\n--- Handling Missing Values ---")
    print(combined_df.isnull().sum())

    median_age = combined_df['Age'].median()
    combined_df['Age'] = combined_df['Age'].fillna(median_age)

    median_fare = combined_df['Fare'].median()
    combined_df['Fare'] = combined_df['Fare'].fillna(median_fare)

    mode_embarked = combined_df['Embarked'].mode()[0]
    combined_df['Embarked'] = combined_df['Embarked'].fillna(mode_embarked)

    combined_df.drop(['PassengerId', 'Cabin', 'Name', 'Ticket'], axis=1, inplace=True)
    
    print("\n--- Feature Engineering ---")
    combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1
    
    print("\n--- Encoding Categorical Variables ---")
    combined_df = pd.get_dummies(combined_df, columns=['Sex', 'Embarked'], drop_first=True)
    
    combined_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)
    
    train_processed = combined_df[combined_df['source'] == 'train'].drop('source', axis=1)
    test_processed = combined_df[combined_df['source'] == 'test'].drop(['source', 'Survived'], axis=1)

    print("\n--- Preprocessing Complete ---")
    print("Processed training data shape:", train_processed.shape)
    print("Processed test data shape:", test_processed.shape)
    
    return train_processed, test_processed


def train_and_evaluate_model(train_df):
    """
    Builds and trains a logistic regression model, evaluates it, and saves it.
    """
    print("\n--- Model Building and Evaluation ---")
    X = train_df.drop('Survived', axis=1)
    y = train_df['Survived']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print("\n--- Model Evaluation on Validation Data ---")
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    
    print("\n--- Visualizing ROC Curve ---")
    RocCurveDisplay.from_estimator(model, X_val, y_val)
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()

    print("\n--- Interpreting Model Coefficients ---")
    coefficients = pd.DataFrame({'feature': X.columns, 'coefficient': model.coef_[0]})
    print(coefficients.sort_values(by='coefficient', ascending=False))
    print("\nInterpretation:")
    print("- A positive coefficient means that an increase in that feature's value increases the odds of survival.")
    print("- A negative coefficient means that an increase in that feature's value decreases the odds of survival.")
    print("- 'Sex_male' having a large negative coefficient suggests males were less likely to survive.")
    print("- 'Pclass' having a negative coefficient means higher class (lower Pclass value) increases survival odds, which makes sense.")

    model_filename = 'titanic_model.pkl'
    joblib.dump(model, model_filename)
    print(f"\nModel saved as '{model_filename}'")
    
    return model, X_train.columns


def run_streamlit_app(model, feature_names):
    """
    Builds the Streamlit user interface and prediction logic.
    """
    st.title('🚢 Titanic Survival Predictor')
    st.write('Enter passenger information below to predict their survival probability.')
    
    st.sidebar.header('Passenger Details')
    
    pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3], index=2)
    age = st.sidebar.slider('Age', 0, 100, 25)
    family_size = st.sidebar.slider('Family Size', 1, 11, 1)
    fare = st.sidebar.slider('Fare', 0.0, 512.0, 32.0)
    sex = st.sidebar.selectbox('Sex', ['female', 'male'])
    embarked = st.sidebar.selectbox('Port of Embarkation', ['S', 'C', 'Q'])
    
    user_data = {
        'Pclass': pclass,
        'Age': age,
        'Fare': fare,
        'FamilySize': family_size,
        'Sex_male': 1 if sex == 'male' else 0,
        'Embarked_Q': 1 if embarked == 'Q' else 0,
        'Embarked_S': 1 if embarked == 'S' else 0
    }
    
    user_df = pd.DataFrame([user_data], columns=feature_names)

    if st.sidebar.button('Predict Survival'):
        prediction_prob = model.predict_proba(user_df)[:, 1]
        prediction = model.predict(user_df)[0]
        
        st.subheader('Prediction Result')
        
        if prediction == 1:
            st.success(f"Prediction: Survived! 🎉")
        else:
            st.error(f"Prediction: Did not survive.  ")
        
        st.write(f"Survival Probability: **{prediction_prob[0]:.2f}**")
        st.write('---')
        st.markdown(
            """
            *Disclaimer: This is a simplified model for demonstration. 
            Survival was influenced by many complex factors.*
            """
        )

if __name__ == '__main__':
    train_file = 'Titanic_train.csv'
    test_file = 'Titanic_test.csv'
    
    model_file = 'titanic_model.pkl'

    
    if os.path.exists(model_file):
        print("Existing model file found. Deleting it to ensure a fresh training run.")
        os.remove(model_file)
    
    print("Model file not found. Starting training process...")
    
    processed_train_df, processed_test_df = load_and_preprocess_data(train_file, test_file)
    
    if processed_train_df is not None:
        trained_model, features = train_and_evaluate_model(processed_train_df)
    
    run_streamlit_app(trained_model, features)
 