# streamlit_app.py

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

st.title("🚢 Titanic Survival Prediction App")

# Upload train and test data
train_file = st.file_uploader("Upload Titanic Train CSV", type="csv")
test_file = st.file_uploader("Upload Titanic Test CSV", type="csv")

if train_file is not None and test_file is not None:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Preprocessing
    def preprocess(df):
        df = df.copy()
        df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
        df['Embarked'].fillna('S', inplace=True)
        df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        return df

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    # Features and labels
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = train_df[features]
    y = train_df['Survived']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_val)
    st.subheader("📊 Model Evaluation")
    st.write(f"**Accuracy:** {accuracy_score(y_val, y_pred):.2f}")
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_val, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_val, y_pred))

    # Predict on test set
    st.subheader("🧪 Predict on Test Set")
    test_predictions = model.predict(test_df[features])
    test_df['Survived_Prediction'] = test_predictions
    st.write(test_df[['PassengerId', 'Survived_Prediction']].head())

    # Download prediction
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=test_df[['PassengerId', 'Survived_Prediction']].to_csv(index=False),
        file_name='titanic_predictions.csv',
        mime='text/csv'
    )
