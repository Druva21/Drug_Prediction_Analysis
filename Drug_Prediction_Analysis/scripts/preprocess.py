import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(file_path='data/Drug_Prediction_Dataset.csv'):
    # Load dataset
    df = pd.read_csv(file_path)

    # Handle missing values
    df = df.copy()
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Na_to_K'] = df['Na_to_K'].fillna(df['Na_to_K'].mean())
    df['Sex'] = df['Sex'].fillna(df['Sex'].mode()[0])
    df['BP'] = df['BP'].fillna(df['BP'].mode()[0])
    df['Cholesterol'] = df['Cholesterol'].fillna(df['Cholesterol'].mode()[0])
    df['Drug'] = df['Drug'].fillna(df['Drug'].mode()[0])

    # Encoding
    sex_mapping = {'F': 0, 'M': 1}
    bp_mapping = {'HIGH': 0, 'LOW': 1, 'NORMAL': 2}
    cholesterol_mapping = {'HIGH': 0, 'NORMAL': 1}
    drug_mapping = {'DrugY': 0, 'drugA': 1, 'drugB': 2, 'drugC': 3, 'drugX': 4}

    df['Sex'] = df['Sex'].map(sex_mapping)
    df['BP'] = df['BP'].map(bp_mapping)
    df['Cholesterol'] = df['Cholesterol'].map(cholesterol_mapping)
    df['Drug'] = df['Drug'].map(drug_mapping)

    # Split features/target
    X = df.drop(columns=['Drug'])
    y = df['Drug']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train[['Age', 'Na_to_K']] = scaler.fit_transform(X_train[['Age','Na_to_K']])
    X_test[['Age', 'Na_to_K']] = scaler.transform(X_test[['Age','Na_to_K']])

    return X_train, X_test, y_train, y_test, scaler
