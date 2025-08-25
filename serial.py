import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
import joblib


def load_data(file_path):
    """
    Load dataset from CSV file

    Args:
        file_path: Path to the CSV file

    Returns:
        pandas DataFrame containing the dataset
    """
    print(f"Loading data from {file_path}...")
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def identify_column_types(df):
    """
    Identify numeric and categorical columns

    Args:
        df: pandas DataFrame

    Returns:
        tuple of (numeric_columns, categorical_columns)
    """
    # Assuming the target column is named 'target' - adjust if needed
    feature_cols = [col for col in df.columns if col != 'target']

    numeric_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Numeric features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")

    return numeric_cols, categorical_cols


def create_preprocessing_pipeline(numeric_cols, categorical_cols):
    """
    Create a preprocessing pipeline for numeric and categorical features

    Args:
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names

    Returns:
        ColumnTransformer preprocessing pipeline
    """
    # Numeric features pipeline: impute missing values and scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical features pipeline: impute missing values and one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # Drop any columns not specified
    )

    return preprocessor


def build_and_train_model(X_train, y_train, preprocessor):
    """
    Build and train a logistic regression model without parallel processing

    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: Data preprocessing pipeline

    Returns:
        Trained pipeline
    """
    # Create the full pipeline with preprocessing and model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Define hyperparameters for grid search
    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__solver': ['liblinear', 'saga'],
        'classifier__penalty': ['l1', 'l2']
    }

    # Use GridSearchCV without parallel processing
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring='f1',
        verbose=1
    )

    print("\nTraining model with GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target

    Returns:
        Dictionary with evaluation metrics
    """
    print("\nEvaluating model on test data...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print evaluation results
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'f1_score': f1
    }


def main(data_path='data.csv'):
    """
    Main function to run the entire pipeline

    Args:
        data_path: Path to the CSV file
    """
    start_time = time.time()

    # Load data
    data = load_data(data_path)

    # Assuming the target column is named 'target'
    target_col = 'target'
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(data)

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numeric_cols, categorical_cols)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    # Build and train the model
    model = build_and_train_model(X_train, y_train, preprocessor)

    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)

    # Calculate and print the total execution time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

    # Save the model (optional)
    # joblib.dump(model, 'logistic_regression_model.joblib')

    return metrics


if __name__ == "__main__":
    # You can change the data path if needed
    main(data_path='pdc_dataset_with_target.csv')