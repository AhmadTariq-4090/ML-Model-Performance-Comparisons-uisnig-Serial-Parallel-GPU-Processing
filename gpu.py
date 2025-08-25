import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

# Enable mixed precision for faster computation on modern GPUs
mixed_precision.set_global_policy('mixed_float16')

# GPU check
def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU available: {len(gpus)} device(s)")
    else:
        print("⚠️ No GPU found. Using CPU.")
    return bool(gpus)

# Run GPU check at start
check_gpu()
print(f"TensorFlow {tf.__version__}, Policy: {mixed_precision.global_policy()}\n")

def load_and_preprocess(file_path):
    data = pd.read_csv(file_path)
    target_col = 'target'
    X = data.drop(columns=[target_col])
    y = data[target_col].values

    # Identify column types
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build preprocessing pipelines
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preproc = ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, categorical_cols)
    ], remainder='drop')

    # Fit-transform and split
    X_processed = preproc.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y)

    return (X_train, y_train), (X_test, y_test)


def make_tf_dataset(X, y, batch_size=256, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    return ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)


def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(1, activation='sigmoid', dtype='float32')  # Logistic regression
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
        jit_compile=True
    )
    return model


def main(data_path='data.csv'):
    start = time.time()

    # Load & preprocess
    (X_train, y_train), (X_test, y_test) = load_and_preprocess(data_path)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Create tf.data datasets
    train_ds = make_tf_dataset(X_train, y_train, shuffle=True)
    test_ds = make_tf_dataset(X_test, y_test)

    # Build & train
    model = build_model(X_train.shape[1])
    es = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    train_time_start = time.time()
    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=50,
        callbacks=[es],
        verbose=1
    )
    train_time = time.time() - train_time_start

    # Evaluate
    eval_start = time.time()
    loss, acc = model.evaluate(test_ds, verbose=0)
    eval_time = time.time() - eval_start
    y_pred = (model.predict(test_ds) > 0.5).astype(int).flatten()

    # Compute metrics
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nResults:\n Accuracy: {acc:.4f}\n F1 Score: {f1:.4f}")
    print("Confusion Matrix :", cm)
    print(f"Training Time: {train_time:.2f}s, Evaluation Time: {eval_time:.2f}s")
    print(f"Total Elapsed: {time.time() - start:.2f}s")

    return {'accuracy': acc, 'f1_score': f1, 'confusion_matrix': cm}


if __name__ == '__main__':
    main('pdc_dataset_with_target.csv')
