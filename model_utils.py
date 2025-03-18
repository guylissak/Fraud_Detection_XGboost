""" Model Utils Module
modeling functions and model evaluations utils such as: splitting train test 
validation,model training, metrics evaluation, metrics logging to MLflow, scale
& encode & PCA functionalities. """
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import xgboost as xgb
from datetime import datetime, timezone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, classification_report, \
 roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.decomposition import PCA


def impute_missing_values(X_train: pd.DataFrame,
                          X_val: pd.DataFrame,
                          X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values:
      - Numeric features are imputed with the mean.
      - Categorical features are imputed with the most frequent value.

    Numeric features are selected based on dtypes (int8, int16, int32, int64, float32, float64)
    excluding 'TransactionID' and 'isFraud'.
    """
    # Identify numeric features based on specific dtypes
    numeric_features = X_train.select_dtypes(include=['int8','int16','int32','int64','float32','float64']).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in ['TransactionID', 'isFraud']]

    # Identify categorical features: all columns not in numeric_features and not in ['TransactionID', 'isFraud']
    categorical_features = [col for col in X_train.columns if col not in numeric_features and col not in ['TransactionID', 'isFraud']]

    # Impute numeric features with mean
    imputer_numeric = SimpleImputer(strategy="mean")
    X_train_numeric = pd.DataFrame(imputer_numeric.fit_transform(X_train[numeric_features]),
                                   columns=numeric_features, index=X_train.index)
    X_val_numeric   = pd.DataFrame(imputer_numeric.transform(X_val[numeric_features]),
                                   columns=numeric_features, index=X_val.index)
    X_test_numeric  = pd.DataFrame(imputer_numeric.transform(X_test[numeric_features]),
                                   columns=numeric_features, index=X_test.index)

    # Impute categorical features with the most frequent value
    imputer_categorical = SimpleImputer(strategy="most_frequent")
    X_train_cat = pd.DataFrame(imputer_categorical.fit_transform(X_train[categorical_features]),
                               columns=categorical_features, index=X_train.index)
    X_val_cat   = pd.DataFrame(imputer_categorical.transform(X_val[categorical_features]),
                               columns=categorical_features, index=X_val.index)
    X_test_cat  = pd.DataFrame(imputer_categorical.transform(X_test[categorical_features]),
                               columns=categorical_features, index=X_test.index)

    # Combine the imputed numeric and categorical features, preserving the original column order
    X_train_imputed = X_train.copy()
    X_val_imputed = X_val.copy()
    X_test_imputed = X_test.copy()

    for col in numeric_features:
        X_train_imputed[col] = X_train_numeric[col]
        X_val_imputed[col] = X_val_numeric[col]
        X_test_imputed[col] = X_test_numeric[col]

    for col in categorical_features:
        X_train_imputed[col] = X_train_cat[col]
        X_val_imputed[col] = X_val_cat[col]
        X_test_imputed[col] = X_test_cat[col]

    return X_train_imputed, X_val_imputed, X_test_imputed


def prepare_data_for_training(train: pd.DataFrame, test: pd.DataFrame, impute = False):
    """Prepare final datasets for model training."""
    X = train.drop(columns=['isFraud', 'TransactionID'])
    y = train['isFraud']
    X_test = test.drop(columns=['TransactionID'], errors='ignore')

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Handle missing values
    if impute:
       X_train, X_val, X_test = impute_missing_values(X_train, X_val, X_test)

    return X_train, X_val, y_train, y_val, X_test

def prepare_time_based_data_for_training(train: pd.DataFrame, test: pd.DataFrame, split_ratio=0.8):
    """
    Prepare final datasets for model training using a time-based split.
    """
    # Define split index (chronological split)
    split_index = int(len(train) * split_ratio)

    # Train and validation split based on time
    X_train, y_train = train.iloc[:split_index].drop(columns=["isFraud", "TransactionID"]), train.iloc[:split_index]["isFraud"]
    X_val, y_val = train.iloc[split_index:].drop(columns=["isFraud", "TransactionID"]), train.iloc[split_index:]["isFraud"]

    X_test = test.drop(columns=["TransactionID"], errors="ignore")

    # Handle missing values
    X_train, X_val, X_test = impute_missing_values(X_train, X_val, X_test)

    return X_train, X_val, y_train, y_val, X_test


def get_model_params():
    """Return the best model parameters."""
    return {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': 800,
        "learning_rate": 0.08,
        "max_depth": 8,
        "min_child_weight": 5,
        "gamma": 0,
        "subsample": 0.6,
        "colsample_bytree": 0.8,
        "lambda": 10,
        "alpha": 0
    }


def scale_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """ Scale the features using StandardScaler, fitting only on X_train to avoid data leakage. """

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to preserve column names & indices
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_val_scaled, X_test_scaled

def plot_pca_variance(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
    """Plots the cumulative explained variance to find the optimal number of PCA components."""

    top_features = ['V258', 'V257', 'V201', 'V156', 'V91', 'V294']

    # Get all V feature columns
    all_v_columns = [col for col in X_train.columns if col.startswith('V')]

    # Separate top V features from others
    v_to_pca = [col for col in all_v_columns if col not in top_features]

    # Extract V features for PCA
    train_v = X_train[v_to_pca]
    val_v = X_val[v_to_pca]
    test_v = X_test[v_to_pca]

    # Scale the V features using the helper function
    train_v_scaled, val_v_scaled, test_v_scaled = scale_data(train_v, val_v, test_v)

    pca = PCA()
    pca.fit(train_v_scaled)

    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance vs. Number of Components")
    plt.axhline(y=0.95, color='r', linestyle='--', label="95% Variance Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

def reduce_v_features_selectively(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                                  n_components: int, top_features: Optional[List[str]]=None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Selectively reduce V features with PCA while keeping top important features.
    Supports train, validation, and test datasets.
    """
    if top_features is None:
        top_features = ['V258', 'V257', 'V201', 'V156', 'V91', 'V294']

    # Get all V feature columns
    all_v_columns = [col for col in X_train.columns if col.startswith('V')]

    # Separate top V features from others
    v_to_pca = [col for col in all_v_columns if col not in top_features]

    print(f"Keeping {len(top_features)} top V features directly")
    print(f"Applying PCA to {len(v_to_pca)} remaining V features")

    # Extract V features for PCA
    train_v = X_train[v_to_pca]
    val_v = X_val[v_to_pca]
    test_v = X_test[v_to_pca]

    # Scale the V features using the helper function
    train_v_scaled, val_v_scaled, test_v_scaled = scale_data(train_v, val_v, test_v)

    # Apply PCA - fit only on training data to prevent data leakage
    pca = PCA(n_components=n_components)
    train_v_pca = pca.fit_transform(train_v_scaled)
    val_v_pca = pca.transform(val_v_scaled)
    test_v_pca = pca.transform(test_v_scaled)

    # Create new column names
    pca_columns = [f'V_PCA_{i+1}' for i in range(n_components)]

    # Convert to DataFrames
    train_v_pca_df = pd.DataFrame(train_v_pca, columns=pca_columns, index=X_train.index)
    val_v_pca_df = pd.DataFrame(val_v_pca, columns=pca_columns, index=X_val.index)
    test_v_pca_df = pd.DataFrame(test_v_pca, columns=pca_columns, index=X_test.index)

    # Drop PCA-transformed V features (keep the top ones)
    train_no_pca_v = X_train.drop(columns=v_to_pca)
    val_no_pca_v = X_val.drop(columns=v_to_pca)
    test_no_pca_v = X_test.drop(columns=v_to_pca)

    # Combine with PCA features
    train_pca = pd.concat([train_no_pca_v, train_v_pca_df], axis=1)
    val_pca = pd.concat([val_no_pca_v, val_v_pca_df], axis=1)
    test_pca = pd.concat([test_no_pca_v, test_v_pca_df], axis=1)

    print(f"Final feature count: {train_pca.shape[1]} (original: {X_train.shape[1]})")
    print(f"PCA variance explained: {sum(pca.explained_variance_ratio_):.4f}")

    return train_pca, val_pca, test_pca

def train_model(X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.DataFrame,
                y_val: pd.DataFrame, model_params: Dict) -> xgb.XGBClassifier:
    """Train and return the XGBoost model."""
    model = xgb.XGBClassifier(**model_params, use_label_encoder=False, early_stopping_rounds=100, tree_method="gpu_hist",
    predictor="gpu_predictor", random_state=42)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )
    return model

def evaluate_model(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.ndarray, float]:
    """Evaluate model and return validation predictions and score."""
    predictions = model.predict_proba(X)[:, 1]
    auc_score = roc_auc_score(y, predictions)
    return predictions, auc_score

def export_test_predictions(test_predictions: np.ndarray, test: pd.DataFrame, prefix="baseline"):
    """ Export test predictions to CSV."""
    # Get current UTC time
    utc_time = datetime.now(timezone.utc)
    utc_time = utc_time.strftime("%Y-%m-%d %H:%M:%S")

    # Save test predictions
    test_df = pd.DataFrame({'TransactionID': test['TransactionID'], 'isFraud': test_predictions})
    test_df.to_csv(f"{prefix}_submission_{utc_time}.csv", index=False)

def plot_metric_curve(model: xgb.XGBClassifier, display = False, metric='auc'):
    """
    Plot training and validation loss (RMSE) curves.
    """
    eval_results = model.evals_result()
    plt.figure(figsize=(10, 6))
    epochs = range(len(eval_results["validation_0"][metric]))
    plt.plot(epochs, eval_results["validation_0"][metric], label="Train AUC")
    if "validation_1" in eval_results:
        plt.plot(epochs, eval_results["validation_1"][metric], label="Validation AUC")
    plt.title("Training and Validation Loss (AUC)")
    plt.xlabel("Boosting Rounds")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plot_path = "auc_plot.png"
    plt.savefig(plot_path)
    if display:
        plt.show()
    else:
      plt.close()

def plot_validation_roc_auc_curve(val_pred: np.ndarray, y_val: np.ndarray, display=False):
    """
    Plot ROC curve for validation data.
    """
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_val, val_pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plot_path = "roc_auc_validation_plot.png"
    plt.savefig(plot_path)
    if display:
        plt.show()
    else:
      plt.close()

def plot_confusion_matrix(y_true, y_pred, display=False, threshold=0.5):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred > threshold)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    if display:
        plt.show()
    else:
      plt.close()

def plot_feature_importance(model: xgb.XGBClassifier, metric="gain", display=False):
  feature_importance = model.get_booster().get_score(importance_type=metric)

  # Convert to DataFrame
  importance_df = pd.DataFrame(feature_importance.items(), columns=["Feature", "Importance"])

  # Sort in descending order
  importance_df = importance_df.sort_values(by="Importance", ascending=False).head(30)

  # Plot using Seaborn
  plt.figure(figsize=(10, 6))
  sns.barplot(x="Importance", y="Feature", data=importance_df, palette="pastel")
  plt.title(f"Top 30 Feature Importances {metric.upper()}")
  plt.savefig(f"feature_importance_{metric}.png")
  if display:
      plt.show()
  else:
    plt.close()


def log_mlflow_artifacts(model: xgb.XGBClassifier, model_params: Dict, train_auc: float, val_auc: float,
                         val_pred: np.ndarray, y_val: np.ndarray, model_name="baseline", log_model=False):
    """Log model artifacts to MLflow."""
    mlflow.log_params(model_params)
    evals_results = model.evals_result()
    for i, metric in enumerate(evals_results["validation_0"]["auc"]):
        mlflow.log_metric("train_auc", metric, step=i)

    for i, metric in enumerate(evals_results["validation_1"]["auc"]):
        mlflow.log_metric("validation_auc", metric, step=i)

    mlflow.log_metric("train_roc_auc", train_auc)
    mlflow.log_metric("validation_roc_auc", val_auc)
    mlflow.log_metric("overfitt_diff", train_auc - val_auc)

    plot_metric_curve(model, True)
    plot_validation_roc_auc_curve(val_pred, y_val, True)
    mlflow.log_artifact('auc_plot.png')
    mlflow.log_artifact('roc_auc_validation_plot.png')

    plot_confusion_matrix(y_val, val_pred, True)
    mlflow.log_artifact('confusion_matrix.png')

    plot_feature_importance(model, "gain", True)
    mlflow.log_artifact('feature_importance_gain.png')

    report_str = classification_report(y_val, val_pred > 0.5)
    filename = "classification_report.txt"
    with open(filename, "w") as f:
        f.write(report_str)

    mlflow.log_artifact(filename)

    if log_model:
        mlflow.xgboost.log_model(model, model_name)
