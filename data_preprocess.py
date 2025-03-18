""" Data Prepreocess Module
Data preprocess module - utils functions for feature engineering,
data cleaning & normalization, handle missing values, etc.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

def get_top_missing_cols(df: pd.DataFrame ,n: int=10,thresh: int =80) -> pd.DataFrame:
    """
    Returns missing columns in dataframe with missing values percent > thresh
    if n=None. It will gave whole dataframe with missing values percent > thresh
    """

    dff = (df.isnull().sum()/df.shape[0])*100
    dff = dff.reset_index()
    dff.columns = ['col','missing_percent']
    dff = dff.sort_values(by=['missing_percent'],ascending=False).reset_index(drop=True)
    print(f'There are {df.isnull().any().sum()} columns in this dataset with missing values.')
    print(f'There are {dff[dff["missing_percent"] > thresh].shape[0]} columns with missing percent values than {thresh}%')
    if n:
        return dff.head(n)
    else:
        return dff

def drop_high_missing_columns(train: pd.DataFrame, test: pd.DataFrame,
                              missing_train_df: pd.DataFrame, missing_test_df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Drop columns with high missing value percentages.
    """
    train = train.drop(columns=missing_train_df['col'].tolist()[:-1])
    test = test.drop(columns=missing_test_df['col'].tolist()[:-1])

    # Drop additional problematic columns
    train.drop(columns='V107', inplace=True)
    test.drop(columns='V107', inplace=True)

    return train, test

def create_time_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Create day and hour features from TransactionDT.
    """

    def apply_fraud_hour_feature(hr):
      if hr >= 7 and hr < 10:
          return "HighestFraudProb"
      if hr >= 14 and hr < 16:
          return "LowestFraudProb"
      if (hr >= 4 and hr < 7) or (hr >= 10 and hr < 14):
          return "MediumFraudProb"
      else:
          return "LowFraudProb"

    # Create day of week feature (1-7)
    train['day'] = ((train['TransactionDT'] // (3600*24) - 1) % 7) + 1
    test['day'] = ((test['TransactionDT'] // (3600*24) - 1) % 7) + 1

    # Create hour of day feature (1-24)
    train['hour'] = ((train['TransactionDT'] // 3600) % 24) + 1
    test['hour'] = ((test['TransactionDT'] // 3600) % 24) + 1

    # Create hour fraud probability category
    train['hour_fraud_prob'] = train['hour'].apply(apply_fraud_hour_feature)
    test['hour_fraud_prob'] = test['hour'].apply(apply_fraud_hour_feature)

    return train, test

def create_transaction_amount_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Clean and transform TransactionAmt feature.
    """
    # Remove high transaction amount outliers
    train.drop(train[train.TransactionAmt > 10000].index, axis=0, inplace=True)

    # Create log transformed amount feature
    train['LogTransactionAmt'] = np.log(train['TransactionAmt'])
    test['LogTransactionAmt'] = np.log(test['TransactionAmt'])

    return train, test

def create_card_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Create features based on card information.
    """

    def apply_card_to_fraud_high_prob(row):
        if row == np.nan:
            return row
        else:
            if float(row) > 160:
                return 'True'
            else:
                return 'False'

    train['prob_card3'] = train['card3'].apply(apply_card_to_fraud_high_prob)
    test['prob_card3'] = test['card3'].apply(apply_card_to_fraud_high_prob)

    return train, test

def create_product_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Create features based on product information.
    """

    def apply_product_fraud_risk(product_cd):
      if product_cd == np.nan:
          return product_cd
      else:
          if product_cd == 'C':
              return "True"
          else:
              return "False"

    train['product_fraud_risk'] = train['ProductCD'].apply(apply_product_fraud_risk)
    test['product_fraud_risk'] = test['ProductCD'].apply(apply_product_fraud_risk)

    return train, test


def process_email_domains(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Clean and categorize email domains.
    """
    for df in [train, test]:
        # Process purchaser email domains (P_emaildomain)
        # Group Google domains
        df.loc[df['P_emaildomain'].isin(['gmail.com', 'gmail']), 'P_emaildomain'] = 'Google'

        # Group Yahoo domains
        df.loc[df['P_emaildomain'].isin([
            'yahoo.com', 'yahoo.com.mx', 'yahoo.co.uk', 'yahoo.co.jp',
            'yahoo.de', 'yahoo.fr', 'yahoo.es'
        ]), 'P_emaildomain'] = 'Yahoo'

        # Group Microsoft domains
        df.loc[df['P_emaildomain'].isin([
            'hotmail.com', 'outlook.com', 'msn.com', 'live.com.mx', 'hotmail.es',
            'hotmail.co.uk', 'hotmail.de', 'outlook.es', 'live.com', 'live.fr',
            'hotmail.fr'
        ]), 'P_emaildomain'] = 'Microsoft'

        # Group rare domains
        df.loc[df.P_emaildomain.isin(
            df.P_emaildomain.value_counts()[df.P_emaildomain.value_counts() <= 500].index
        ), 'P_emaildomain'] = "Others"

        # Fill missing values
        df.P_emaildomain.fillna("Unknown", inplace=True)

        # Process receiver email domains (R_emaildomain)
        # Group Google domains
        df.loc[df['R_emaildomain'].isin(['gmail.com', 'gmail']), 'R_emaildomain'] = 'Google'

        # Group Yahoo domains
        df.loc[df['R_emaildomain'].isin([
            'yahoo.com', 'yahoo.com.mx', 'yahoo.co.uk', 'yahoo.co.jp',
            'yahoo.de', 'yahoo.fr', 'yahoo.es'
        ]), 'R_emaildomain'] = 'Yahoo'

        # Group Microsoft domains
        df.loc[df['R_emaildomain'].isin([
            'hotmail.com', 'outlook.com', 'msn.com', 'live.com.mx', 'hotmail.es',
            'hotmail.co.uk', 'hotmail.de', 'outlook.es', 'live.com', 'live.fr',
            'hotmail.fr'
        ]), 'R_emaildomain'] = 'Microsoft'

        # Group rare domains
        df.loc[df.R_emaildomain.isin(
            df.R_emaildomain.value_counts()[df.R_emaildomain.value_counts() <= 300].index
        ), 'R_emaildomain'] = "Others"

        # Fill missing values
        df.R_emaildomain.fillna("Unknown", inplace=True)

    return train, test

def remove_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers from numerical C features.
    """
    train.drop(train[train.C1 > 3000].index, axis=0, inplace=True)
    train.drop(train[train.C2 > 3000].index, axis=0, inplace=True)
    train.drop(train[train.C7 > 1400].index, axis=0, inplace=True)
    train.drop(train[train.C4 > 1400].index, axis=0, inplace=True)
    train.drop(train[train.C6 > 1400].index, axis=0, inplace=True)
    train.drop(train[train.C7 > 1400].index, axis=0, inplace=True)
    train.drop(train[train.C8 > 1000].index, axis=0, inplace=True)
    train.drop(train[train.C10 > 1000].index, axis=0, inplace=True)
    train.drop(train[train.C11 > 2000].index, axis=0, inplace=True)
    train.drop(train[train.C12 > 2000].index, axis=0, inplace=True)
    train.drop(train[train.C13 > 1200].index, axis=0, inplace=True)
    train.drop(train[train.C14 > 800].index, axis=0, inplace=True)
    # train.drop(train[train.dist1>6000].index, axis=0, inplace=True)
    # train.drop(train[train.dist2>8000].index, axis=0, inplace=True)

    return train

def transform_device_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform DeviceInfo into standardized device corporation categories.
    """
    df['DeviceCorp'] = df['DeviceInfo']

    # Brand mappings
    df.loc[df['DeviceInfo'].str.contains('HUAWEI|HONOR', case=False, na=False, regex=True), 'DeviceCorp'] = 'HUAWEI'
    df.loc[df['DeviceInfo'].str.contains('OS', na=False, regex=False), 'DeviceCorp'] = 'APPLE'
    df.loc[df['DeviceInfo'].str.contains('Idea|TA', case=False, na=False), 'DeviceCorp'] = 'Lenovo'
    df.loc[df['DeviceInfo'].str.contains('Moto|XT|Edison', case=False, na=False), 'DeviceCorp'] = 'Moto'
    df.loc[df['DeviceInfo'].str.contains('MI|Mi|Redmi', na=False), 'DeviceCorp'] = 'Mi'
    df.loc[df['DeviceInfo'].str.contains('VS|LG|EGO', na=False), 'DeviceCorp'] = 'LG'
    df.loc[df['DeviceInfo'].str.contains('ONE TOUCH|ALCATEL', case=False, na=False, regex=False), 'DeviceCorp'] = 'ALCATEL'
    df.loc[df['DeviceInfo'].str.contains('ONE A', na=False, regex=False), 'DeviceCorp'] = 'ONEPLUS'
    df.loc[df['DeviceInfo'].str.contains('OPR6', na=False, regex=False), 'DeviceCorp'] = 'HTC'
    df.loc[df['DeviceInfo'].str.contains('Nexus|Pixel', case=False, na=False, regex=True), 'DeviceCorp'] = 'google'
    df.loc[df['DeviceInfo'].str.contains('STV', na=False, regex=False), 'DeviceCorp'] = 'blackberry'
    df.loc[df['DeviceInfo'].str.contains('ASUS', case=False, na=False, regex=False), 'DeviceCorp'] = 'ASUS'
    df.loc[df['DeviceInfo'].str.contains('BLADE', case=False, na=False, regex=False), 'DeviceCorp'] = 'ZTE'

    # Extract first part of device info as fallback
    df['DeviceCorp'] = df['DeviceInfo'].astype('str').str.split(':', expand=True)[0].\
                            str.split('-', expand=True)[0].str.split(expand=True)[0]

    # Additional specific mappings
    df.loc[df['DeviceInfo'].isin(['rv', 'SM', 'GT', 'SGH']), 'DeviceCorp'] = 'SAMSUNG'
    df.loc[df['DeviceInfo'].str.startswith('Z', na=False), 'DeviceCorp'] = 'ZTE'
    df.loc[df['DeviceInfo'].str.startswith('KF', na=False), 'DeviceCorp'] = 'Amazon'

    # Sony device mappings
    for i in ['D', 'E', 'F', 'G']:
        df.loc[df['DeviceInfo'].str.startswith(i, na=False), 'DeviceCorp'] = 'SONY'

    # Group rare device corporations
    df.loc[df['DeviceCorp'].isin(df['DeviceCorp'].value_counts()\
                               [df['DeviceCorp'].value_counts() < 100].index), 'DeviceCorp'] = 'Other'

    # Standardize to uppercase
    df['DeviceCorp'] = df['DeviceCorp'].str.upper()

    return df

def create_device_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Create features from device information fields (id_30, id_31).
    """

    def setting_nan(value):
        """
        Convert 'NAN' string values to actual np.nan.
        """
        if value == 'NAN':
            return np.nan
        else:
            return value

    # Extract OS and version from id_30
    train['OS_id_30'] = train['id_30'].str.split(' ', expand=True)[0]
    train['version_id_30'] = train['id_30'].str.split(' ', expand=True)[1]
    test['OS_id_30'] = test['id_30'].str.split(' ', expand=True)[0]
    test['version_id_30'] = test['id_30'].str.split(' ', expand=True)[1]

    # Extract browser and version from id_31
    train['browser_id_31'] = train['id_31'].str.split(' ', expand=True)[0]
    train['version_id_31'] = train['id_31'].str.split(' ', expand=True)[1]
    test['browser_id_31'] = test['id_31'].str.split(' ', expand=True)[0]
    test['version_id_31'] = test['id_31'].str.split(' ', expand=True)[1]

    train = transform_device_info(train)
    test = transform_device_info(test)

    # Fix 'NAN' values
    train['DeviceCorp'] = train['DeviceCorp'].apply(setting_nan)
    test['DeviceCorp'] = test['DeviceCorp'].apply(setting_nan)

    return train, test


def get_categorical_features(baseline=False) -> List:
    """Return list of categorical features."""
    categorical_features = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo', 'hour_fraud_prob', 'LogTransactionAmt',
        'prob_card3', 'product_fraud_risk', 'OS_id_30', 'version_id_30',
        'browser_id_31', 'version_id_31', 'DeviceCorp'
    ]

    if baseline:
        categorical_features = categorical_features[:20]

    categorical_features += [f'id_{i}' for i in range(12, 39)]
    return categorical_features


def encode_categorical_features(train: pd.DataFrame,
                                val: pd.DataFrame,
                                test: pd.DataFrame,
                                categorical_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Encode categorical features using LabelEncoder on train, validation, and test sets."""
    for col in categorical_features:
        if col in train.columns:
            # Convert to string
            train[col] = train[col].astype(str)
            val[col] = val[col].astype(str)
            test[col] = test[col].astype(str)

            # Combine the column from all datasets to capture all categories
            combined_data = pd.concat([train[col], val[col], test[col]], axis=0)
            le = LabelEncoder()
            le.fit(combined_data)

            # Transform each dataset
            train[col] = le.transform(train[col])
            val[col] = le.transform(val[col])
            test[col] = le.transform(test[col])
    return train, val, test


def get_categorical_features_indices(train, baseline = False):
    """Return list of categorical features and their indices."""
    categorical_features = [
        'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'P_emaildomain', 'R_emaildomain',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'DeviceType', 'DeviceInfo', 'hour_fraud_prob', 'LogTransactionAmt',
        'prob_card3', 'product_fraud_risk', 'OS_id_30', 'version_id_30',
        'browser_id_31', 'version_id_31', 'DeviceCorp'
    ]

    if baseline:
        categorical_features = categorical_features[:20]

    categorical_features += [f'id_{i}' for i in range(12, 39)]

    # Get indices
    # First get the column names of your dataframe excluding TransactionID and isFraud
    all_features = train.drop(['TransactionID', 'isFraud'], axis=1, errors="ignore").columns

    # Get indices of categorical features
    cat_indices = [list(all_features).index(col) for col in categorical_features if col in all_features]

    return categorical_features, cat_indices


def normalize_d_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    train['D1n'] = np.floor(train.TransactionDT / (24*60*60)) - train.D1
    train['D2n'] = np.floor(train.TransactionDT / (24*60*60)) - train.D2
    train['D3n'] = np.floor(train.TransactionDT / (24*60*60)) - train.D3
    train['D15n'] = np.floor(train.TransactionDT / (24*60*60)) - train.D15
    test['D1n'] = np.floor(test.TransactionDT / (24*60*60)) - test.D1
    test['D2n'] = np.floor(test.TransactionDT / (24*60*60)) - test.D2
    test['D3n'] = np.floor(test.TransactionDT / (24*60*60)) - test.D3
    test['D15n'] = np.floor(test.TransactionDT / (24*60*60)) - test.D15

    return train, test



def run_feature_engineering(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Run the complete feature engineering pipeline.
    """
    missing_train_df = get_top_missing_cols(train, thresh=90)
    missing_test_df = get_top_missing_cols(test, thresh=90)

    # Drop columns with high missing values
    train, test = drop_high_missing_columns(train, test, missing_train_df, missing_test_df)

    # Remove outliers from train data
    train = remove_outliers(train)

    # Create time-based features
    train, test = create_time_features(train, test)
    train, test = normalize_d_features(train, test)

    # Process transaction amounts
    train, test = create_transaction_amount_features(train, test)

    # Create card features
    train, test = create_card_features(train, test)

    # Create product features
    train, test = create_product_features(train, test)

    # Process email domains
    train, test = process_email_domains(train, test)

    # Create device features
    train, test = create_device_features(train, test)

    return train, test
