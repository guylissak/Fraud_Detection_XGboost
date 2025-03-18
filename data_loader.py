""" Data loader module """
import pandas as pd
from typing import Tuple, Dict

def get_data_dtypes():
    """Define and return dtypes for all datasets."""
    dtypes_train_transaction = {
        'TransactionID': 'int32',
        'isFraud': 'int8',
        'TransactionDT': 'int32',
        'TransactionAmt': 'float32',
        'ProductCD': 'object',
        'card1': 'object',
        'card2': 'object',
        'card3': 'object',
        'card4': 'object',
        'card5': 'object',
        'card6': 'object',
        'addr1': 'float32',
        'addr2': 'float32',
        'dist1': 'float32',
        'dist2': 'float32',
        'P_emaildomain': 'object',
        'R_emaildomain': 'object',
        **{f'C{i}': 'float32' for i in range(1, 15)},
        **{f'D{i}': 'float32' for i in range(1, 16)},
        **{f'M{i}': 'object' for i in range(1, 10)},
        **{f'V{i}': 'float32' for i in range(1, 340)}
    }

    dtypes_train_identity = {
        **{f'id_{i:02}': 'float32' for i in range(1, 12)},
        'id_12': 'object',
        'id_13': 'float32',
        'id_14': 'float32',
        'id_15': 'object',
        'id_16': 'object',
        **{f'id_{i:02}': 'float32' for i in range(17, 40)},
        **{f'id_{i:02}': 'object' for i in range(23, 39)},
        'DeviceType': 'object',
        'DeviceInfo': 'object'
    }

    dtypes_test_transaction = dtypes_train_transaction.copy()
    del dtypes_test_transaction['isFraud']

    dtypes_test_identity = dtypes_train_identity.copy()

    return dtypes_train_transaction, dtypes_train_identity, dtypes_test_transaction, dtypes_test_identity

def load_raw_data(dtypes_train_transaction: Dict, dtypes_train_identity: Dict, dtypes_test_transaction: Dict,
                  dtypes_test_identity: Dict, local_path: str) -> Tuple[pd.DataFrame, ...]:
    """Load raw data from CSV files."""
    TRANSACTION_TRAIN_PATH = local_path + '/train_transaction.csv'
    IDENTITY_TRAIN_PATH = local_path + '/train_identity.csv'
    TRANSACTION_TEST_PATH = local_path + '/test_transaction.csv'
    IDENTITY_TEST_PATH = local_path + '/test_identity.csv'

    train_transaction = pd.read_csv(TRANSACTION_TRAIN_PATH, dtype=dtypes_train_transaction)
    train_identity = pd.read_csv(IDENTITY_TRAIN_PATH, dtype=dtypes_train_identity)
    test_transaction = pd.read_csv(TRANSACTION_TEST_PATH, dtype=dtypes_test_transaction)
    test_identity = pd.read_csv(IDENTITY_TEST_PATH, dtype=dtypes_test_identity)

    return train_transaction, train_identity, test_transaction, test_identity

def merge_datasets(train_transaction: pd.DataFrame, train_identity: pd.DataFrame,
                   test_transaction: pd.DataFrame, test_identity: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """Merge transaction and identity datasets."""
    # Standardize column names
    test_identity.columns = test_identity.columns.str.replace('-', '_')
    test_transaction.columns = test_transaction.columns.str.replace('-', '_')

    # Merge datasets
    train = train_transaction.merge(train_identity, how='left', on='TransactionID')
    test = test_transaction.merge(test_identity, how='left', on='TransactionID')

    return train, test

