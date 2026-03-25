import numpy as np
import pandas as pd
import os
import sqlite3
from sklearn.model_selection import train_test_split
import yaml
from src.utils.logging import logging
from src.utils.exception import customexception
from src.utils.utils import *


try:
    params = load_params(params_path='params.yaml')
    db_path = params['data_ingestion']['database_path']
    test_size = params['data_ingestion']['test_size']
    random_state = params['data_ingestion']['random_state']

    logging.info('Loading vendor_invoice...')
    vendor_df = load_sqlite_data(db_path, 'vendor_invoice')

    logging.info('Loading purchase_prices...')
    purchase_prices_df = load_sqlite_data(db_path, 'purchase_prices')

    logging.info('Loading purchases...')
    purchases_df = load_sqlite_data(db_path, 'purchases')

    # Add features (Size and days_to_receive)
    logging.info('Adding features...')
    vendor_df = add_features(vendor_df, purchase_prices_df, purchases_df)

    # Select required columns for modeling
    required_cols = ['Quantity', 'Dollars', 'Size', 'days_to_receive', 'Freight']
    available_cols = [col for col in required_cols if col in vendor_df.columns]

    logging.info('Selected columns: %s', available_cols)

    # Prepare features and target
    X = vendor_df[available_cols].copy()
    y = vendor_df['Freight'].copy()

    # Drop rows with missing values
    # Check for NaN values before dropping
    initial_rows = len(X)
    X = X.dropna()
    y = y.loc[X.index]

    logging.info('Data shape after dropping missing: %s', X.shape)
    logging.info('Dropped %d rows with missing values', initial_rows - len(X))

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Combine features and target for saving
    train_data = X_train.copy()
    train_data['Freight'] = y_train

    test_data = X_test.copy()
    test_data['Freight'] = y_test

    # Save train and test data
    save_data(train_data, test_data, os.path.join('data', 'raw'))

    logging.info('='*60)
    logging.info('DATA INGESTION COMPLETED SUCCESSFULLY')
    logging.info('='*60)
    logging.info('Train data saved to: ./data/raw/train.csv')
    logging.info('Test data saved to: ./data/raw/test.csv')
    logging.info('Train size: %d rows', len(train_data))
    logging.info('Test size: %d rows', len(test_data))
    logging.info('Features: %s', available_cols)
        
except Exception as e:
    logging.error('Failed to complete the data ingestion process: %s', e)
    raise customexception(str(e), sys)