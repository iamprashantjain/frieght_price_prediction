import numpy as np
import pandas as pd
import os
import sqlite3
import sys
import yaml
from src.utils.logging import logging
from src.utils.exception import customexception


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise customexception(f"File not found: {params_path}", sys)
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise customexception(f"YAML error: {e}", sys)
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise customexception(f"Unexpected error: {e}", sys)


def load_sqlite_data(db_path: str, table_name: str) -> pd.DataFrame:
    """Load data from SQLite database table."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        logging.debug('Data loaded from %s table in %s', table_name, db_path)
        return df
    except Exception as e:
        logging.error('Failed to load data from database: %s', e)
        raise customexception(f"Failed to load data: {e}", sys)


def add_features(vendor_df: pd.DataFrame, purchase_prices_df: pd.DataFrame, purchases_df: pd.DataFrame) -> pd.DataFrame:
    """Add Size and days_to_receive columns to vendor_invoice."""
    try:
        # Add Size column from purchase_prices using PONumber
        if 'PONumber' in vendor_df.columns and 'PONumber' in purchase_prices_df.columns:
            logging.info('Adding Size column from purchase_prices...')
            vendor_df = vendor_df.merge(
                purchase_prices_df[['PONumber', 'Size']],
                on='PONumber',
                how='left'
            )
            logging.info('Added Size column')
        else:
            logging.warning('PONumber column missing in vendor_df or purchase_prices_df')
        
        # Add days_to_receive column from purchases
        if 'PONumber' in vendor_df.columns and 'PONumber' in purchases_df.columns:
            logging.info('Adding days_to_receive column...')
            # Convert to datetime safely
            purchases_df['PODate'] = pd.to_datetime(purchases_df['PODate'], errors='coerce')
            purchases_df['ReceivingDate'] = pd.to_datetime(purchases_df['ReceivingDate'], errors='coerce')
            purchases_df['days_to_receive'] = (purchases_df['ReceivingDate'] - purchases_df['PODate']).dt.days
            
            vendor_df = vendor_df.merge(
                purchases_df[['PONumber', 'days_to_receive']],
                on='PONumber',
                how='left'
            )
            logging.info('Added days_to_receive column')
        else:
            logging.warning('PONumber column missing in vendor_df or purchases_df')
        
        return vendor_df
        
    except Exception as e:
        logging.error('Error adding features: %s', e)
        raise customexception(f"Error adding features: {e}", sys)


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        processed_data_path = os.path.join(data_path, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(processed_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(processed_data_path, "test.csv"), index=False)
        
        logging.debug('Train and test data saved to %s', processed_data_path)
        logging.info('Train shape: %s', train_data.shape)
        logging.info('Test shape: %s', test_data.shape)
        
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise customexception(f"Error saving data: {e}", sys)