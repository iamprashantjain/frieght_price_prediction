import numpy as np
import pandas as pd
import os
import sys
import logging
from typing import List, Tuple
from src.utils.exception import customexception
from src.utils.utils import *
from src.utils.logging import logging

def load_raw_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the train and test data from data ingestion stage.
    
    Args:
        data_path: Path to the raw data folder
    
    Returns:
        Tuple of (train_data, test_data)
    """
    try:
        train_path = os.path.join(data_path, 'train.csv')
        test_path = os.path.join(data_path, 'test.csv')
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        logging.info(f'Loaded train data: {train_data.shape}')
        logging.info(f'Loaded test data: {test_data.shape}')
        
        return train_data, test_data
        
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        raise customexception(f"Error loading data: {e}", sys)


def engineer_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Add all engineered features to the dataframe.
    
    Args:
        df: Input dataframe
        is_train: Whether this is training data (affects some calculations)
    
    Returns:
        Dataframe with engineered features
    """
    try:
        df_copy = df.copy()
        
        # 1. Average unit price
        if 'Quantity' in df_copy.columns and 'Dollars' in df_copy.columns:
            df_copy['avg_unit_price'] = np.where(
                df_copy['Quantity'] > 0, 
                df_copy['Dollars'] / df_copy['Quantity'], 
                0
            )
            logging.info('✓ Added avg_unit_price')
        
        # 2. Days to pay (if dates available)
        if 'invoice_date' in df_copy.columns and 'payment_date' in df_copy.columns:
            df_copy['invoice_date'] = pd.to_datetime(df_copy['invoice_date'])
            df_copy['payment_date'] = pd.to_datetime(df_copy['payment_date'])
            df_copy['days_to_pay'] = (df_copy['payment_date'] - df_copy['invoice_date']).dt.days
            logging.info('✓ Added days_to_pay')
        
        # 3. Time-based features from invoice_date
        if 'invoice_date' in df_copy.columns:
            df_copy['invoice_month'] = df_copy['invoice_date'].dt.month
            df_copy['invoice_quarter'] = df_copy['invoice_date'].dt.quarter
            df_copy['invoice_weekend'] = (df_copy['invoice_date'].dt.dayofweek >= 5).astype(int)
            df_copy['is_holiday_season'] = df_copy['invoice_month'].isin([11, 12]).astype(int)
            df_copy['is_summer'] = df_copy['invoice_month'].isin([6, 7, 8]).astype(int)
            df_copy['is_q4'] = (df_copy['invoice_quarter'] == 4).astype(int)
            logging.info('✓ Added time-based features')
        
        # 4. Payment efficiency
        if 'days_to_pay' in df_copy.columns:
            # For training, use max from training data only
            if is_train:
                max_days = df_copy['days_to_pay'].max()
            else:
                # For test, we'll need to handle separately or use training max
                max_days = df_copy['days_to_pay'].max() if df_copy['days_to_pay'].max() > 0 else 1
            
            if max_days > 0:
                df_copy['payment_efficiency'] = 1 - (df_copy['days_to_pay'] / max_days)
            else:
                df_copy['payment_efficiency'] = 0
            logging.info('✓ Added payment_efficiency')
        
        # 5. Freight per unit
        if 'Freight' in df_copy.columns and 'Quantity' in df_copy.columns:
            df_copy['freight_per_unit'] = np.where(
                df_copy['Quantity'] > 0,
                df_copy['Freight'] / df_copy['Quantity'],
                0
            )
            logging.info('✓ Added freight_per_unit')
        
        # 6. Freight per dollar
        if 'Freight' in df_copy.columns and 'Dollars' in df_copy.columns:
            df_copy['freight_per_dollar'] = np.where(
                df_copy['Dollars'] > 0,
                df_copy['Freight'] / df_copy['Dollars'],
                0
            )
            logging.info('✓ Added freight_per_dollar')
        
        # 7. Log transformations
        if 'Quantity' in df_copy.columns:
            df_copy['log_quantity'] = np.log1p(df_copy['Quantity'])
            logging.info('✓ Added log_quantity')
        
        if 'Dollars' in df_copy.columns:
            df_copy['log_dollars'] = np.log1p(df_copy['Dollars'])
            logging.info('✓ Added log_dollars')
        
        if 'Freight' in df_copy.columns:
            df_copy['log_freight'] = np.log1p(df_copy['Freight'])
            logging.info('✓ Added log_freight')
        
        # 8. Handle infinite values
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
        
        # 9. Fill NaN values with 0 for engineered features
        engineered_features = [
            'avg_unit_price', 'days_to_pay', 'payment_efficiency',
            'freight_per_unit', 'freight_per_dollar',
            'log_quantity', 'log_dollars', 'log_freight'
        ]
        
        for feat in engineered_features:
            if feat in df_copy.columns:
                df_copy[feat] = df_copy[feat].fillna(0)
        
        logging.info(f'Total features after engineering: {len(df_copy.columns)}')
        
        return df_copy
        
    except Exception as e:
        logging.error(f'Error in feature engineering: {e}')
        raise customexception(f"Error in feature engineering: {e}", sys)


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the best features for modeling.
    
    Args:
        df: Input dataframe
    
    Returns:
        Dataframe with selected features
    """
    best_features = [
        'Quantity', 'Dollars', 'avg_unit_price', 'days_to_pay',
        'invoice_month', 'invoice_quarter', 'invoice_weekend',
        'is_holiday_season', 'is_summer', 'is_q4', 'payment_efficiency',
        'freight_per_unit', 'freight_per_dollar',
        'log_quantity', 'log_dollars', 'log_freight', 'Freight'
    ]
    
    # Keep only features that exist in the dataframe
    available_features = [f for f in best_features if f in df.columns]
    
    if 'Freight' not in available_features:
        logging.error('Target column Freight not found!')
        raise ValueError("Target column 'Freight' is required")
    
    logging.info(f'Selected {len(available_features)} features for modeling')
    
    return df[available_features]


def main():
    """
    Main feature engineering pipeline.
    """
    try:
        # Load parameters
        params = load_params(params_path='params.yaml')
        data_path = params['data_ingestion']['output_path']  # Path where data ingestion saved files
        
        logging.info('='*60)
        logging.info('FEATURE ENGINEERING STAGE STARTED')
        logging.info('='*60)
        
        # 1. Load raw data from data ingestion
        logging.info('Loading data from data ingestion stage...')
        train_data, test_data = load_raw_data(data_path)
        
        # 2. Engineer features for train data
        logging.info('\nEngineering features for train data...')
        train_engineered = engineer_features(train_data, is_train=True)
        
        # 3. Engineer features for test data
        logging.info('\nEngineering features for test data...')
        test_engineered = engineer_features(test_data, is_train=False)
        
        # 4. Select only best features
        logging.info('\nSelecting best features...')
        train_final = select_features(train_engineered)
        test_final = select_features(test_engineered)
        
        # 5. Save engineered data
        output_path = os.path.join('data', 'processed')
        os.makedirs(output_path, exist_ok=True)
        
        train_final.to_csv(os.path.join(output_path, 'train_engineered.csv'), index=False)
        test_final.to_csv(os.path.join(output_path, 'test_engineered.csv'), index=False)
        
        logging.info('\n' + '='*60)
        logging.info('FEATURE ENGINEERING COMPLETED SUCCESSFULLY')
        logging.info('='*60)
        logging.info(f'Engineered train data saved to: {output_path}/train_engineered.csv')
        logging.info(f'Engineered test data saved to: {output_path}/test_engineered.csv')
        logging.info(f'Final train shape: {train_final.shape}')
        logging.info(f'Final test shape: {test_final.shape}')
        logging.info(f'Features used: {list(train_final.columns[:-1])}')
        
    except Exception as e:
        logging.error(f'Feature engineering failed: {e}')
        raise customexception(str(e), sys)


if __name__ == '__main__':
    main()