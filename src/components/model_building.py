import numpy as np
import pandas as pd
import os
import sys
import pickle
import json
import logging
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from src.utils.exception import customexception
from src.utils.utils import load_params

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_engineered_data(input_path: str):
    """
    Load engineered train and test data.
    """
    try:
        train_path = os.path.join(input_path, 'train_engineered.csv')
        test_path = os.path.join(input_path, 'test_engineered.csv')
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        logging.info(f'Loaded train data: {train_data.shape}')
        logging.info(f'Loaded test data: {test_data.shape}')
        
        return train_data, test_data
        
    except Exception as e:
        logging.error(f'Error loading engineered data: {e}')
        raise


def prepare_data(df: pd.DataFrame, feature_list: list, target_col: str = 'Freight'):
    """
    Separate features and target.
    """
    try:
        # Select only the features in feature_list
        available_features = [f for f in feature_list if f in df.columns]
        
        if len(available_features) != len(feature_list):
            missing = set(feature_list) - set(available_features)
            logging.warning(f'Missing features: {missing}')
        
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        logging.info(f'Features shape: {X.shape}')
        logging.info(f'Features used: {available_features}')
        
        return X, y, available_features
        
    except Exception as e:
        logging.error(f'Error preparing data: {e}')
        raise


def scale_features(X_train, X_test, scaler_type='RobustScaler'):
    """
    Scale features using specified scaler.
    """
    try:
        if scaler_type == 'RobustScaler':
            scaler = RobustScaler()
        elif scaler_type == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler_type == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logging.info(f'Features scaled using {scaler_type}')
        
        return X_train_scaled, X_test_scaled, scaler
        
    except Exception as e:
        logging.error(f'Error scaling features: {e}')
        raise


def train_xgboost_model(X_train, y_train, params):
    """
    Train XGBoost model with given hyperparameters.
    """
    try:
        model = xgb.XGBRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            min_child_weight=params['min_child_weight'],
            random_state=params.get('random_state', 42),
            n_jobs=-1,
            objective='reg:squarederror'
        )
        
        logging.info('Training XGBoost model...')
        logging.info(f'Hyperparameters: {params}')
        
        model.fit(X_train, y_train, verbose=False)
        
        logging.info('XGBoost model trained successfully')
        
        return model
        
    except Exception as e:
        logging.error(f'Error training XGBoost model: {e}')
        raise


def save_model(model, scaler, feature_names, params, save_path):
    """
    Save model, scaler, and metadata.
    """
    try:
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_path, 'xgboost_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler
        scaler_path = os.path.join(save_path, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save metadata
        metadata = {
            'model_type': 'XGBoost',
            'features': feature_names,
            'n_features': len(feature_names),
            'hyperparameters': params,
            'train_samples': params.get('train_samples', None),
            'test_samples': params.get('test_samples', None),
            'total_features': len(feature_names),
            'experiment_name': params.get('experiment_name', 'XGBoost_Experiment'),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(save_path, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logging.info(f'Model saved to: {model_path}')
        logging.info(f'Scaler saved to: {scaler_path}')
        logging.info(f'Metadata saved to: {metadata_path}')
        
    except Exception as e:
        logging.error(f'Error saving model: {e}')
        raise


def main():
    """
    Main XGBoost model building pipeline.
    """
    try:
        # XGBoost Hyperparameters
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'gamma': 0,
            'min_child_weight': 1,
            'random_state': 42,
            'experiment_name': 'XGBoost_Advanced_Features',
            'scaler_type': 'RobustScaler',
            'train_samples': 4346,
            'test_samples': 1087,
            'total_features': 22
        }
        
        # Feature list
        feature_list = [
            'Quantity', 'Dollars', 'avg_unit_price', 'days_to_pay',
            'invoice_month', 'invoice_quarter', 'invoice_weekend',
            'is_holiday_season', 'is_summer', 'is_q4', 'payment_efficiency',
            'avg_begin_inventory', 'avg_end_inventory', 'avg_inventory',
            'inventory_turnover', 'freight_per_unit', 'freight_per_dollar',
            'log_quantity', 'log_dollars', 'log_freight'
        ]
        
        logging.info('='*60)
        logging.info('XGBOOST MODEL BUILDING')
        logging.info('='*60)
        logging.info(f'Experiment: {xgb_params["experiment_name"]}')
        logging.info(f'Total features: {len(feature_list)}')
        logging.info(f'Train samples: {xgb_params["train_samples"]}')
        logging.info(f'Test samples: {xgb_params["test_samples"]}')
        logging.info('='*60)
        
        # 1. Load engineered data
        logging.info('\n1. Loading engineered data...')
        input_path = 'data/processed'
        train_data, test_data = load_engineered_data(input_path)
        
        # 2. Prepare data with selected features
        logging.info('\n2. Preparing data with selected features...')
        X_train, y_train, used_features = prepare_data(train_data, feature_list, target_col='Freight')
        X_test, y_test, _ = prepare_data(test_data, feature_list, target_col='Freight')
        
        # 3. Scale features
        logging.info('\n3. Scaling features...')
        scaler_type = xgb_params['scaler_type']
        X_train_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_test, scaler_type
        )
        
        # 4. Train XGBoost model
        logging.info('\n4. Training XGBoost model...')
        model = train_xgboost_model(X_train_scaled, y_train, xgb_params)
        
        # 5. Save model and artifacts
        logging.info('\n5. Saving model and artifacts...')
        save_path = 'models/xgboost'
        save_model(model, scaler, used_features, xgb_params, save_path)
        
        logging.info('\n' + '='*60)
        logging.info('XGBOOST MODEL BUILDING COMPLETED')
        logging.info('='*60)
        logging.info(f'Model saved to: {save_path}')
        logging.info(f'Features used: {len(used_features)}')
        
    except Exception as e:
        logging.error(f'XGBoost model building failed: {e}')
        raise customexception(str(e), sys)


if __name__ == '__main__':
    main()