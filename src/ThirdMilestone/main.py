"""
E-Commerce Fraud Detection Learning
COMPSCI 4AL3 - Group 34
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, precision_score, recall_score, accuracy_score,
                             fbeta_score)
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import cupy as cp
import cudf
import cuml
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.linear_model import LogisticRegression as cuLR
import xgboost as xgb
import shap
import kagglehub

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 123
np.random.seed(SEED)
DEVICE = 0

# GPU Detection and Configuration
def detect_gpu():
    """Detect available GPUs and configure accordingly"""
    gpu_info = {
        'gpu_available': False,
        'device': 'cpu',
        'gpu_name': None,
        'cuda_available': False,
        'cupy_available': False,
        'rapids_available': False
    }
    
    # Check for CUDA
    try:
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['gpu_available'] = True
            gpu_info['device'] = 'cuda'
            gpu_info['gpu_name'] = torch.cuda.get_device_name(DEVICE)
            print(f"CUDA GPU detected: {gpu_info['gpu_name']}")
    except ImportError:
        pass
    
    # Check for CuPy
    try:
        gpu_info['cupy_available'] = True
        gpu_info['gpu_available'] = True
        print("CuPy available")
    except ImportError:
        pass
    
    # Check for RAPIDS cuML
    try:
        gpu_info['rapids_available'] = True
        gpu_info['gpu_available'] = True
        print("RAPIDS cuML available")
    except ImportError:
        pass
    
    return gpu_info

# Initialize GPU detection
GPU_INFO = detect_gpu()


# 1. DATA LOADING

def load_data_from_kaggle(version='v1', cache_dir=None):
    """Load data (either v1 or v2) from Kaggle from cache or download if needed"""
    
    if cache_dir is None:
        # Default kagglehub cache location
        cache_dir = os.path.join(
            os.path.expanduser('~'), 
            '.cache', 'kagglehub', 'datasets',
            'shriyashjagtap', 'fraudulent-e-commerce-transactions', 
            'versions'
        )
    
    # Determine which version to load
    if version.lower() == 'v1':
        version_num = '1'
        expected_file = 'Fraudulent_E-Commerce_Transaction_Data.csv'
        print("Loading Version 1")
    elif version.lower() == 'v2':
        version_num = '2'
        expected_file = 'Fraudulent_E-Commerce_Transaction_Data_2.csv'
        print("Loading Version 2")
    else:
        raise ValueError("Version must be 'v1' or 'v2'")
    
    # Construct full path
    version_dir = os.path.join(cache_dir, version_num)
    filepath = os.path.join(version_dir, expected_file)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print("Dataset not found at expected location.")
        print("Attempting to download from Kaggle...")
        
        try:
            # Download latest version (this gets both versions)
            download_path = kagglehub.dataset_download(
                "shriyashjagtap/fraudulent-e-commerce-transactions"
            )
            print(f"Downloaded to: {download_path}")
            
            # Try to find the file in the download location
            for root, dirs, files in os.walk(download_path):
                if expected_file in files:
                    filepath = os.path.join(root, expected_file)
                    break
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Could not find {expected_file} after download")
                
        except ImportError:
            print("\nkagglehub not installed.")
            raise
        except Exception as e:
            print(f"\nError downloading dataset: {str(e)}")
            raise
    
    # Load the dataset
    print(f"\nLoading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    
    print("\nDATASET OVERVIEW")
    print(f"Dataset version: {version.upper()}")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nClass distribution:\n{df['Is Fraudulent'].value_counts()}")
    print(f"\nFraud percentage: {df['Is Fraudulent'].mean()*100:.2f}%")
    
    return df

def analyze_customer_location_leakage(df):
    """Analyze Customer Location feature for potential data leakage"""
    print("\nCUSTOMER LOCATION LEAKAGE ANALYSIS")
    
    # Calculate fraud rate by location
    location_fraud_rate = df.groupby('Customer Location')['Is Fraudulent'].agg(['mean', 'count'])
    location_fraud_rate = location_fraud_rate.sort_values('mean', ascending=False)
    
    print(f"\nTop 10 locations by fraud rate:")
    print(location_fraud_rate.head(10))
    
    print(f"\nBottom 10 locations by fraud rate:")
    print(location_fraud_rate.tail(10))
    
    # Check if any locations have extreme fraud rates
    extreme_fraud = location_fraud_rate[location_fraud_rate['mean'] > 0.9]
    extreme_legit = location_fraud_rate[location_fraud_rate['mean'] < 0.1]
    
    print(f"\nLocations with >90% fraud rate: {len(extreme_fraud)}")
    print(f"Locations with <10% fraud rate: {len(extreme_legit)}")
    
    # Calculate correlation with target
    location_encoded = df.groupby('Customer Location')['Is Fraudulent'].transform('mean')
    correlation = location_encoded.corr(df['Is Fraudulent'])
    print(f"\nCorrelation between location fraud rate and target: {correlation:.4f}")
    
    if correlation > 0.5:
        print("WARNING: High correlation detected - possible data leakage")
        print("Consider removing this feature")
    else:
        print("Location shows normal geographic patterns")
    
    return location_fraud_rate

# 2. FEATURE ENGINEERING

def engineer_features_gpu(df):
    """GPU-accelerated feature engineering using RAPIDS cuDF"""
    print("\nGPU FEATURE ENGINEERING")
    
    try:        
        # Set GPU device to match torch (device 1)
        cp.cuda.Device(DEVICE).use()

        # Convert pandas → cuDF if needed
        if isinstance(df, pd.DataFrame):
            df = cudf.from_pandas(df)
    except (RuntimeError, MemoryError, Exception) as e:
        print(f"GPU out of memory or error: {e}")
        print("Falling back to CPU feature engineering...")
        return engineer_features(df)

    # Parse date
    df['Transaction Date'] = cudf.to_datetime(df['Transaction Date'])

    print("Creating basic features...")
    df['Address Mismatch'] = (df['Shipping Address'] != df['Billing Address']).astype('int8')
    df['Day of Week'] = df['Transaction Date'].dt.weekday
    df['Month'] = df['Transaction Date'].dt.month
    df['Is Weekend'] = (df['Day of Week'] >= 5).astype('int8')
    df['New Account'] = (df['Account Age Days'] < 30).astype('int8')

    # GPU quantile
    q95 = df['Transaction Amount'].quantile(0.95)
    
    print("Creating advanced features (log transforms, z-scores, binning, interactions)...")
    
    # 1. Transaction Amount Features
    df['Amount Log'] = cudf.Series(cp.log1p(df['Transaction Amount'].values))
    df['Amount per Quantity'] = df['Transaction Amount'] / (df['Quantity'] + 1)
    # Z-score using cupy
    amount_mean = df['Transaction Amount'].mean()
    amount_std = df['Transaction Amount'].std()
    df['Amount zscore'] = (df['Transaction Amount'] - amount_mean) / amount_std
    
    # 2. Time features
    df['Hour Bin'] = cudf.Series(['Night'] * len(df), dtype='object')
    df.loc[df['Transaction Hour'] >= 6, 'Hour Bin'] = 'Morning'
    df.loc[df['Transaction Hour'] >= 12, 'Hour Bin'] = 'Afternoon'
    df.loc[df['Transaction Hour'] >= 18, 'Hour Bin'] = 'Evening'
    
    # 3. Customer profile
    df['Age Category'] = cudf.Series(['Young'] * len(df), dtype='object')
    df.loc[df['Customer Age'] >= 25, 'Age Category'] = 'Young_Adult'
    df.loc[df['Customer Age'] >= 35, 'Age Category'] = 'Adult'
    df.loc[df['Customer Age'] >= 50, 'Age Category'] = 'Senior'
    df.loc[df['Customer Age'] >= 65, 'Age Category'] = 'Elder'
    
    df['Account Age Weeks'] = df['Account Age Days'] // 7
    
    # 4. Transaction patterns
    quantiles = df['Transaction Amount'].quantile([0.2, 0.4, 0.6, 0.8])
    df['Transaction Size'] = cudf.Series(['Very_Small'] * len(df), dtype='object')
    df.loc[df['Transaction Amount'] > quantiles[0.2], 'Transaction Size'] = 'Small'
    df.loc[df['Transaction Amount'] > quantiles[0.4], 'Transaction Size'] = 'Medium'
    df.loc[df['Transaction Amount'] > quantiles[0.6], 'Transaction Size'] = 'Large'
    df.loc[df['Transaction Amount'] > quantiles[0.8], 'Transaction Size'] = 'Very_Large'
    
    df['Quantity Log'] = cudf.Series(cp.log1p(df['Quantity'].values))
    
    # 6. Risk Indicators
    df['High Amount Flag'] = (df['Transaction Amount'] > q95).astype('int8')
    q95_quantity = df['Quantity'].quantile(0.95)
    df['High Quantity Flag'] = (df['Quantity'] > q95_quantity).astype('int8')
    df['Unusual Hour Flag'] = ((df['Transaction Hour'] < 6) | (df['Transaction Hour'] > 22)).astype('int8')

    print("Computing customer aggregations...")
    customer_agg = df.groupby('Customer ID').agg({
        'Transaction ID': 'nunique',
        'Transaction Amount': ['mean', 'std', 'max'],
        'Product Category': 'nunique'
    })

    # Flatten columns
    customer_agg.columns = [
        'Total Customer Transactions',
        'Avg Customer Transaction',
        'Std Customer Transaction',
        'Max Customer Transaction',
        'Product Category Diversity'
    ]

    # Merge
    df = df.merge(customer_agg, on='Customer ID', how='left')

    df['Transaction Amount Ratio'] = (
        df['Transaction Amount'] /
        (df['Avg Customer Transaction'] + 1)
    )

    print("Computing temporal features...")

    # Sort once
    df = df.sort_values(['Customer ID', 'Transaction Date'])

    # Time since last transaction (cuDF diff)
    df['Time Since Last Transaction'] = (
        df.groupby('Customer ID')['Transaction Date']
        .diff()
        .dt.total_seconds()
        / 3600
    ).fillna(0)

    # Rolling average (GPU)
    df['Rolling Avg Amount'] = (
        df.groupby('Customer ID')['Transaction Amount']
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(drop=True)
    )

    df['Amount Deviation From History'] = (
        (df['Transaction Amount'] - df['Rolling Avg Amount']) /
        (df['Rolling Avg Amount'] + 1)
    )

    # Transaction velocity
    df['Transaction Date Only'] = df['Transaction Date'].dt.floor('D')

    daily_counts = df.groupby(['Customer ID', 'Transaction Date Only']).size()
    customer_velocity = daily_counts.groupby('Customer ID').mean()

    df['Avg Daily Transaction Velocity'] = df['Customer ID'].map(customer_velocity)
    
    # Add interaction features
    print("Creating interaction features...")
    df['Amount Age Interaction'] = df['Transaction Amount'] * df['Customer Age']
    df['Amount Velocity Interaction'] = df['Transaction Amount'] * df['Avg Daily Transaction Velocity']
    df['New Account High Value'] = df['New Account'] * df['High Amount Flag']
    df['Weekend High Value'] = df['Is Weekend'] * df['High Amount Flag']
    df['High Risk Profile'] = df['New Account'] * df['High Amount Flag'] * df['Address Mismatch']
    df['Velocity Deviation'] = df['Avg Daily Transaction Velocity'] * df['Amount Deviation From History']
    df['Suspicious Pattern'] = (df['Unusual Hour Flag'] * df['High Amount Flag'] * df['Is Weekend']).astype('int8')

    # Cleanup
    df = df.drop(['Rolling Avg Amount', 'Transaction Date Only'], axis=1)

    print(f"\nTotal of {df.shape[1]-1} features after GPU engineering")
    return df.to_pandas()

def engineer_features(df):
    """Create features from data"""
    print("\nFEATURE ENGINEERING")
    
    df = df.copy()
    
    # Parse transaction date once
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

    print("Creating basic features...")
    df['Address Mismatch'] = (df['Shipping Address'] != df['Billing Address']).astype(int)
    df['Day of Week'] = df['Transaction Date'].dt.dayofweek
    df['Month'] = df['Transaction Date'].dt.month
    df['Is Weekend'] = (df['Day of Week'] >= 5).astype(int)
    df['New Account'] = (df['Account Age Days'] < 30).astype(int)
    
    print("Creating advanced features (log transforms, z-scores, binning, interactions)...")
    
    # 1. Transaction Amount Features
    df['Amount Log'] = np.log1p(df['Transaction Amount'])
    df['Amount per Quantity'] = df['Transaction Amount'] / (df['Quantity'] + 1)
    from scipy import stats
    df['Amount zscore'] = stats.zscore(df['Transaction Amount'], nan_policy='omit')
    
    # 2. Time-based Features
    df['Hour Bin'] = pd.cut(df['Transaction Hour'], bins=[-np.inf, 6, 12, 18, np.inf], 
                            labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    # 3. Customer Profile Features
    df['Age Category'] = pd.cut(df['Customer Age'], bins=[0, 25, 35, 50, 65, np.inf], 
                                labels=['Young', 'Young_Adult', 'Adult', 'Senior', 'Elder'])
    df['Account Age Weeks'] = df['Account Age Days'] // 7
    
    # 4. Transaction Pattern Features
    df['Transaction Size'] = pd.qcut(df['Transaction Amount'], q=5, 
                                      labels=['Very_Small', 'Small', 'Medium', 'Large', 'Very_Large'], 
                                      duplicates='drop')
    df['Quantity Log'] = np.log1p(df['Quantity'])
    
    # 6. Risk Indicators
    df['High Amount Flag'] = (df['Transaction Amount'] > df['Transaction Amount'].quantile(0.95)).astype(int)
    df['High Quantity Flag'] = (df['Quantity'] > df['Quantity'].quantile(0.95)).astype(int)
    df['Unusual Hour Flag'] = ((df['Transaction Hour'] < 6) | (df['Transaction Hour'] > 22)).astype(int)
    
    # Aggregated customer features (vectorized)
    print("Computing customer-level aggregations...")
    customer_agg = df.groupby('Customer ID').agg({
        'Transaction ID': 'nunique',
        'Transaction Amount': ['mean', 'std', 'max'],
        'Product Category': 'nunique'
    })
    customer_agg.columns = ['Total Customer Transactions', 'Avg Customer Transaction', 
                            'Std Customer Transaction', 'Max Customer Transaction', 
                            'Product Category Diversity']
    
    # Merge customer features
    df = df.merge(customer_agg, left_on='Customer ID', right_index=True, how='left')
    df['Transaction Amount Ratio'] = df['Transaction Amount'] / (df['Avg Customer Transaction'] + 1)
    
    # Temporal features
    print("Computing temporal features...")
    
    # Sort once
    df = df.sort_values(['Customer ID', 'Transaction Date']).reset_index(drop=True)
    
    # Time since last transaction (vectorized within groups)
    df['Time Since Last Transaction'] = df.groupby('Customer ID')['Transaction Date'].diff().dt.total_seconds() / 3600
    df['Time Since Last Transaction'] = df['Time Since Last Transaction'].fillna(0)

    # Rolling average and deviation (optimized)
    df['Rolling Avg Amount'] = df.groupby('Customer ID')['Transaction Amount'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    df['Amount Deviation From History'] = (df['Transaction Amount'] - df['Rolling Avg Amount']) / (df['Rolling Avg Amount'] + 1)

    # Transaction velocity (optimized with date grouping)
    df['Transaction Date Only'] = df['Transaction Date'].dt.date
    daily_counts = df.groupby(['Customer ID', 'Transaction Date Only']).size()
    customer_velocity = daily_counts.groupby('Customer ID').mean()
    df['Avg Daily Transaction Velocity'] = df['Customer ID'].map(customer_velocity)
    
    # Add interaction features
    print("Creating interaction features...")
    df['Amount Age Interaction'] = df['Transaction Amount'] * df['Customer Age']
    df['Amount Velocity Interaction'] = df['Transaction Amount'] * df['Avg Daily Transaction Velocity']
    df['New Account High Value'] = df['New Account'] * df['High Amount Flag']
    df['Weekend High Value'] = df['Is Weekend'] * df['High Amount Flag']
    df['High Risk Profile'] = df['New Account'] * df['High Amount Flag'] * df['Address Mismatch']
    df['Velocity Deviation'] = df['Avg Daily Transaction Velocity'] * df['Amount Deviation From History']
    df['Suspicious Pattern'] = (df['Unusual Hour Flag'] * df['High Amount Flag'] * df['Is Weekend']).astype(int)
    
    # Clean up temporary columns
    df = df.drop(['Rolling Avg Amount', 'Transaction Date Only'], axis=1)
    
    print(f"\nTotal features: {df.shape[1]-1}")
    return df


# 3. DATA PREPROCESSING

def preprocess_data(df, label_encoder=False):
    """Preprocess data for modeling"""
    print("\nDATA PREPROCESSING")
    
    df = df.copy()
        
    # Select features to use
    features_to_use = [
        # Numerical features
        'Transaction Amount', 'Quantity', 'Customer Age', 
        'Account Age Days', 'Transaction Hour',
        # Engineered features
        'Total Customer Transactions', 'Address Mismatch',
        'Day of Week', 'Month', 'Is Weekend', 'New Account',
        'Transaction Amount Ratio',
        # Advanced behavioral features
        'Avg Daily Transaction Velocity', 'Time Since Last Transaction',
        'Amount Deviation From History', 'Product Category Diversity',
        'Amount Log', 'Amount per Quantity', 'Amount zscore',
        'Hour Bin', 'Age Category', 'Account Age Weeks',
        'Transaction Size', 'Quantity Log',
        'High Amount Flag', 'High Quantity Flag', 'Unusual Hour Flag',
        # Interaction features
        'Amount Age Interaction', 'Amount Velocity Interaction',
        'New Account High Value', 'Weekend High Value',
        'High Risk Profile', 'Velocity Deviation', 'Suspicious Pattern',
        # Categorical features (to be encoded)
        'Payment Method', 'Product Category', 'Device Used', 'Customer Location'
    ]
    
    # Check for available columns (this is used for testing, since we edit code a lot)
    features_to_use = [f for f in features_to_use if f in df.columns]
    
    # Create feature dataframe
    X = df[features_to_use].copy()
    y = df['Is Fraudulent'].copy()
    
    print(f"Using {len(features_to_use)} features")
    
    # Handle missing values
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Fill missing numerical values with median
    for col in numerical_features:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)
    
    # Fill missing categorical values with mode
    for col in categorical_features:
        if X[col].isnull().any():
            X[col].fillna(X[col].mode()[0], inplace=True)
    
    if label_encoder:
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        print(f"Encoded {len(categorical_features)} categorical features")
    else:
        # One-hot encode categorical features (including the new ones)
        one_hot_cols = ['Payment Method', 'Product Category', 'Device Used', 
                       'Hour Bin', 'Age Category', 'Transaction Size']
    
        for col in one_hot_cols:
            if col in X.columns:
                X = pd.get_dummies(X, columns=[col], drop_first=True)
        
        # Target encode Customer Location (if not already dropped via Location Device)
        if 'Customer Location' in X.columns:
            location_means = df.groupby('Customer Location')['Is Fraudulent'].mean()
            X['Customer Location'] = X['Customer Location'].map(location_means)
        
        print("Applied one-hot encoding and target encoding to categorical features")
    
    print(f"Total of {X.shape[1]} features will be used by the model")
    return X, y


# 4. TRAIN-TEST SPLIT AND SCALING

def split_and_scale_data(X, y, test_size=0.15, val_size=0.15):
    """Split data into train, validation, and test sets with scaling"""
    print("\nDATA SPLITTING AND SCALING")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    
    # Second split: separate validation set from remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=SEED, stratify=y_temp
    )
    
    print(f"Train set size: {X_train.shape[0]} ({len(y_train[y_train==1])} fraudulent)")
    print(f"Validation set size: {X_val.shape[0]} ({len(y_val[y_val==1])} fraudulent)")
    print(f"Test set size: {X_test.shape[0]} ({len(y_test[y_test==1])} fraudulent)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nApplying feature selection to remove low-variance features...")
    selector = VarianceThreshold(threshold=0.01)
    X_train_scaled = selector.fit_transform(X_train_scaled)
    X_val_scaled = selector.transform(X_val_scaled)
    X_test_scaled = selector.transform(X_test_scaled)
    print(f"Reduced to {X_train_scaled.shape[1]} features after variance threshold")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler



def apply_smote(X_train, y_train): # bru
    """Apply SMOTE to balance training data"""
    print("\nAPPLYING SMOTE FOR CLASS BALANCE")

    print(f"Before SMOTE - Fraudulent: {sum(y_train)}, Legitimate: {len(y_train) - sum(y_train)}")

    smote = SMOTE(random_state=SEED)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE - Fraudulent: {sum(y_train_balanced)}, Legitimate: {len(y_train_balanced) - sum(y_train_balanced)}")

    return X_train_balanced, y_train_balanced

def apply_hybrid_sampling(X_train, y_train, target_ratio=0.3):
    """Apply hybrid sampling: SMOTE to increase minority, then undersample majority"""
    print("\nAPPLYING HYBRID SAMPLING (SMOTE + UNDERSAMPLING)")
    print(f"Target ratio: {target_ratio:.2f}")
    print(f"Before sampling - Fraudulent: {sum(y_train)}, Legitimate: {len(y_train) - sum(y_train)}")
    
    # First, use SMOTE to increase minority class moderately
    smote = SMOTE(sampling_strategy=target_ratio, random_state=SEED)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Then undersample majority to achieve better balance
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=SEED)  # 1:2 ratio
    X_balanced, y_balanced = rus.fit_resample(X_resampled, y_resampled)
    
    print(f"After hybrid sampling - Fraudulent: {sum(y_balanced)}, Legitimate: {len(y_balanced) - sum(y_balanced)}")
    
    return X_balanced, y_balanced

# 5. HANDLE CLASS IMBALANCE WITH ADASYN + UNDERSAMPLING

def apply_adasyn(X_train, y_train, sampling_ratio=1.0):
    """Apply ADASYN to balance training data"""
    print("\nAPPLYING ADASYN")
    print(f"Target ratio: {sampling_ratio:.2f}")
    print(f"Before resampling - Fraudulent: {sum(y_train)}, Legitimate: {len(y_train) - sum(y_train)}")
    
    try:
        # ADASYN Oversampling - adaptively generates synthetic samples based on density
        # Using n_neighbors=10 for better synthetic sample quality with large dataset
        adasyn = ADASYN(sampling_strategy=sampling_ratio, random_state=SEED, n_neighbors=10)
        X_balanced, y_balanced = adasyn.fit_resample(X_train, y_train)
        
        fraud_count = sum(y_balanced)
        legit_count = len(y_balanced) - fraud_count
        actual_ratio = fraud_count / legit_count if legit_count > 0 else 0
        
        print(f"After ADASYN - Fraudulent: {fraud_count}, Legitimate: {legit_count}")
        print(f"Actual class ratio: {actual_ratio:.3f} ({fraud_count/(fraud_count+legit_count)*100:.1f}% fraudulent)")
        
        return X_balanced, y_balanced
    except Exception as e:
        print(f"ADASYN failed: {e}")
        print("Falling back to SMOTE...")
        return apply_smote(X_train, y_train)

def optimize_threshold(y_true, y_proba, beta=1, cost_fp=1, cost_fn=5, optimize_for='f_beta'):
    """Find optimal classification threshold"""
    print(f"\nOPTIMIZING THRESHOLD (beta={beta}, FN cost={cost_fn}x, FP cost={cost_fp}x, optimize_for={optimize_for})")
    
    thresholds = np.arange(0.02, 0.95, 0.01)  # Start lower for better recall
    best_score = float('inf') if optimize_for == 'cost' else 0
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            f_beta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        else:
            f1 = 0
            f_beta = 0
        
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        
        # Choose optimization criterion
        if optimize_for == 'cost':
            current_score = total_cost
            is_better = current_score < best_score
        else:  # optimize for f_beta
            current_score = f_beta
            is_better = current_score > best_score
        
        if is_better:
            best_score = current_score
            best_threshold = threshold
            best_metrics = {
                'f_beta': f_beta,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'fp': fp,
                'fn': fn,
                'total_cost': total_cost
            }
    
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"F{beta}-Score: {best_metrics['f_beta']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1-Score: {best_metrics['f1']:.4f}")
    print(f"Total Cost: {best_metrics['total_cost']} (FP: {best_metrics['fp']}, FN: {best_metrics['fn']})")
    
    return best_threshold, best_metrics


# 6. MODEL TRAINING

def train_logistic_regression(X_train, y_train, X_val, y_val, use_gpu=True, optimize_thresh=True, optimization_mode='recall'):
    """Train Logistic Regression model"""
    print(f"\n== TRAINING LOGISTIC REGRESSION (Optimized for {optimization_mode.upper()}) ==")
    
    if optimization_mode == 'recall':
        class_weights = {0: 1, 1: 50}
        print("Mode: Aggressive weights for high recall")
    else:
        class_weights = {0: 1, 1: 10}
        print("Mode: Conservative weights for high precision")
    
    # Try GPU-accelerated cuML Logistic Regression
    if use_gpu and GPU_INFO['rapids_available']:
        print("Using GPU-accelerated RAPIDS cuML Logistic Regression")
        try:
            # Set GPU device
            cp.cuda.Device(DEVICE).use()
            
            # Convert to GPU arrays
            X_train_gpu = cp.array(X_train)
            y_train_gpu = cp.array(y_train)
            X_val_gpu = cp.array(X_val)
            
            # cuML LogisticRegression uses class_weight parameter
            best_c = 1.0
            best_penalty = 'l2'
            best_f1 = 0
            
            print("GPU hyperparameter tuning...")
            for C in [0.001, 0.01, 0.1, 1, 10, 100]:
                for penalty in ['l1', 'l2']:
                    lr_gpu = cuLR(
                        C=C,
                        penalty=penalty,
                        max_iter=1000,
                        tol=1e-4,
                        class_weight=class_weights
                    )
                    lr_gpu.fit(X_train_gpu, y_train_gpu)
                    
                    # Validate
                    y_val_pred_gpu = lr_gpu.predict(X_val_gpu)
                    y_val_pred = cp.asnumpy(y_val_pred_gpu)
                    f1 = f1_score(y_val, y_val_pred)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_c = C
                        best_penalty = penalty
            
            print(f"Best parameters: C={best_c}, penalty={best_penalty}")
            print(f"Applied class weights: {class_weights}")
            
            # Train final model with best parameters
            best_lr = cuLR(
                C=best_c,
                penalty=best_penalty,
                max_iter=1000,
                tol=1e-4,
                class_weight=class_weights
            )
            best_lr.fit(X_train_gpu, y_train_gpu)
            
            # Validation predictions with threshold optimization
            y_val_proba_gpu = best_lr.predict_proba(X_val_gpu)[:, 1]
            y_val_prob = cp.asnumpy(y_val_proba_gpu)
            
            if optimize_thresh:
                if optimization_mode == 'recall':
                    # Maximize recall: low threshold, heavily penalize missed fraud (FN)
                    best_threshold, _ = optimize_threshold(y_val, y_val_prob, beta=5, cost_fn=50, cost_fp=1)
                else:
                    # Maximize precision: higher threshold, penalize false alarms (FP)
                    best_threshold, _ = optimize_threshold(y_val, y_val_prob, beta=0.5, cost_fn=1, cost_fp=50)
                best_lr.threshold_ = best_threshold
                y_val_pred = (y_val_prob >= best_threshold).astype(int)
            else:
                best_lr.threshold_ = 0.5
                y_val_pred_gpu = best_lr.predict(X_val_gpu)
                y_val_pred = cp.asnumpy(y_val_pred_gpu)
            
            print("\nValidation Performance for Logistic Regression:")
            print_metrics(y_val, y_val_pred, y_val_prob)
            
            return best_lr
            
        except Exception as e:
            print(f"GPU training failed: {e}")
            print("Falling back to CPU...")
    
    # CPU-based sklearn Logistic Regression with hyperparameter tuning
    print("Using CPU-based scikit-learn Logistic Regression")
    print(f"Applied class weights: {class_weights}")
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    lr = LogisticRegression(class_weight=class_weights, random_state=SEED, max_iter=1000, n_jobs=-1)
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    grid_search = GridSearchCV(lr, param_grid, cv=stratified_kfold, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_lr = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Validation predictions with threshold optimization
    y_val_prob = best_lr.predict_proba(X_val)[:, 1]
    
    if optimize_thresh:
        if optimization_mode == 'recall':
            # Maximize recall: low threshold, heavily penalize missed fraud (FN)
            best_threshold, _ = optimize_threshold(y_val, y_val_prob, beta=6, cost_fn=50, cost_fp=1)
        else:
            # Maximize precision: higher threshold, penalize false alarms (FP)
            best_threshold, _ = optimize_threshold(y_val, y_val_prob, beta=0.5, cost_fn=1, cost_fp=50)
        best_lr.threshold_ = best_threshold
        y_val_pred = (y_val_prob >= best_threshold).astype(int)
    else:
        best_lr.threshold_ = 0.5
        y_val_pred = best_lr.predict(X_val)
    
    print("\nValidation Performance for Logistic Regression:")
    print_metrics(y_val, y_val_pred, y_val_prob)
    
    return best_lr

def permutation_importance(model, X, y, metric=accuracy_score):
    """Compute feature importances using permutation importance on GPU"""
    base_score = metric(cp.asnumpy(y), cp.asnumpy(model.predict(X)))
    importances = cp.zeros(X.shape[1], dtype=cp.float32)

    for i in range(X.shape[1]):
        X_perm = X.copy()
        X_perm[:, i] = cp.random.permutation(X_perm[:, i])
        perm_score = metric(cp.asnumpy(y), cp.asnumpy(model.predict(X_perm)))
        importances[i] = base_score - perm_score

    return importances / importances.sum()

def train_random_forest(X_train, y_train, X_val, y_val, use_gpu=True, optimize_thresh=True, optimization_mode='recall'):
    """Train Random Forest model"""
    print(f"\n== TRAINING RANDOM FOREST (Optimized for {optimization_mode.upper()}) ==")
    
    if optimization_mode == 'recall':
        class_weights = {0: 1, 1: 50}
        print("Mode: Aggressive weights for high recall")
    else:
        class_weights = {0: 1, 1: 10}
        print("Mode: Conservative weights for high precision")
    
    # Use RAPIDS cuML Random Forest if available and requested
    if use_gpu and GPU_INFO['rapids_available']:
        print("Using GPU-accelerated RAPIDS cuML Random Forest")
        try:            
            # Convert to GPU arrays
            X_train_gpu = cp.array(X_train)
            y_train_gpu = cp.array(y_train)
            X_val_gpu = cp.array(X_val)
            
            # Train on GPU (RAPIDS cuML doesn't support sample_weight parameter)
            # Cost-sensitive learning will be handled through threshold optimization
            rf = cuRF(
                n_estimators=800, # Larger forest (500–1000 trees) - bru
                max_depth=40, # Deeper trees (30–40)
                min_samples_split=5,
                random_state=SEED,
                n_streams=4  # Parallel GPU streams
            )
            
            rf.fit(X_train_gpu, y_train_gpu)
            
            # Validation predictions with threshold optimization
            y_val_proba = cp.asnumpy(rf.predict_proba(X_val_gpu)[:, 1])
            
            if optimize_thresh:
                if optimization_mode == 'recall':
                    # Maximize recall: low threshold, heavily penalize missed fraud
                    best_threshold, _ = optimize_threshold(y_val, y_val_proba, beta=6, cost_fn=50, cost_fp=1)
                else:
                    # Maximize precision: higher threshold, penalize false alarms
                    best_threshold, _ = optimize_threshold(y_val, y_val_proba, beta=0.5, cost_fn=1, cost_fp=50)
                rf.threshold_ = best_threshold
                y_val_pred = (y_val_proba >= best_threshold).astype(int)
            else:
                rf.threshold_ = 0.5
                y_val_pred = cp.asnumpy(rf.predict(X_val_gpu))
            
            print("\nValidation Performance for Random Forest:")
            print_metrics(y_val, y_val_pred, y_val_proba)
            
            # Compute GPU feature importances
            rf.feature_importances_ = cp.asnumpy(permutation_importance(rf, X_train_gpu, y_train_gpu))
            
            return rf
            
        except Exception as e:
            print(f"GPU training failed: {e}")
            print("Falling back to CPU...")
    
    # CPU-based Random Forest with hyperparameter tuning
    print("Using CPU-based scikit-learn Random Forest")
    print(f"Applied class weights: {class_weights}")
    
    param_grid = {
        'n_estimators': [300, 500, 800, 1000], # more trees - bru
        'max_depth': [20, 30, 40, None], # deeper trees
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED) # 5-Fold Cross Validation
    rf = RandomForestClassifier(class_weight=class_weights, random_state=SEED, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=stratified_kfold, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Validation predictions with threshold optimization
    y_val_proba = best_rf.predict_proba(X_val)[:, 1]
    
    if optimize_thresh:
        if optimization_mode == 'recall':
            # Maximize recall: low threshold, heavily penalize missed fraud
            best_threshold, _ = optimize_threshold(y_val, y_val_proba, beta=5, cost_fn=50, cost_fp=1)
        else:
            # Maximize precision: higher threshold, penalize false alarms
            best_threshold, _ = optimize_threshold(y_val, y_val_proba, beta=0.5, cost_fn=1, cost_fp=50)
        best_rf.threshold_ = best_threshold
        y_val_pred = (y_val_proba >= best_threshold).astype(int)
    else:
        best_rf.threshold_ = 0.5
        y_val_pred = best_rf.predict(X_val)
    
    print("\nValidation Performance for Random Forest:")
    print_metrics(y_val, y_val_pred, y_val_proba)
    
    return best_rf
        
def train_neural_network(X_train, y_train, X_val, y_val, use_gpu=True, optimize_thresh=True, optimization_mode='precision'):
    """Train Neural Network model"""
    print(f"\n== TRAINING NEURAL NETWORK (Optimized for {optimization_mode.upper()}) ==")
    
    # Try GPU-accelerated PyTorch neural network
    if use_gpu and GPU_INFO['cuda_available']:
        print("Using PyTorch NN")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Define Neural Network model with Batch Normalization and Dropout
            class FraudDetectionNN(nn.Module):
                def __init__(self, input_dim):
                    super(FraudDetectionNN, self).__init__()
                    # Larger architecture for better capacity
                    self.fc1 = nn.Linear(input_dim, 1024)
                    self.bn1 = nn.BatchNorm1d(1024)
                    self.dropout1 = nn.Dropout(0.5)
                    
                    self.fc2 = nn.Linear(1024, 512)
                    self.bn2 = nn.BatchNorm1d(512)
                    self.dropout2 = nn.Dropout(0.5)
                    
                    self.fc3 = nn.Linear(512, 256)
                    self.bn3 = nn.BatchNorm1d(256)
                    self.dropout3 = nn.Dropout(0.5)
                    
                    self.fc4 = nn.Linear(256, 128)
                    self.bn4 = nn.BatchNorm1d(128)
                    self.dropout4 = nn.Dropout(0.4)
                    
                    self.fc5 = nn.Linear(128, 64)
                    self.bn5 = nn.BatchNorm1d(64)
                    self.dropout5 = nn.Dropout(0.3)
                    
                    self.output = nn.Linear(64, 1)
                    self.relu = nn.ReLU()
                
                def forward(self, x):
                    # Layer 1: Linear -> BatchNorm -> ReLU -> Dropout
                    x = self.fc1(x)
                    x = self.bn1(x)
                    x = self.relu(x)
                    x = self.dropout1(x)
                    
                    # Layer 2
                    x = self.fc2(x)
                    x = self.bn2(x)
                    x = self.relu(x)
                    x = self.dropout2(x)
                    
                    # Layer 3
                    x = self.fc3(x)
                    x = self.bn3(x)
                    x = self.relu(x)
                    x = self.dropout3(x)
                    
                    # Layer 4
                    x = self.fc4(x)
                    x = self.bn4(x)
                    x = self.relu(x)
                    x = self.dropout4(x)
                    
                    # Layer 5
                    x = self.fc5(x)
                    x = self.bn5(x)
                    x = self.relu(x)
                    x = self.dropout5(x)
                    
                    # Output layer (no activation, BCEWithLogitsLoss includes sigmoid)
                    x = self.output(x)
                    return x
            
            # Prepare data
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1).to(device)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            
            # Initialize model
            model = FraudDetectionNN(X_train.shape[1]).to(device)
            
            # Focal Loss for hard example mining - adjusted based on optimization mode
            if optimization_mode == 'recall':
                # Higher alpha for recall - focus more on positive class
                focal_alpha = 0.75
                print("RECALL-FOCUSED: Using high alpha in Focal Loss to emphasize fraud detection")
            else:
                # Lower alpha for precision - balanced focus
                focal_alpha = 0.25
                print("PRECISION-FOCUSED: Using balanced alpha in Focal Loss for high-confidence predictions")
            
            class FocalLoss(nn.Module):
                def __init__(self, alpha=0.25, gamma=2.0):
                    super(FocalLoss, self).__init__()
                    self.alpha = alpha
                    self.gamma = gamma
                    
                def forward(self, inputs, targets):
                    bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
                    pt = torch.exp(-bce_loss)
                    focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
                    return focal_loss.mean()
            
            criterion = FocalLoss(alpha=focal_alpha, gamma=2.0)  # Focuses on hard-to-classify examples
            
            # Adam optimizer with gradient clipping and weight decay
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
            print(f"Model architecture: 1024 -> 512 -> 256 -> 128 -> 64 -> 1")
            print(f"Batch Normalization: Enabled on all hidden layers")
            print(f"Dropout rates: 0.5, 0.5, 0.5, 0.4, 0.3")
            
            # Training loop with improved early stopping
            best_val_loss = float('inf')
            patience = 10  # Increased patience for better convergence
            patience_counter = 0

            for epoch in range(200):  # More epochs with early stopping
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    # Gradient clipping for stable training
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/200], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
                
                # Early stopping with best model saving
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            class PyTorchModelWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, model=None, device='cpu'):
                    self.model = model
                    self.device = device
                    self.classes_ = np.array([0, 1])
                
                def get_params(self, deep=True):
                    return {'model': self.model, 'device': self.device}
                
                def set_params(self, **params):
                    for key, value in params.items():
                        setattr(self, key, value)
                    return self
                
                def fit(self, X, y):
                    # Already fitted, just return self
                    return self
                
                def predict(self, X):
                    if self.model is None:
                        raise ValueError("Model is not initialized")
                    self.model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X).to(self.device)
                        proba = self.model(X_tensor).cpu().numpy().flatten()
                    return (proba >= 0.5).astype(int)
                
                def predict_proba(self, X):
                    if self.model is None:
                        raise ValueError("Model is not initialized")
                    self.model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X).to(self.device)
                        proba = self.model(X_tensor).cpu().numpy().flatten()
                    return np.column_stack([1 - proba, proba])
            
            wrapped_model = PyTorchModelWrapper(model, device)
            
            # Validation predictions with threshold optimization
            y_val_proba = wrapped_model.predict_proba(X_val)[:, 1]
            
            if optimize_thresh:
                if optimization_mode == 'recall':
                    # Maximize recall: low threshold, heavily penalize missed fraud
                    best_threshold, _ = optimize_threshold(y_val, y_val_proba, beta=5, cost_fn=50, cost_fp=1)
                else:
                    # Maximize precision: higher threshold, penalize false alarms
                    best_threshold, _ = optimize_threshold(y_val, y_val_proba, beta=0.5, cost_fn=1, cost_fp=50)
                wrapped_model.threshold_ = best_threshold
                y_val_pred = (y_val_proba >= best_threshold).astype(int)
            else:
                wrapped_model.threshold_ = 0.5
                y_val_pred = wrapped_model.predict(X_val)
            
            print("\nValidation Performance for Neural Network:")
            print_metrics(y_val, y_val_pred, y_val_proba)
            
            return wrapped_model
            
        except Exception as e:
            print(f"GPU training failed: {e}")
            print("Falling back to CPU.")
    
    # CPU-based sklearn MLP with improved architecture
    print("Using CPU-based scikit-learn MLP Classifier")
    print("Architecture: 1024 -> 512 -> 256 -> 128 -> 64")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(1024, 512, 256, 128, 64),  # Larger network to match GPU version
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 regularization
        batch_size=256,
        learning_rate='adaptive',  # Adaptive learning rate
        learning_rate_init=0.001,
        max_iter=200,  # More iterations
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,  # Increased patience
        random_state=SEED,
        verbose=True
    )
    
    mlp.fit(X_train, y_train)
    
    # Validation predictions with threshold optimization
    y_val_proba = mlp.predict_proba(X_val)[:, 1]
    
    if optimize_thresh:
        if optimization_mode == 'recall':
            # Maximize recall: low threshold, heavily penalize missed fraud
            best_threshold, _ = optimize_threshold(y_val, y_val_proba, beta=5, cost_fn=50, cost_fp=1)
        else:
            # Maximize precision: higher threshold, penalize false alarms
            best_threshold, _ = optimize_threshold(y_val, y_val_proba, beta=0.5, cost_fn=1, cost_fp=50)
        mlp.threshold_ = best_threshold
        y_val_pred = (y_val_proba >= best_threshold).astype(int)
    else:
        mlp.threshold_ = 0.5
        y_val_pred = mlp.predict(X_val)
    
    print("\nValidation Performance for Neural Network:")
    print_metrics(y_val, y_val_pred, y_val_proba)
    
    return mlp

def train_xgboost(X_train, y_train, X_val, y_val, use_gpu=True, optimize_thresh=True, optimization_mode='precision'):
    """Train XGBoost model"""
    print(f"\n== TRAINING XGBOOST (Optimized for {optimization_mode.upper()}) ==")
    
    try:
        # Determine device
        if use_gpu and GPU_INFO['cuda_available']:
            print("Using GPU-accelerated XGBoost")
            device = 'cuda'
        else:
            print("Using CPU-based XGBoost")
            device = 'cpu'
        
        # Calculate scale_pos_weight based on optimization mode
        if optimization_mode == 'recall':
            # Higher weight for recall optimization
            scale_pos_weight = 50
            print("RECALL-FOCUSED: Using high scale_pos_weight to catch all fraud")
        else:
            # Lower weight for precision optimization
            scale_pos_weight = 10
            print("PRECISION-FOCUSED: Using moderate scale_pos_weight for high-confidence predictions")
        
        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'max_depth': [5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            device=device,
            tree_method='hist',
            scale_pos_weight=scale_pos_weight,
            random_state=SEED,
            eval_metric='logloss'
        )
        
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED) # 5-Fold Cross Validation
        
        print(f"scale_pos_weight: {scale_pos_weight}")
        print("Tuning hyperparameters...")
        
        grid_search = GridSearchCV(
            xgb_model, 
            param_grid, 
            cv=stratified_kfold, 
            scoring='f1', 
            n_jobs=-1 if device == 'cpu' else 1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_xgb = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Validation predictions with threshold optimization
        y_val_proba = best_xgb.predict_proba(X_val)[:, 1]
        
        if optimize_thresh:
            if optimization_mode == 'recall':
                # Maximize recall: low threshold, heavily penalize missed fraud
                best_threshold, _ = optimize_threshold(y_val, y_val_proba, beta=5, cost_fn=50, cost_fp=1)
            else:
                # Maximize precision: higher threshold, penalize false alarms
                best_threshold, _ = optimize_threshold(y_val, y_val_proba, beta=0.5, cost_fn=1, cost_fp=50)
            best_xgb.threshold_ = best_threshold
            y_val_pred = (y_val_proba >= best_threshold).astype(int)
        else:
            best_xgb.threshold_ = 0.5
            y_val_pred = best_xgb.predict(X_val)
        
        print("\nValidation Performance for XGBoost:")
        print_metrics(y_val, y_val_pred, y_val_proba)
        
        return best_xgb
        
    except ImportError:
        print("XGBoost not installed. Skipping XGBoost training.")
        print("   Install with: pip install xgboost")
        return None
    except Exception as e:
        print(f"XGBoost training failed: {e}")
        return None

def train_ensemble_stacking(base_models, X_train, y_train, X_val, y_val, optimize_thresh=True):
    """Train weighted ensemble combining recall and precision models"""
    print("\n== TRAINING WEIGHTED ENSEMBLE ==")
    print("Stage 1: LR + RF for recall")
    print("Stage 2: NN + XGBoost for precision")
    print("Weights: 70% recall + 30% precision")
    
    # Filter out None models
    valid_models = {name: model for name, model in base_models if model is not None}
    
    if len(valid_models) < 2:
        print("Need at least 2 base models for ensemble. Skipping.")
        return None
    
    print(f"Using {len(valid_models)} base models: {list(valid_models.keys())}")
    
    # Separate models by role
    recall_models = {}
    precision_models = {}
    
    if 'lr' in valid_models:
        recall_models['lr'] = valid_models['lr']
    if 'rf' in valid_models:
        recall_models['rf'] = valid_models['rf']
    if 'nn' in valid_models:
        precision_models['nn'] = valid_models['nn']
    if 'xgb' in valid_models:
        precision_models['xgb'] = valid_models['xgb']
    
    print(f"\nRecall specialists: {list(recall_models.keys())}")
    print(f"Precision specialists: {list(precision_models.keys())}")
    
    # Helper to get predictions (handle GPU models)
    def get_proba(model, X):
        if hasattr(model, '__module__') and 'cuml' in model.__module__:
            X_gpu = cp.array(X)
            proba = cp.asnumpy(model.predict_proba(X_gpu)[:, 1])
        else:
            proba = model.predict_proba(X)[:, 1]
        return proba
    
    # Stage 1: Recall models (OR logic - maximum probability)
    print("\n--- Stage 1: Recall Detection ---")
    recall_probas = []
    for name, model in recall_models.items():
        proba = get_proba(model, X_val)
        recall_probas.append(proba)
        fraud_detected = (proba >= model.threshold_).sum()
        print(f"{name.upper()}: Detected {fraud_detected} potential fraud cases (threshold={model.threshold_:.3f})")
    
    # Take maximum probability from recall models (OR logic - if ANY model flags it)
    if recall_probas:
        stage1_proba = np.maximum.reduce(recall_probas)
        stage1_detected = (stage1_proba >= 0.5).sum()
        print(f"Stage 1 Total: {stage1_detected} cases flagged for validation")
    else:
        stage1_proba = np.zeros(len(X_val))
    
    # Stage 2: Precision models (weighted validation)
    print("\n--- Stage 2: Precision Validation ---")
    precision_probas = []
    for name, model in precision_models.items():
        proba = get_proba(model, X_val)
        precision_probas.append(proba)
        high_conf = (proba >= model.threshold_).sum()
        print(f"{name.upper()}: Confirmed {high_conf} high-confidence fraud cases (threshold={model.threshold_:.3f})")
    
    # Average precision model probabilities (both must have high confidence)
    if precision_probas:
        stage2_proba = np.mean(precision_probas, axis=0)
    else:
        stage2_proba = np.ones(len(X_val))
    
    # Combine stages
    print("\n--- Combining Stages ---")
    if recall_probas and precision_probas:
        recall_weight = 0.70
        precision_weight = 0.30
        y_val_proba = (recall_weight * stage1_proba) + (precision_weight * stage2_proba)
        print(f"P(fraud) = {recall_weight}*P(recall) + {precision_weight}*P(precision)")
    elif recall_probas:
        # Only recall models available
        y_val_proba = stage1_proba
        print("Using only recall models (no precision models available)")
    elif precision_probas:
        # Only precision models available
        y_val_proba = stage2_proba
        print("Using only precision models (no recall models available)")
    else:
        print("No valid models found!")
        return None
    
    if optimize_thresh:
        # Optimize for F2 (emphasize recall but maintain precision)
        best_threshold, _ = optimize_threshold(y_val, y_val_proba, beta=2.0, cost_fp=5, cost_fn=20, optimize_for='f_beta')
        y_val_pred = (y_val_proba >= best_threshold).astype(int)
    else:
        best_threshold = 0.5
        y_val_pred = (y_val_proba >= best_threshold).astype(int)
    
    # Create weighted ensemble wrapper
    class WeightedEnsemble:
        def __init__(self, recall_models, precision_models, threshold):
            self.recall_models = recall_models
            self.precision_models = precision_models
            self.threshold_ = threshold
        
        def predict_proba(self, X):
            # Stage 1: Recall detection (OR logic)
            recall_probas = []
            for name, model in self.recall_models.items():
                if hasattr(model, '__module__') and 'cuml' in model.__module__:
                    X_gpu = cp.array(X)
                    proba = cp.asnumpy(model.predict_proba(X_gpu)[:, 1])
                else:
                    proba = model.predict_proba(X)[:, 1]
                recall_probas.append(proba)
            
            stage1_proba = np.maximum.reduce(recall_probas) if recall_probas else np.zeros(len(X))
            
            # Stage 2: Precision validation
            precision_probas = []
            for name, model in self.precision_models.items():
                if hasattr(model, '__module__') and 'cuml' in model.__module__:
                    X_gpu = cp.array(X)
                    proba = cp.asnumpy(model.predict_proba(X_gpu)[:, 1])
                else:
                    proba = model.predict_proba(X)[:, 1]
                precision_probas.append(proba)
            
            stage2_proba = np.mean(precision_probas, axis=0) if precision_probas else np.ones(len(X))
            
            # Weighted combination (70% recall, 30% precision)
            if recall_probas and precision_probas:
                final_proba = (0.70 * stage1_proba) + (0.30 * stage2_proba)
            elif recall_probas:
                final_proba = stage1_proba
            else:
                final_proba = stage2_proba
            
            return np.column_stack([1 - final_proba, final_proba])
        
        def predict(self, X):
            proba = self.predict_proba(X)[:, 1]
            return (proba >= self.threshold_).astype(int)
    
    ensemble = WeightedEnsemble(recall_models, precision_models, best_threshold)
    
    print("\nValidation Performance for Weighted Ensemble:")
    print_metrics(y_val, y_val_pred, y_val_proba)
    
    return ensemble


# 7. MODEL EVALUATION

def print_metrics(y_true, y_pred, y_proba):
    """Print comprehensive evaluation metrics for a model"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraudulent']))

def evaluate_models(models, X_test, y_test, model_names):
    """Evaluate models on test set"""
    print("\nFINAL TEST SET EVALUATION")
    
    results = {}
    
    for model, name in zip(models, model_names):
        if model is None:
            print(f"\nSkipping {name} (model not available)")
            continue
            
        print(f"\n== FINAL EVALUATION: {name.upper()} ==")
        
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Use optimized threshold if available
        if hasattr(model, 'threshold_'):
            threshold = model.threshold_
            print(f"Using optimized threshold: {threshold:.3f}")
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)
        
        print_metrics(y_test, y_pred, y_proba)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'f2': fbeta_score(y_test, y_pred, beta=2),
            'auc': roc_auc_score(y_test, y_proba),
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    return results

def analyze_shap_values(model, X_test, feature_names, model_name, version, max_display=20):
    """Generate SHAP plots for model interpretability"""
    print(f"\nSHAP ANALYSIS for {model_name}")
    
    try:
        # For tree-based models, use TreeExplainer
        if 'Random Forest' in model_name or 'XGBoost' in model_name:
            print("Using TreeExplainer...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            # For binary classification, get positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # For other models, use KernelExplainer (slower but works for all models)
            print("Using KernelExplainer (this may take a while)...")
            # Use a sample of training data as background
            background = shap.sample(X_test, min(100, len(X_test)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_test[:100])  # Analyze first 100 samples
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                         max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(f'shap_summary_{model_name.replace(" ", "_")}_{version}.png', 
                   bbox_inches='tight', dpi=150)
        print(f"Saved SHAP summary plot to 'shap_summary_{model_name.replace(' ', '_')}_{version}.png'")
        plt.close()
        
        # Feature importance bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                         plot_type="bar", max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(f'shap_importance_{model_name.replace(" ", "_")}_{version}.png',
                   bbox_inches='tight', dpi=150)
        print(f"Saved SHAP importance plot to 'shap_importance_{model_name.replace(' ', '_')}_{version}.png'")
        plt.close()
        
        return shap_values, explainer
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return None, None


# 8. VISUALIZATION

def plot_roc_curves(results, y_test, version):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'roc_curves_{version}.png', bbox_inches='tight')
    print(f"Saved ROC curves to 'roc_curves_{version}.png'")
    plt.close()

def plot_precision_recall_curves(results, y_test, version):
    """Plot Precision-Recall curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_test, result['y_proba'])
        plt.plot(recall, precision, label=f"{name}", linewidth=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'precision_recall_curves_{version}.png', bbox_inches='tight')
    print(f"Saved Precision-Recall curves to 'precision_recall_curves_{version}.png'")
    plt.close()

def plot_feature_importance(rf_model, feature_names, version, top_n=15):
    """Plot feature importance from Random Forest"""
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top Feature Importances (Random Forest) - {version.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{version}.png', bbox_inches='tight')
    print(f"Saved feature importance plot to 'feature_importance_{version}.png'")
    plt.close()

def plot_confusion_matrices(results, y_test, version):
    """Plot confusion matrices for all models"""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    # Handle single model case
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['y_pred'])
        seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                   cbar=False, square=True)
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_xticklabels(['Legitimate', 'Fraudulent'])
        axes[idx].set_yticklabels(['Legitimate', 'Fraudulent'])
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrices_{version}.png', bbox_inches='tight')
    print(f"Saved confusion matrices to 'confusion_matrices_{version}.png'")
    plt.close()

def plot_model_comparison(results, version):
    """Plot bar chart comparing model metrics"""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'f2', 'auc']
    model_names = list(results.keys())
    
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)  # Dynamic width based on number of models
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for idx, name in enumerate(model_names):
        values = [results[name][metric] for metric in metrics]
        ax.bar(x + idx*width, values, width, label=name)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison (Weighted Ensemble Strategy)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(model_names)-1) / 2)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'F2-Score', 'AUC-ROC'])
    ax.legend(loc='best')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'model_comparison_{version}.png', bbox_inches='tight')
    print(f"Saved model comparison to 'model_comparison_{version}.png'")
    plt.close()

# 9. MAIN EXECUTION

def main(version='v1'):
    """Run fraud detection pipeline"""
    
    print("\nE-COMMERCE FRAUD DETECTION SYSTEM")
    print("Group 34 - COMPSCI 4AL3")
    
    # 1. Load data
    print(f"\nLoading dataset from Kaggle (Version: {version.upper()})")
    print("   V1: Large dataset (~1.47M rows)")
    print("   V2: Small dataset (~23K rows)")
    df = load_data_from_kaggle(version=version)
    
    # Analyze Customer Location for data leakage
    analyze_customer_location_leakage(df)
    
    # 2. Feature engineering
    df = engineer_features_gpu(df) if GPU_INFO['cuda_available'] else engineer_features(df)
    
    # 3. Preprocess data
    X, y, = preprocess_data(df, label_encoder=False)
    
    # 4. Split and scale data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale_data(X, y)

    # 5. Apply ADASYN with full balance for optimal precision/recall/AUC
    try:
        X_train_balanced, y_train_balanced = apply_adasyn(X_train, y_train, sampling_ratio=1.0)
    except Exception as e:
        print(f"Resampling failed: {e}, trying hybrid sampling...")
        X_train_balanced, y_train_balanced = apply_hybrid_sampling(X_train, y_train, target_ratio=1.0)

    # 6. Train models
    print("\nENSEMBLE STRATEGY")
    print("Recall models: LR, RF")
    print("Precision models: NN, XGBoost")
    print("Combination: 70% recall + 30% precision")
    
    lr_model = train_logistic_regression(X_train_balanced, y_train_balanced, X_val, y_val, use_gpu=True, optimize_thresh=True, optimization_mode='recall')
    rf_model = train_random_forest(X_train_balanced, y_train_balanced, X_val, y_val, use_gpu=True, optimize_thresh=True, optimization_mode='recall')
    nn_model = train_neural_network(X_train_balanced, y_train_balanced, X_val, y_val, optimize_thresh=True, optimization_mode='precision')
    xgb_model = train_xgboost(X_train_balanced, y_train_balanced, X_val, y_val, optimize_thresh=True, optimization_mode='precision')
    
    # 7. Train ensemble stacking
    base_models = [
        ('lr', lr_model),
        ('rf', rf_model),
        ('nn', nn_model),
        ('xgb', xgb_model)
    ]
    ensemble_model = train_ensemble_stacking(base_models, X_train_balanced, y_train_balanced, X_val, y_val, optimize_thresh=True)
    
    # 8. Evaluate models on test set
    models = [lr_model, rf_model, nn_model, xgb_model, ensemble_model]
    model_names = ['Logistic Regression', 'Random Forest', 'Neural Network', 'XGBoost', 'Stacking Ensemble']
    results = evaluate_models(models, X_test, y_test, model_names)
    
    # 9. Generate visualizations
    plot_roc_curves(results, y_test, version)
    plot_precision_recall_curves(results, y_test, version)
    if rf_model is not None:
        plot_feature_importance(rf_model, X.columns.tolist(), version)
    plot_confusion_matrices(results, y_test, version)
    plot_model_comparison(results, version)
    
    # 10. SHAP analysis for interpretability
    if xgb_model is not None:
        analyze_shap_values(xgb_model, X_test, X.columns.tolist(), 'XGBoost', version)
    if rf_model is not None:
        analyze_shap_values(rf_model, X_test, X.columns.tolist(), 'Random Forest', version)
    if nn_model is not None:
        analyze_shap_values(nn_model, X_test, X.columns.tolist(), 'Neural Network', version)
    if lr_model is not None:
        analyze_shap_values(lr_model, X_test, X.columns.tolist(), 'Logistic Regression', version)
    if ensemble_model is not None:
        analyze_shap_values(ensemble_model, X_test, X.columns.tolist(), 'Stacking Ensemble', version)
    
    print("\n=== OUTPUT FILES ===")
    print("Plots: roc_curves.png, precision_recall_curves.png, feature_importance.png")
    print("       confusion_matrices.png, model_comparison.png")
    print("SHAP: shap_summary_*.png, shap_importance_*.png")
    
    return models, results

models, results = main(
            version='v1'
        )