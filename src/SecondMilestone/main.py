"""
E-Commerce Fraud Detection Learning
COMPSCI 4AL3 - Group 34
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, precision_score, recall_score, accuracy_score)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

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
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['gpu_available'] = True
            gpu_info['device'] = 'cuda'
            gpu_info['gpu_name'] = torch.cuda.get_device_name(0)
            print(f"CUDA GPU detected: {gpu_info['gpu_name']}")
    except ImportError:
        pass
    
    # Check for CuPy (NumPy-compatible GPU arrays)
    try:
        import cupy
        gpu_info['cupy_available'] = True
        gpu_info['gpu_available'] = True
        print("CuPy available for GPU-accelerated NumPy operations")
    except ImportError:
        pass
    
    # Check for RAPIDS cuML (GPU-accelerated scikit-learn)
    try:
        import cuml
        gpu_info['rapids_available'] = True
        gpu_info['gpu_available'] = True
        print("RAPIDS cuML available for GPU-accelerated ML")
    except ImportError:
        pass
    
    return gpu_info

# Initialize GPU detection
GPU_INFO = detect_gpu()


# 1. DATA LOADING

def load_data_from_kaggle(version='v1', cache_dir=None):
    """Load data (either v1 or v2) from Kaggle from cache or download if needed"""
    import os
    
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
            import kagglehub
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

# def exploratory_data_analysis(df):
#     """Print out data analysis"""
#     print("\nDATA ANALYSIS")
    
#     # Numerical features statistics
#     numerical_cols = ['Transaction Amount', 'Quantity', 'Customer Age', 
#                      'Account Age Days', 'Transaction Hour']
#     print("\nNumerical Features Statistics:")
#     print(df[numerical_cols].describe())
    
#     # Categorical features
#     categorical_cols = ['Payment Method', 'Product Category', 'Device Used']
#     print("\nCategorical Features Distribution:")
#     for col in categorical_cols:
#         if col in df.columns:
#             print(f"\n{col}:")
#             print(df[col].value_counts())
    
#     return df


# 2. FEATURE ENGINEERING

def engineer_features(df):
    """Create new features from existing data"""
    print("\nFEATURE ENGINEERING")
    
    df = df.copy()
    
    # Total Customer Transactions
    customer_transaction_counts = df.groupby('Customer ID')['Transaction ID'].nunique()
    df['Total Customer Transactions'] = df['Customer ID'].map(customer_transaction_counts)
    
    # Address Mismatch Flag
    df['Address Mismatch'] = (df['Shipping Address'] != df['Billing Address']).astype(int)
    
    # Time-Related (Temporal) Features Created from Transaction Date
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    df['Day of Week'] = df['Transaction Date'].dt.dayofweek
    df['Month'] = df['Transaction Date'].dt.month
    df['Is Weekend'] = (df['Day of Week'] >= 5).astype(int)
    
    # New Account Flag
    df['New Account'] = (df['Account Age Days'] < 30).astype(int)
    
    # Transaction Amount Statistics per Customer
    customer_stats = df.groupby('Customer ID')['Transaction Amount'].agg(['mean', 'std', 'max'])
    df['Avg Customer Transaction'] = df['Customer ID'].map(customer_stats['mean'])
    df['Transaction Amount Ratio'] = df['Transaction Amount'] / (df['Avg Customer Transaction'] + 1)
    
    # High Value Transaction Flag
    df['High Value Transaction'] = (df['Transaction Amount'] > df['Transaction Amount'].quantile(0.95)).astype(int)
    
    print(f"\nTotal of {df.shape[1]-1} features after engineering:\n{list(df.columns)}")
    return df


# 3. DATA PREPROCESSING

def preprocess_data(df, label_encoder=False):
    """Preprocess data for modeling with label encoding or one-hot encoding and target encoding"""
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
        'Transaction Amount Ratio', 'High Value Transaction',
        # Categorical features (to be encoded)
        'Payment Method', 'Product Category', 'Device Used', 'Customer Location'
    ]
    
    # Check for available columns (this is used for testing, since we edit code a lot)
    features_to_use = [f for f in features_to_use if f in df.columns]
    
    # Create feature dataframe
    X = df[features_to_use].copy()
    y = df['Is Fraudulent'].copy()
    
    print(f"Features used: {len(features_to_use)}")
    print(f"Feature list: {features_to_use}")
    
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
        # One-hot encode categorical features
        one_hot_cols = ['Payment Method', 'Product Category', 'Device Used']
    
        for col in one_hot_cols:
            if col in X.columns:
                X = pd.get_dummies(X, columns=[col], drop_first=True)
        
        # Target encode Customer Location
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
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: separate validation set from remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"Train set size: {X_train.shape[0]} ({len(y_train[y_train==1])} fraudulent)")
    print(f"Validation set size: {X_val.shape[0]} ({len(y_val[y_val==1])} fraudulent)")
    print(f"Test set size: {X_test.shape[0]} ({len(y_test[y_test==1])} fraudulent)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


# 5. HANDLE CLASS IMBALANCE WITH SMOTE

def apply_smote(X_train, y_train):
    """Apply SMOTE to balance training data"""
    print("\nAPPLYING SMOTE FOR CLASS BALANCE")

    print(f"Before SMOTE - Fraudulent: {sum(y_train)}, Legitimate: {len(y_train) - sum(y_train)}")

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE - Fraudulent: {sum(y_train_balanced)}, Legitimate: {len(y_train_balanced) - sum(y_train_balanced)}")

    return X_train_balanced, y_train_balanced


# 6. MODEL TRAINING

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train Logistic Regression model"""
    print("\n== TRAINING LOGISTIC REGRESSION ==")
    
    # Hyperparameter tuning
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear']
    }
    
    lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    grid_search = GridSearchCV(lr, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_lr = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Validation predictions
    y_val_pred = best_lr.predict(X_val)
    y_val_prob = best_lr.predict_proba(X_val)[:, 1]
    
    print("\nValidation Performance for Logistic Regression:")
    print_metrics(y_val, y_val_pred, y_val_prob)
    
    return best_lr

def permutation_importance(model, X, y, metric=accuracy_score):
    """Compute feature importances using permutation importance on GPU"""
    import cupy as cp
    base_score = metric(cp.asnumpy(y), cp.asnumpy(model.predict(X)))
    importances = cp.zeros(X.shape[1], dtype=cp.float32)

    for i in range(X.shape[1]):
        X_perm = X.copy()
        X_perm[:, i] = cp.random.permutation(X_perm[:, i])
        perm_score = metric(cp.asnumpy(y), cp.asnumpy(model.predict(X_perm)))
        importances[i] = base_score - perm_score

    return importances / importances.sum()

def train_random_forest(X_train, y_train, X_val, y_val, use_gpu=True):
    """Train Random Forest model with optional GPU acceleration via CuPy"""
    print("\n== TRAINING RANDOM FOREST ==")
    
    # Use RAPIDS cuML Random Forest if available and requested
    if use_gpu and GPU_INFO['rapids_available']:
        print("Using GPU-accelerated RAPIDS cuML Random Forest")
        try:
            from cuml.ensemble import RandomForestClassifier as cuRF
            import cupy as cp
            
            # Convert to GPU arrays
            X_train_gpu = cp.array(X_train)
            y_train_gpu = cp.array(y_train)
            X_val_gpu = cp.array(X_val)
            
            # Train on GPU
            rf = cuRF(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_streams=4  # Parallel GPU streams
            )
            
            rf.fit(X_train_gpu, y_train_gpu)
            
            # Validation predictions
            y_val_pred = cp.asnumpy(rf.predict(X_val_gpu))
            y_val_proba = cp.asnumpy(rf.predict_proba(X_val_gpu)[:, 1])
            
            print("\nValidation Performance for Random Forest:")
            print_metrics(y_val, y_val_pred, y_val_proba)
            
            # Compute GPU feature importances
            rf.feature_importances_ = cp.asnumpy(permutation_importance(rf, X_train_gpu, y_train_gpu))
            
            return rf
            
        except Exception as e:
            print(f"⚠️  GPU training failed: {e}")
            print("Falling back to CPU...")
    
    # CPU-based Random Forest with hyperparameter tuning
    print("Using CPU-based scikit-learn Random Forest")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Validation predictions
    y_val_pred = best_rf.predict(X_val)
    y_val_proba = best_rf.predict_proba(X_val)[:, 1]
    
    print("\nValidation Performance for Random Forest:")
    print_metrics(y_val, y_val_pred, y_val_proba)
    
    return best_rf

def train_neural_network(X_train, y_train, X_val, y_val, use_gpu=True):
    """Train Neural Network model with optional GPU acceleration via PyTorch"""
    print("\n== TRAINING NEURAL NETWORK ==")
    
    # Try GPU-accelerated PyTorch neural network
    if use_gpu and GPU_INFO['cuda_available']:
        print("Using GPU-accelerated PyTorch Neural Network")
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Define Neural Network model
            class FraudDetectionNN(nn.Module):
                def __init__(self, input_dim):
                    super(FraudDetectionNN, self).__init__()
                    self.fc1 = nn.Linear(input_dim, 128)
                    self.dropout1 = nn.Dropout(0.3)
                    self.fc2 = nn.Linear(128, 64)
                    self.dropout2 = nn.Dropout(0.3)
                    self.fc3 = nn.Linear(64, 1)
                    self.relu = nn.ReLU()
                    self.sigmoid = nn.Sigmoid()
                
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.dropout1(x)
                    x = self.relu(self.fc2(x))
                    x = self.dropout2(x)
                    x = self.sigmoid(self.fc3(x))
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
            pos_weight = torch.tensor([ (len(y_train) - sum(y_train)) / sum(y_train) ]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            best_val_loss = float('inf')
            patience = 5
            patience_counter = 0

            for epoch in range(100):
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/100], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
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
            
            # Create wrapper class for using sklearn tools
            class PyTorchModelWrapper:
                def __init__(self, model, device):
                    self.model = model
                    self.device = device
                    self.classes_ = np.array([0, 1])
                
                def predict(self, X):
                    self.model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X).to(self.device)
                        proba = self.model(X_tensor).cpu().numpy().flatten()
                    return (proba >= 0.5).astype(int)
                
                def predict_proba(self, X):
                    self.model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X).to(self.device)
                        proba = self.model(X_tensor).cpu().numpy().flatten()
                    return np.column_stack([1 - proba, proba])
            
            wrapped_model = PyTorchModelWrapper(model, device)
            
            # Validation predictions
            y_val_pred = wrapped_model.predict(X_val)
            y_val_proba = wrapped_model.predict_proba(X_val)[:, 1]
            
            print("\nValidation Performance for Neural Network:")
            print_metrics(y_val, y_val_pred, y_val_proba)
            
            return wrapped_model
            
        except Exception as e:
            print(f"GPU training failed: {e}")
            print("Falling back to CPU.")
    
    # CPU-based sklearn MLP
    print("Using CPU-based scikit-learn MLP Classifier")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=256,
        learning_rate_init=0.001,
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        verbose=True
    )
    
    mlp.fit(X_train, y_train)
    
    # Validation predictions
    y_val_pred = mlp.predict(X_val)
    y_val_proba = mlp.predict_proba(X_val)[:, 1]
    
    print("\nValidation Performance for Neural Network:")
    print_metrics(y_val, y_val_pred, y_val_proba)
    
    return mlp


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
    """Evaluate multiple models on test set"""
    print("\nFINAL TEST SET EVALUATION")
    
    results = {}
    
    for model, name in zip(models, model_names):
        print(f"\n= FINAL EVALUATION: {name.upper()} =")
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        print_metrics(y_test, y_pred, y_proba)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba),
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    return results


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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
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
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    model_names = list(results.keys())
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, name in enumerate(model_names):
        values = [results[name][metric] for metric in metrics]
        ax.bar(x + idx*width, values, width, label=name)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'model_comparison_{version}.png', bbox_inches='tight')
    print(f"Saved model comparison to 'model_comparison_{version}.png'")
    plt.close()

# 9. MAIN EXECUTION

def main(version='v1'):
    """Main execution that runs the entire E-COMMERCE FRAUD DETECTION SYSTEM experiment"""
    
    print("\nE-COMMERCE FRAUD DETECTION SYSTEM")
    print("Group 34 - COMPSCI 4AL3")
    
    # 1. Load data
    print(f"\nLoading dataset from Kaggle (Version: {version.upper()})")
    print("   V1: Large dataset (~1.47M rows)")
    print("   V2: Small dataset (~23K rows)")
    df = load_data_from_kaggle(version=version)
    
    # Exploratory data analysis
    # df = exploratory_data_analysis(df)
    
    # 2. Feature engineering
    df = engineer_features(df)
    
    # 3. Preprocess data
    X, y, = preprocess_data(df, label_encoder=False)
    
    # 4. Split and scale data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale_data(X, y)

    # 5. Apply SMOTE to training data
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    # 6. Train models
    lr_model = train_logistic_regression(X_train_balanced, y_train_balanced, X_val, y_val)
    rf_model = train_random_forest(X_train_balanced, y_train_balanced, X_val, y_val)
    nn_model = train_neural_network(X_train_balanced, y_train_balanced, X_val, y_val)
    
    # 7. Evaluate models on test set
    models = [lr_model, rf_model, nn_model]
    model_names = ['Logistic Regression', 'Random Forest', 'Neural Network']
    results = evaluate_models(models, X_test, y_test, model_names)
    
    # 8. Generate visualizations
    plot_roc_curves(results, y_test, version)
    plot_precision_recall_curves(results, y_test, version)
    plot_feature_importance(rf_model, X.columns.tolist(), version)
    plot_confusion_matrices(results, y_test, version)
    plot_model_comparison(results, version)
    
    print("\nGenerated files:")
    print("  - roc_curves.png")
    print("  - precision_recall_curves.png")
    print("  - feature_importance.png")
    print("  - confusion_matrices.png")
    print("  - model_comparison.png")
    
    return models, results

models, results = main(
            version='v2'
        )
