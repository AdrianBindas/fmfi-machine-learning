import numpy as np
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    balanced_accuracy_score
)
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from feature_analysis import analyze_feature_importance


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for the training pipeline."""
    
    # Data paths - Labels (CSV files)
    TRAIN_LABELS = Path("data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")
    VALID_LABELS = Path("data/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv")
    TEST_LABELS = Path("data/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv")
    
    # Data paths - Segmentation features (parquet)
    TRAIN_SEG_FEATURES = Path("train_features/train_segmentation_features.parquet")
    VALID_SEG_FEATURES = Path("validation_features/valid_segmentation_features.parquet")
    TEST_SEG_FEATURES = Path("test_features/test_segmentation_features.parquet")
    
    # Data paths - ResNet features (parquet)
    TRAIN_RESNET_FEATURES = Path("train_features/features.parquet")
    VALID_RESNET_FEATURES = Path("validation_features/features.parquet")
    TEST_RESNET_FEATURES = Path("test_features/features.parquet")
    
    # Output directory
    OUTPUT_DIR = Path("results")
    
    # Feature selection
    # Options: 'all', 'segmentation', 'resnet', 'custom'
    FEATURE_MODE = 'all'
    
    # Custom feature columns (if FEATURE_MODE == 'custom')
    CUSTOM_FEATURES = [
        'area', 'perimeter', 'circularity', 'solidity', 
        'gray_variance', 'color_variance'
    ]
    
    # Columns with list/array values that need to be expanded
    # Format: {'column_name': 'prefix_for_expanded_columns'}
    LIST_COLUMNS = {
        'color_histogram_g': 'hist_g',
        'color_histogram_b': 'hist_b',
        'color_histogram_r': 'hist_r',
        'features': 'resnet'
    }
    
    # Label columns (multi-class classification)
    LABEL_COLUMNS = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    METADATA_COLUMS = ['sample_id', 'image', 'image_path', 'img_paths', 'image_name', 'clahe', 'median_blur', 'otsu', 'morph', 'contour', 'convex_hull']
    
    # Join column name (column to merge on)
    JOIN_COLUMN = 'sample_id'
    
    # PCA configuration
    USE_PCA = True
    PCA_N_COMPONENTS = 0.95  # Number of components or variance ratio (e.g., 0.95)
    PCA_VARIANCE_THRESHOLD = 0.95  # Alternative: use 0.95 to keep 95% variance
    
    # Class balancing
    BALANCE_CLASSES = True  # Whether to balance training classes
    BALANCE_STRATEGY = 'oversample'  # 'undersample' or 'oversample'

    # Model configurations
    RANDOM_STATE = 42
    
    # Decision Tree
    DT_MAX_DEPTH = 20
    DT_MIN_SAMPLES_SPLIT = 100
    DT_MIN_SAMPLES_LEAF = 10
    
    # Logistic Regression
    LR_MAX_ITER = 1000
    LR_C = 0.01
    
    # Neural Network
    NN_HIDDEN_SIZES = [128, 64, 32]
    NN_DROPOUT = 0.2
    NN_LEARNING_RATE = 0.0001
    NN_BATCH_SIZE = 64
    NN_EPOCHS = 100
    NN_EARLY_STOPPING_PATIENCE = 20
    
    # Device for PyTorch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def expand_list_columns(df, list_columns_config):
    """
    Expand columns containing lists/arrays into multiple columns.
    
    Args:
        df: DataFrame
        list_columns_config: Dict mapping column names to prefixes
    
    Returns:
        DataFrame with expanded columns
    """
    df_expanded = df.clone()
    
    for col_name, prefix in list_columns_config.items():
        if col_name not in df.columns:
            print(f"  Warning: Column '{col_name}' not found, skipping")
            continue
        
        print(f"  Expanding '{col_name}' column...")
        
        # Get first non-null value to determine array length
        sample_val = df[col_name].drop_nans()[0]
        
        # Handle different types of list representations
        if isinstance(sample_val, str):
            # If stored as string representation, convert
            import ast
            sample_array = np.array(ast.literal_eval(sample_val))
        elif isinstance(sample_val, (list, np.ndarray)):
            sample_array = np.array(sample_val)
        elif isinstance(sample_val, pl.series.series.Series):
            sample_array = sample_val.to_numpy()
        else:
            print(f"    Warning: Unknown type {type(sample_val)}, skipping")
            continue
        
        n_features = len(sample_array)
        print(f"    Creating {n_features} columns: {prefix}_0 to {prefix}_{n_features-1}")
        
        # Convert all values to arrays
        arrays = []
        for val in df[col_name]:
            arrays.append(val)
        
        # Create new columns
        arrays_stacked = np.vstack(arrays)
        for i in range(n_features):
            df_expanded = df_expanded.with_columns(pl.Series(name=f'{prefix}_{i}', values=arrays_stacked[:, i]))
        
        # Drop original column
        df_expanded = df_expanded.drop([col_name])
    
    return df_expanded


def load_and_merge_data(labels_path, seg_features_path, resnet_features_path, 
                        join_column, list_columns_config):
    """
    Load labels and feature files, merge them, and expand list columns.
    
    Args:
        labels_path: Path to label CSV
        seg_features_path: Path to segmentation features parquet
        resnet_features_path: Path to ResNet features parquet
        join_column: Column name to join on
        list_columns_config: Dict for expanding list columns
    
    Returns:
        Merged pandas DataFrame
    """
    print(f"  Loading labels from {labels_path.name}...")
    labels_df = pl.read_csv(labels_path)
    
    print(f"  Loading segmentation features from {seg_features_path.name}...")
    seg_df = pl.read_parquet(seg_features_path)
    
    print(f"  Loading ResNet features from {resnet_features_path.name}...")
    resnet_df = pl.read_parquet(resnet_features_path)
    resnet_df = resnet_df.rename({'indices': 'sample_id'})
    
    print(f"  Merging on '{join_column}' column...")
    # Merge labels with segmentation features
    merged_df = labels_df.join(seg_df, on='image', how='inner')
    
    # Merge with ResNet features
    merged_df = merged_df.join(resnet_df, on='sample_id', how='inner')
    
    print(f"  Merged shape: {merged_df.shape}")
    print(f"  Columns: {len(merged_df.columns)}")
    
    # Expand list columns
    if list_columns_config:
        print(f"  Expanding list-valued columns...")
        merged_df = expand_list_columns(merged_df, list_columns_config)
        print(f"  After expansion: {merged_df.shape}")
    
    return merged_df


def load_data(config):
    """
    Load train, validation, and test datasets.
    
    Args:
        config: Config object
    
    Returns:
        Tuple of (train_df, valid_df, test_df)
    """
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load training data
    print("\nTraining set:")
    train_df = load_and_merge_data(
        config.TRAIN_LABELS,
        config.TRAIN_SEG_FEATURES,
        config.TRAIN_RESNET_FEATURES,
        config.JOIN_COLUMN,
        config.LIST_COLUMNS
    )
    
    # Load validation data
    print("\nValidation set:")
    valid_df = load_and_merge_data(
        config.VALID_LABELS,
        config.VALID_SEG_FEATURES,
        config.VALID_RESNET_FEATURES,
        config.JOIN_COLUMN,
        config.LIST_COLUMNS
    )
    
    # Load test data
    print("\nTest set:")
    test_df = load_and_merge_data(
        config.TEST_LABELS,
        config.TEST_SEG_FEATURES,
        config.TEST_RESNET_FEATURES,
        config.JOIN_COLUMN,
        config.LIST_COLUMNS
    )
    
    print("\n" + "="*60)
    print(f"Train: {train_df.shape}")
    print(f"Valid: {valid_df.shape}")
    print(f"Test: {test_df.shape}")
    print("="*60)
    
    return train_df, valid_df, test_df


def select_features(df, mode='all', custom_features=None, label_columns=None):
    """
    Select features based on mode.
    
    Args:
        df: DataFrame
        mode: 'all', 'segmentation', 'resnet', or 'custom'
        custom_features: List of feature names (if mode='custom')
        label_columns: List of label column names to exclude
    
    Returns:
        List of feature column names
    """
    all_cols = df.columns
    
    # Columns to always exclude
    exclude = label_columns
    
    if mode == 'all':
        # All features except metadata and labels
        features = [c for c in all_cols if c not in exclude]
    
    elif mode == 'segmentation':
        # Only segmentation features (exclude resnet_* and hist_*)
        features = [c for c in all_cols 
                   if c not in exclude 
                   and not c.startswith('resnet_')
                   and not c.startswith('hist_')]
    
    elif mode == 'resnet':
        # Only ResNet features
        features = [c for c in all_cols if c.startswith('resnet_')]
    
    elif mode == 'custom':
        if custom_features is None:
            raise ValueError("custom_features must be provided when mode='custom'")
        features = custom_features
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Verify all features exist
    missing = [f for f in features if f not in all_cols]
    if missing:
        print(f"Warning: {len(missing)} features not found: {missing[:5]}...")
        features = [f for f in features if f in all_cols]
    
    return features


def prepare_data(train_df, valid_df, test_df, config):
    """
    Prepare features and labels for training.
    
    Args:
        train_df, valid_df, test_df: DataFrames
        config: Config object
    
    Returns:
        Dictionary with X, y for train/valid/test
    """
    print("\n" + "="*60)
    print("PREPARING DATA")
    print("="*60)
    
    # Select features
    feature_cols = select_features(
        train_df, 
        config.FEATURE_MODE, 
        config.CUSTOM_FEATURES,
        config.LABEL_COLUMNS + config.METADATA_COLUMS
    )
    print(f"\nFeature mode: {config.FEATURE_MODE}")
    print(f"Number of features: {len(feature_cols)}")
    if len(feature_cols) <= 20:
        print(f"Features: {feature_cols}")
    else:
        print(f"First 10 features: {feature_cols[:10]}")
        print(f"Last 10 features: {feature_cols[-10:]}")
    
    # Extract features
    X_train = train_df[feature_cols]
    X_valid = valid_df[feature_cols]
    X_test = test_df[feature_cols]
    
    # Extract labels (convert one-hot to class indices)
    y_train = train_df[config.LABEL_COLUMNS].to_numpy().argmax(axis=1)
    y_valid = valid_df[config.LABEL_COLUMNS].to_numpy().argmax(axis=1)
    y_test = test_df[config.LABEL_COLUMNS].to_numpy().argmax(axis=1)
    
    print(f"\nClass distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls} ({config.LABEL_COLUMNS[cls]}): {count} samples ({count/len(y_train)*100:.1f}%)")
    
    # Handle missing values
    n_missing_train = np.isnan(X_train).sum()
    n_missing_valid = np.isnan(X_valid).sum()
    n_missing_test = np.isnan(X_test).sum()
    
    print(f"\nMissing values - Train: {n_missing_train}, Valid: {n_missing_valid}, Test: {n_missing_test}")
        
    return {
        'X_train': X_train,
        'X_valid': X_valid,
        'X_test': X_test,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test,
        'feature_names': feature_cols,
        'n_classes': len(config.LABEL_COLUMNS)
    }


def apply_pca(data, config):
    """
    Apply PCA to reduce dimensionality.
    
    Args:
        data: Dictionary from prepare_data
        config: Config object
    
    Returns:
        Updated data dictionary with transformed features
    """
    if not config.USE_PCA:
        print("\nSkipping PCA (USE_PCA=False)")
        return data
    
    print("\n" + "="*60)
    print("APPLYING PCA")
    print("="*60)
    
    # Determine n_components
    if config.PCA_VARIANCE_THRESHOLD is not None:
        n_components = config.PCA_VARIANCE_THRESHOLD
    else:
        n_components = min(config.PCA_N_COMPONENTS, data['X_train'].shape[1])
    
    print(f"PCA components: {n_components}")
    
    # Calculate PCA for the purpose of plotting
    pca = PCA(n_components=None, random_state=config.RANDOM_STATE)
    X_train_pca = pca.fit_transform(data['X_train'])
    X_valid_pca = pca.transform(data['X_valid'])
    X_test_pca = pca.transform(data['X_test'])
    plot_pca_variance(
        pca,
        config.PCA_VARIANCE_THRESHOLD,
        config.OUTPUT_DIR / "pca_variance.png"
    )
    
    # Fit PCA on training data and transfrom valid and test data
    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    X_train_pca = pca.fit_transform(data['X_train'])
    X_valid_pca = pca.transform(data['X_valid'])
    X_test_pca = pca.transform(data['X_test'])

    # Report variance explained
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"Variance explained: {var_explained:.4f}")
    print(f"Number of components: {pca.n_components_}")
    print(f"Original features: {data['X_train'].shape[1]}")
    print(f"Reduced features: {X_train_pca.shape[1]}")
    
    # Update data dictionary
    data['X_train'] = X_train_pca
    data['X_valid'] = X_valid_pca
    data['X_test'] = X_test_pca
    data['pca'] = pca
    data['pca_variance_explained'] = var_explained
    
    return data


def standardize_features(data, scaler):
    X_train_scaled = scaler.fit_transform(data['X_train'])
    X_valid_scaled = scaler.transform(data['X_valid'])
    X_test_scaled = scaler.transform(data['X_test'])
    
    print(f"Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
    
    data['X_train'] = X_train_scaled
    data['X_valid'] = X_valid_scaled
    data['X_test'] = X_test_scaled
    data['scaler'] = scaler
    
    return data


# ============================================================================
# MODEL 1: DECISION TREE
# ============================================================================

def train_decision_tree(data, config):
    """
    Train a decision tree classifier.
    
    Args:
        data: Dictionary with features and labels
        config: Config object
    
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("TRAINING DECISION TREE")
    print("="*60)
    
    model = DecisionTreeClassifier(
        max_depth=config.DT_MAX_DEPTH,
        min_samples_split=config.DT_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.DT_MIN_SAMPLES_LEAF,
        random_state=config.RANDOM_STATE
    )
    
    model.fit(data['X_train'], data['y_train'])
    
    # Evaluate
    y_pred_train = model.predict(data['X_train'])
    y_pred_valid = model.predict(data['X_valid'])
    
    train_acc = accuracy_score(data['y_train'], y_pred_train)
    valid_acc = accuracy_score(data['y_valid'], y_pred_valid)
    
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Balanced train accuracy: {balanced_accuracy_score(data['y_train'], y_pred_train)}")
    print(f"Valid accuracy: {valid_acc:.4f}")
    print(f"Balanced valid accuracy: {balanced_accuracy_score(data['y_valid'], y_pred_valid)}")
    
    return model


# ============================================================================
# MODEL 2: LOGISTIC REGRESSION
# ============================================================================

def train_logistic_regression(data, config):
    """
    Train multinomial logistic regression.
    
    Args:
        data: Dictionary with features and labels
        config: Config object
    
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)
    
    model = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        max_iter=config.LR_MAX_ITER,
        C=config.LR_C,
        random_state=config.RANDOM_STATE
    )
    
    model.fit(data['X_train'], data['y_train'])
    
    # Evaluate
    y_pred_train = model.predict(data['X_train'])
    y_pred_valid = model.predict(data['X_valid'])
    
    train_acc = accuracy_score(data['y_train'], y_pred_train)
    valid_acc = accuracy_score(data['y_valid'], y_pred_valid)
    
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Balanced train accuracy: {balanced_accuracy_score(data['y_train'], y_pred_train)}")
    print(f"Valid accuracy: {valid_acc:.4f}")
    print(f"Balanced valid accuracy: {balanced_accuracy_score(data['y_valid'], y_pred_valid)}")
    
    return model


# ============================================================================
# MODEL 3: NEURAL NETWORK (PyTorch)
# ============================================================================

class FeedForwardNN(nn.Module):
    """Simple feed-forward neural network."""
    
    def __init__(self, input_size, hidden_sizes, n_classes, dropout=0.3):
        super(FeedForwardNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, n_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_neural_network(data, config):
    """
    Train a feed-forward neural network using PyTorch.
    
    Args:
        data: Dictionary with features and labels
        config: Config object
    
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("TRAINING NEURAL NETWORK")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.LongTensor(data['y_train'])
    X_valid = torch.FloatTensor(data['X_valid'])
    y_valid = torch.LongTensor(data['y_valid'])
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.NN_BATCH_SIZE, 
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config.NN_BATCH_SIZE, 
        shuffle=False
    )
    
    # Create model
    input_size = data['X_train'].shape[1]
    model = FeedForwardNN(
        input_size=input_size,
        hidden_sizes=config.NN_HIDDEN_SIZES,
        n_classes=data['n_classes'],
        dropout=config.NN_DROPOUT
    ).to(config.DEVICE)
    
    print(f"\nModel architecture:")
    print(f"  Input: {input_size}")
    for i, size in enumerate(config.NN_HIDDEN_SIZES):
        print(f"  Hidden {i+1}: {size}")
    print(f"  Output: {data['n_classes']}")
    
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.NN_LEARNING_RATE, momentum=0.9, nesterov=True, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
    factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel')
    
    # Training loop
    best_model_state = model.state_dict().copy()  
    best_valid_bal_acc = 0
    patience_counter = 0
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    train_bal_accs = []
    valid_bal_accs = []
    
    for epoch in range(config.NN_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []
        
        for _, data in enumerate(train_loader):
            batch_X, batch_y = data
            batch_X = batch_X.to(config.DEVICE)
            batch_y = batch_y.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_bal_acc = balanced_accuracy_score(train_labels, train_preds)
        
        # Validation
        model.eval()
        valid_loss = 0
        valid_correct = 0
        valid_total = 0
        valid_preds  = []
        valid_labels  = []
        
        with torch.no_grad():
            for batch_X, batch_y in valid_loader:
                batch_X = batch_X.to(config.DEVICE)
                batch_y = batch_y.to(config.DEVICE)
                
                outputs = model(batch_X)
                
                valid_loss += loss_fn(outputs, batch_y).item()

                _, predicted = outputs.max(1)
                valid_total += batch_y.size(0)
                valid_correct += predicted.eq(batch_y).sum().item()

                valid_preds.extend(predicted.cpu().numpy())
                valid_labels.extend(batch_y.cpu().numpy())
        
        valid_loss /= len(valid_loader)
        valid_acc = valid_correct / valid_total
        valid_bal_acc = balanced_accuracy_score(valid_labels, valid_preds)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        train_bal_accs.append(train_bal_acc)
        valid_bal_accs.append(valid_bal_acc)

        scheduler.step(valid_bal_acc)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{config.NN_EPOCHS} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Balanced Acc: {train_bal_acc:.4f} - "
                f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, Valid Balanced Acc: {valid_bal_acc:.4f} - "
                f"Learning rate: {scheduler.get_last_lr()}"
            )

        
        # Early stopping
        if valid_bal_acc > best_valid_bal_acc:
            best_valid_bal_acc = valid_bal_acc
            best_valid_acc = valid_acc
            best_train_bal_acc = train_bal_acc
            best_train_acc = train_acc
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= config.NN_EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"\nBest Validation Results:")
    print(f"  Accuracy: {best_valid_acc:.4f}")
    print(f"  Balanced Accuracy: {best_valid_bal_acc:.4f}")
    print(f"  Train Accuracy: {best_train_acc:.4f}")
    print(f"  Train Balanced Accuracy: {best_train_bal_acc:.4f}")

    
    # Store training history
    model.train_history = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_accs': train_accs,
        'valid_accs': valid_accs,
        'train_bal_accs': train_bal_accs,
        'valid_bal_accs': valid_bal_accs
    }
    
    return model


def evaluate_model(model, data, config, model_name, split='test', is_pytorch=False):

    # Get data for the specified split
    X_data = data[f'X_{split}']
    y_true = data[f'y_{split}']
    
    # Get predictions
    if is_pytorch:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_data).to(config.DEVICE)
            outputs = model(X_tensor)
            _, y_pred = outputs.max(1)
            y_pred = y_pred.cpu().numpy()
            y_proba = torch.softmax(outputs, dim=1).cpu().numpy()
    else:
        y_pred = model.predict(X_data)
        y_proba = model.predict_proba(X_data)
    
    print(f"Model: {model_name}, split: {split}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred)}")
    print(classification_report(
        y_true, 
        y_pred, 
        target_names=config.LABEL_COLUMNS,
        zero_division=0
    ))
    
    cm = confusion_matrix(y_true, y_pred, normalize='all')
    
    results = {
        'model_name': model_name,
        'split': split,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'y_true': y_true
    }
    
    return results


def plot_confusion_matrices(results_list, config, save_path=None):
    n_models = len(results_list)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, results in enumerate(results_list):
        cm = results['confusion_matrix']
        
        sns.heatmap(
            cm, 
            annot=True, 
            cmap='Blues',
            xticklabels=config.LABEL_COLUMNS,
            yticklabels=config.LABEL_COLUMNS,
            ax=axes[idx]
        )
        axes[idx].set_title(f"{results['model_name']}")
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to: {save_path}")


def plot_nn_training_history(model, save_path=None):
    """
    Plot neural network training history including balanced accuracy.
    
    Args:
        model: Trained PyTorch model with train_history attribute
        save_path: Path to save figure
    """
    history = model.train_history
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    
    # Plot 1: Loss
    ax1.plot(history['train_losses'], label='Train', linewidth=2)
    ax1.plot(history['valid_losses'], label='Valid', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2.plot(history['train_accs'], label='Train', linewidth=2)
    ax2.plot(history['valid_accs'], label='Valid', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Balanced Accuracy (NEW)
    ax3.plot(history['train_bal_accs'], label='Train', linewidth=2, color='#2ca02c')
    ax3.plot(history['valid_bal_accs'], label='Valid', linewidth=2, color='#ff7f0e')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Balanced Accuracy')
    ax3.set_title('Training and Validation Balanced Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to: {save_path}")
    


def plot_pca_variance(pca, target_variance=0.95, save_path=None):
    """
    Plot PCA explained variance ratio with cumulative variance.
    
    Args:
        pca: Fitted PCA object
        target_variance: Target cumulative variance (e.g., 0.9 for 90%)
        save_path: Path to save figure
    """
    if pca is None:
        print("PCA not applied, skipping variance plot")
        return
    
    print("\n" + "="*60)
    print("PCA VARIANCE ANALYSIS")
    print("="*60)
    
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    # Find number of components for target variance
    n_components_target = np.argmax(cumulative_var >= target_variance) + 1
    
    print(f"Total components: {len(explained_var)}")
    print(f"Target variance: {target_variance:.1%}")
    print(f"Components needed: {n_components_target}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Individual explained variance
    n_show = min(50, len(explained_var))  # Show first 50 components
    ax1.bar(range(n_show), explained_var[:n_show], alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title(f'Individual Explained Variance (First {n_show} Components)')
    ax1.grid(True, alpha=0.3, axis='y')
        
    # Plot 2: Cumulative explained variance
    ax2.plot(range(len(cumulative_var)), cumulative_var, 'b-', linewidth=2)
    ax2.axhline(y=target_variance, color='r', linewidth=2, 
                label=f'{target_variance} explained variance')
    
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance by PCA Components')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA variance plot saved to: {save_path}")


def balance_classes(X_train, y_train, strategy='undersample', random_state=42):
    """
    Balance classes by under-sampling or over-sampling.
    
    Args:
        X_train: Training features
        y_train: Training labels
        strategy: 'undersample' or 'oversample'
        random_state: Random seed
    
    Returns:
        Balanced X_train, y_train
    """
    print("\n" + "="*60)
    print("BALANCING CLASSES")
    print("="*60)
    
    unique, counts = np.unique(y_train, return_counts=True)
    
    print(f"\nOriginal class distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    if strategy == 'undersample':
        # Undersample: take minimum class size for all classes
        min_samples = counts.min()
        print(f"\nUndersampling to {min_samples} samples per class")
        
        indices = []
        np.random.seed(random_state)
        
        for cls in unique:
            cls_indices = np.where(y_train == cls)[0]
            selected = np.random.choice(cls_indices, min_samples, replace=False)
            indices.extend(selected)
        
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        X_balanced = X_train[indices]
        y_balanced = y_train[indices]
    
    elif strategy == 'oversample':
        # Oversample: replicate minority classes to match majority
        max_samples = counts.max()
        print(f"\nOversampling to {max_samples} samples per class")
        
        X_balanced_list = []
        y_balanced_list = []
        np.random.seed(random_state)
        
        for cls in unique:
            cls_indices = np.where(y_train == cls)[0]
            cls_X = X_train[cls_indices]
            cls_y = y_train[cls_indices]
            
            if len(cls_indices) < max_samples:
                # Need to oversample
                n_repeat = max_samples - len(cls_indices)
                extra_indices = np.random.choice(len(cls_indices), n_repeat, replace=True)
                cls_X = np.vstack([cls_X, cls_X[extra_indices]])
                cls_y = np.concatenate([cls_y, cls_y[extra_indices]])
            
            X_balanced_list.append(cls_X)
            y_balanced_list.append(cls_y)
        
        X_balanced = np.vstack(X_balanced_list)
        y_balanced = np.concatenate(y_balanced_list)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y_balanced))
        X_balanced = X_balanced[shuffle_idx]
        y_balanced = y_balanced[shuffle_idx]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'undersample' or 'oversample'")
    
    # Report new distribution
    unique_new, counts_new = np.unique(y_balanced, return_counts=True)
    print(f"\nBalanced class distribution:")
    for cls, count in zip(unique_new, counts_new):
        print(f"  Class {cls}: {count} samples ({count/len(y_balanced)*100:.1f}%)")
    
    print(f"\nOriginal size: {len(y_train)} samples")
    print(f"Balanced size: {len(y_balanced)} samples")
    
    return X_balanced, y_balanced


def main():    
    config = Config()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.RANDOM_STATE)
    
    # Data preparation
    train_df, valid_df, test_df = load_data(config)
    
    data = prepare_data(train_df, valid_df, test_df, config)
    
    data = standardize_features(data, StandardScaler())
    
    data = apply_pca(data, config)
    
    if config.BALANCE_CLASSES:
        data['X_train'], data['y_train'] = balance_classes(
            data['X_train'],
            data['y_train'],
            strategy=config.BALANCE_STRATEGY,
            random_state=config.RANDOM_STATE
        )
    
    # Training
    nn_model = train_neural_network(data, config)
    dt_model = train_decision_tree(data, config)
    lr_model = train_logistic_regression(data, config)
    plot_nn_training_history(
        nn_model,
        save_path=config.OUTPUT_DIR / "nn_training_history.png"
    )

    
    # Evaluation on VALIDATION set
    dt_valid = evaluate_model(dt_model, data, config, "Decision Tree", split='valid', is_pytorch=False)
    lr_valid = evaluate_model(lr_model, data, config, "Logistic Regression", split='valid', is_pytorch=False)
    nn_valid = evaluate_model(nn_model, data, config, "Neural Network", split='valid', is_pytorch=True)
    
    valid_results = [dt_valid, lr_valid, nn_valid]
    plot_confusion_matrices(
        valid_results, 
        config, 
        save_path=config.OUTPUT_DIR / "valid_confusion_matrices.png"
    )
    
    # Evaluation on TEST set
    dt_test = evaluate_model(dt_model, data, config, "Decision Tree", split='test', is_pytorch=False)
    lr_test = evaluate_model(lr_model, data, config, "Logistic Regression", split='test', is_pytorch=False)
    nn_test = evaluate_model(nn_model, data, config, "Neural Network", split='test', is_pytorch=True)
    
    test_results = [dt_test, lr_test, nn_test]    
        
    plot_confusion_matrices(
        test_results, 
        config, 
        save_path=config.OUTPUT_DIR / "test_confusion_matrices.png"
    )

    # After training models and before evaluation:

    if config.USE_PCA and 'pca' in data:
        analyze_feature_importance(
            dt_model=dt_model,
            lr_model=lr_model,
            pca=data['pca'],
            original_feature_names=data['feature_names'],
            output_dir=config.OUTPUT_DIR / "feature_importance",
            top_n=15
        )


if __name__ == "__main__":
    main()