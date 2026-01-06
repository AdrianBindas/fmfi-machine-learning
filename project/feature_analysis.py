import numpy as np
import pandas as pd


def get_decision_tree_importance(model, feature_names):
    importance = model.feature_importances_
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return df


def get_logistic_regression_importance(model, feature_names, method='mean_abs'):
    coef = model.coef_
    
    if method == 'mean_abs':
        # Mean absolute value across all classes
        importance = np.abs(coef).mean(axis=0)
    elif method == 'max_abs':
        # Maximum absolute value across all classes
        importance = np.abs(coef).max(axis=0)
    elif method == 'l2_norm':
        # L2 norm across all classes
        importance = np.sqrt((coef ** 2).sum(axis=0))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return df


def map_pca_to_original_features(pca, pca_importance, original_feature_names):
    """
    Map PCA component importance back to original features.
    
    This identifies which original features contribute most to the important
    PCA components.
    
    Args:
        pca: Fitted PCA object
        pca_importance: DataFrame with PCA component importance
        original_feature_names: List of original feature names
        top_n: Number of top original features to return
    
    Returns:
        DataFrame with original feature importance
    """
    # Get PCA components (shape: n_components x n_original_features)
    components = pca.components_
    
    # Initialize importance accumulator for original features
    original_importance = np.zeros(len(original_feature_names))
    
    # For each PCA component, distribute its importance to original features
    # based on the component loadings (weights)
    for idx, row in pca_importance.iterrows():
        pca_idx = int(row['feature'].split('_')[1])  # Extract component index
        pca_imp = row['importance']
        
        # Get loadings for this component
        loadings = np.abs(components[pca_idx])
        
        # Distribute importance proportionally to loading magnitude
        original_importance += pca_imp * loadings
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature': original_feature_names,
        'importance': original_importance
    }).sort_values('importance', ascending=False)
    
    # Normalize to sum to 1 for easier interpretation
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['importance_percent'] = df['importance_normalized'] * 100
    
    return df


def analyze_feature_importance(dt_model, lr_model, pca, original_feature_names, 
                               output_dir, top_n=5):
        
    # Get PCA component names
    n_components = pca.n_components_
    pca_feature_names = [f'PC_{i}' for i in range(n_components)]
    
    # Extract importance from Decision Tree
    print("\n1. Extracting Decision Tree feature importance...")
    dt_importance = get_decision_tree_importance(dt_model, pca_feature_names)
    print(f"   Top 5 PCA components:")
    for idx, row in dt_importance.head(5).iterrows():
        print(f"     {row['feature']}: {row['importance']:.6f}")
    
    # Extract importance from Logistic Regression
    print("\n2. Extracting Logistic Regression feature importance...")
    lr_importance = get_logistic_regression_importance(lr_model, pca_feature_names)
    print(f"   Top 5 PCA components:")
    for idx, row in lr_importance.head(5).iterrows():
        print(f"     {row['feature']}: {row['importance']:.6f}")
        

    dt_original_importance = map_pca_to_original_features(
        pca, dt_importance[['feature', 'importance']],
        original_feature_names
    )

    lr_original_importance = map_pca_to_original_features(
        pca, lr_importance[['feature', 'importance']],
        original_feature_names
    )
    
    print(f"\n   TOP {top_n} ORIGINAL FEATURES:")
    print(f"   {'Rank':<6} {'Feature':<40} {'Importance %':<15}")
    print("   " + "-"*65)
    for rank, (idx, row) in enumerate(dt_original_importance.head(top_n).iterrows(), 1):
        print(f"   {rank:<6} {row['feature']:<40} {row['importance_percent']:<15.2f}%")
    
    for rank, (_, row) in enumerate(dt_original_importance.iterrows(), 1):
        if not str(row['feature']).startswith('resnet'):
            print(f"   {rank:<6} {row['feature']:<40} {row['importance_percent']:<15.2f}%")

    print(f"\n   TOP {top_n} ORIGINAL FEATURES:")
    print(f"   {'Rank':<6} {'Feature':<40} {'Importance %':<15}")
    print("   " + "-"*65)
    for rank, (idx, row) in enumerate(lr_original_importance.head(top_n).iterrows(), 1):
        print(f"   {rank:<6} {row['feature']:<40} {row['importance_percent']:<15.2f}%")
    
    for rank, (_, row) in enumerate(lr_original_importance.iterrows(), 1):
        if not str(row['feature']).startswith('resnet'):
            print(f"   {rank:<6} {row['feature']:<40} {row['importance_percent']:<15.2f}%")
