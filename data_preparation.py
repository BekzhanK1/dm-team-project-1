"""
Common Data Preparation Module for Spotify Churn Dataset
=========================================================
This module provides preprocessed data ready for model building.
All team members should import from this file to ensure consistency.

Problem Statement:
- Type: Binary Classification
- Target Variable: is_churned (0 = Not Churned, 1 = Churned)
- Features: 11 independent variables (demographic, behavioral, subscription data)
- Instances: 8000 users
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_explore_data(filepath='datasets/spotify_churn_dataset.csv'):
    """
    Load the dataset and return basic information
    
    Returns:
        df: Raw dataframe
    """
    df = pd.read_csv(filepath)
    
    print("="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"Shape: {df.shape}")
    print(f"Number of instances: {df.shape[0]}")
    print(f"Number of features: {df.shape[1] - 1}")  # Excluding user_id
    print(f"\nTarget Variable: is_churned")
    print(f"Churn Distribution:\n{df['is_churned'].value_counts()}")
    print(f"Churn Rate: {df['is_churned'].mean():.2%}")
    
    print(f"\n{'='*60}")
    print("MISSING VALUES")
    print("="*60)
    print(df.isnull().sum())
    
    print(f"\n{'='*60}")
    print("DATA TYPES")
    print("="*60)
    print(df.dtypes)
    
    print(f"\n{'='*60}")
    print("BASIC STATISTICS")
    print("="*60)
    print(df.describe())
    
    return df


def visualize_data(df, save_plots=True):
    """
    Create visualizations for data exploration
    
    Args:
        df: DataFrame to visualize
        save_plots: Whether to save plots to files
    """
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Correlation Matrix (for numerical features only)
    plt.figure(figsize=(12, 10))
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1)
    plt.title('Correlation Matrix - Numerical Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if save_plots:
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Churn Rate by Categorical Features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    categorical_features = ['gender', 'country', 'subscription_type', 'device_type']
    
    for idx, feature in enumerate(categorical_features):
        ax = axes[idx // 3, idx % 3]
        churn_by_feature = df.groupby(feature)['is_churned'].mean().sort_values(ascending=False)
        churn_by_feature.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title(f'Churn Rate by {feature}', fontweight='bold')
        ax.set_ylabel('Churn Rate')
        ax.set_xlabel(feature)
        ax.tick_params(axis='x', rotation=45)
    
    # Remove empty subplots
    for idx in range(len(categorical_features), 6):
        fig.delaxes(axes[idx // 3, idx % 3])
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('churn_by_categories.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Distribution of Numerical Features
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    numerical_features = ['age', 'listening_time', 'songs_played_per_day', 
                         'skip_rate', 'ads_listened_per_week', 'offline_listening']
    
    for idx, feature in enumerate(numerical_features):
        ax = axes[idx // 3, idx % 3]
        df[df['is_churned']==0][feature].hist(ax=ax, bins=30, alpha=0.5, 
                                               label='Not Churned', color='green')
        df[df['is_churned']==1][feature].hist(ax=ax, bins=30, alpha=0.5, 
                                               label='Churned', color='red')
        ax.set_title(f'Distribution of {feature}', fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.legend()
    
    # Remove empty subplots
    for idx in range(len(numerical_features), 9):
        fig.delaxes(axes[idx // 3, idx % 3])
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


def prepare_data(filepath='datasets/spotify_churn_dataset.csv', test_size=0.2, val_size=0.125, random_state=42):
    """
    Prepare the dataset for modeling using mixed encoding strategies:
    - Drop unnecessary columns (user_id)
    - Apply different encoders for different categorical features:
      * One-hot encoding: gender, device_type (low cardinality, no ordinality)
      * Ordinal encoding: subscription_type (natural ordering)
      * Target encoding: country (medium cardinality, captures churn patterns)
    - Split into train, validation, and test sets
    - Scale numerical features
    
    Args:
        filepath: Path to the CSV file
        test_size: Proportion of test set (default 0.2 = 20%)
        val_size: Proportion of validation set from training data (default 0.125 = 10% of total)
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_val, X_test: Feature matrices (scaled)
        y_train, y_val, y_test: Target vectors
        feature_names: List of feature names after encoding
        encoders: Dictionary containing fitted encoders
        scaler: Fitted StandardScaler object
    """
    # Load data
    df = pd.read_csv(filepath)
    
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    # Drop user_id (not a feature)
    df = df.drop('user_id', axis=1)
    print(f"✓ Dropped 'user_id' column")
    
    # Separate features and target
    X = df.drop('is_churned', axis=1)
    y = df['is_churned']
    
    # Define encoding strategies for different categorical features
    onehot_features = ['gender', 'device_type']  # Low cardinality, no ordinality
    ordinal_features = ['subscription_type']      # Natural ordering
    target_features = ['country']                 # Medium cardinality, captures churn patterns
    numerical_cols = ['age', 'listening_time', 'songs_played_per_day', 
                     'skip_rate', 'ads_listened_per_week', 'offline_listening']
    
    print(f"\nMixed Encoding Strategy:")
    print(f"  One-hot encoding: {onehot_features}")
    print(f"  Ordinal encoding: {ordinal_features}")
    print(f"  Target encoding: {target_features}")
    print(f"  Numerical features: {numerical_cols}")
    
    # Split data first to prevent target encoding leakage
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split train+val into train and validation sets
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    print(f"\n{'='*60}")
    print("DATASET SPLIT")
    print("="*60)
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    print(f"Total: {len(df)} samples")
    
    print(f"\nChurn distribution:")
    print(f"  Train: {y_train.mean():.2%}")
    print(f"  Validation: {y_val.mean():.2%}")
    print(f"  Test: {y_test.mean():.2%}")
    
    # Initialize encoders
    encoders = {}
    
    # 1. One-hot encoding for gender and device_type
    onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)  # drop='first' to avoid multicollinearity
    X_train_onehot = onehot_encoder.fit_transform(X_train[onehot_features])
    X_val_onehot = onehot_encoder.transform(X_val[onehot_features])
    X_test_onehot = onehot_encoder.transform(X_test[onehot_features])
    encoders['onehot'] = onehot_encoder
    
    onehot_feature_names = onehot_encoder.get_feature_names_out(onehot_features)
    print(f"\n✓ One-hot encoded features:")
    for i, feature in enumerate(onehot_features):
        unique_values = X_train[feature].unique()
        print(f"  {feature}: {list(unique_values)}")
    
    # 2. Ordinal encoding for subscription_type
    ordinal_encoder = OrdinalEncoder(categories=[['Free', 'Student', 'Family', 'Premium']])
    X_train_ordinal = ordinal_encoder.fit_transform(X_train[ordinal_features])
    X_val_ordinal = ordinal_encoder.transform(X_val[ordinal_features])
    X_test_ordinal = ordinal_encoder.transform(X_test[ordinal_features])
    encoders['ordinal'] = ordinal_encoder
    
    print(f"\n✓ Ordinal encoded features:")
    print(f"  subscription_type: Free=0, Student=1, Family=2, Premium=3")
    
    # 3. Target encoding for country (fit only on training data)
    target_encoder = TargetEncoder(random_state=random_state)
    X_train_target = target_encoder.fit_transform(X_train[target_features], y_train)
    X_val_target = target_encoder.transform(X_val[target_features])
    X_test_target = target_encoder.transform(X_test[target_features])
    encoders['target'] = target_encoder
    
    print(f"\n✓ Target encoded features:")
    print(f"  country: Mean churn rate per country")
    
    # Combine all encoded features
    X_train_encoded = np.column_stack([
        X_train[numerical_cols].values,
        X_train_target,
        X_train_ordinal,
        X_train_onehot
    ])
    
    X_val_encoded = np.column_stack([
        X_val[numerical_cols].values,
        X_val_target,
        X_val_ordinal,
        X_val_onehot
    ])
    
    X_test_encoded = np.column_stack([
        X_test[numerical_cols].values,
        X_test_target,
        X_test_ordinal,
        X_test_onehot
    ])
    
    # Create feature names
    feature_names = numerical_cols + target_features + ordinal_features + list(onehot_feature_names)
    
    print(f"\n✓ Total features after mixed encoding: {len(feature_names)}")
    print(f"  Features: {feature_names}")
    
    # Scale numerical features
    scaler = StandardScaler()
    
    # Get indices of numerical features in the combined array
    numerical_indices = list(range(len(numerical_cols)))
    
    # Fit scaler on training data only
    X_train_scaled = X_train_encoded.copy()
    X_val_scaled = X_val_encoded.copy()
    X_test_scaled = X_test_encoded.copy()
    
    X_train_scaled[:, numerical_indices] = scaler.fit_transform(X_train_encoded[:, numerical_indices])
    X_val_scaled[:, numerical_indices] = scaler.transform(X_val_encoded[:, numerical_indices])
    X_test_scaled[:, numerical_indices] = scaler.transform(X_test_encoded[:, numerical_indices])
    
    print(f"\n✓ Scaled numerical features using StandardScaler")
    
    print(f"\n{'='*60}")
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Feature count: {len(feature_names)}")
    print(f"Features: {feature_names}")
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test, 
            feature_names, encoders, scaler)


def get_prepared_data():
    """
    Convenience function to get all prepared data in one call.
    Use this in your model files!
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names, encoders, scaler
    """
    return prepare_data()


if __name__ == "__main__":
    # When run directly, perform exploration and preparation
    print("Loading and exploring data...")
    df = load_and_explore_data()
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    visualize_data(df, save_plots=True)
    
    print("\nPreparing data for modeling...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, encoders, scaler = prepare_data()
    
    print("\n" + "="*60)
    print("READY FOR MODELING!")
    print("="*60)
    print("\nImport in your model files using:")
    print("  from data_preparation import get_prepared_data")
    print("  X_train, X_val, X_test, y_train, y_val, y_test, *_ = get_prepared_data()")

