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
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    Prepare the dataset for modeling:
    - Drop unnecessary columns (user_id)
    - Encode categorical variables
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
        encoders: Dictionary of label encoders for categorical variables
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
    
    # Identify categorical and numerical columns
    categorical_cols = ['gender', 'country', 'subscription_type', 'device_type']
    numerical_cols = ['age', 'listening_time', 'songs_played_per_day', 
                     'skip_rate', 'ads_listened_per_week', 'offline_listening']
    
    print(f"\nCategorical features: {categorical_cols}")
    print(f"Numerical features: {numerical_cols}")
    
    # Encode categorical variables
    encoders = {}
    X_encoded = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        encoders[col] = le
        print(f"✓ Encoded '{col}': {list(le.classes_)}")
    
    # Get feature names after encoding
    feature_names = list(X_encoded.columns)
    
    # Split into train+val and test sets (80-20 split)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split train+val into train and validation sets
    # val_size / (1 - test_size) gives us the right proportion
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
    
    # Scale numerical features
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_val_scaled[numerical_cols] = scaler.transform(X_val[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
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

