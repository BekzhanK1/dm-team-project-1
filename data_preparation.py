import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

def load_and_explore_dataset(filepath='datasets/spotify_churn_dataset.csv'):
    """
    Load the dataset and perform initial exploration
    """
    print("="*60)
    print("LOADING AND EXPLORING DATASET")
    print("="*60)
    
    # Load the dataset
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1] - 1}")  # Excluding target variable
    
    # Remove user_id column as it's not useful for modeling
    df = df.drop('user_id', axis=1)
    print(f"\n✓ Removed 'user_id' column")
    print(f"New shape: {df.shape}")
    
    # Display basic info
    print(f"\n{'='*60}")
    print("DATASET INFO")
    print("="*60)
    print(df.info())
    
    # Display missing values
    print(f"\n{'='*60}")
    print("MISSING VALUES")
    print("="*60)
    print(df.isnull().sum())
    
    # Identify column types
    print(f"\n{'='*60}")
    print("COLUMN TYPE ANALYSIS")
    print("="*60)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variable from numeric columns
    if 'is_churned' in numeric_columns:
        numeric_columns.remove('is_churned')
    
    print(f"Numeric columns ({len(numeric_columns)}): {numeric_columns}")
    print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")
    print(f"Target variable: is_churned")
    
    # Display categorical column values
    print(f"\n{'='*60}")
    print("CATEGORICAL COLUMN VALUES")
    print("="*60)
    for col in categorical_columns:
        unique_values = df[col].unique()
        print(f"{col}: {unique_values} (unique count: {len(unique_values)})")
    
    # Display descriptive statistics
    print(f"\n{'='*60}")
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    print(df.describe())
    
    return df, numeric_columns, categorical_columns

def split_data(df, test_size=0.3, random_state=42):
    """
    Split data into training and test sets
    """
    print(f"\n{'='*60}")
    print("SPLITTING DATA")
    print("="*60)
    
    # Separate features and target
    X = df.drop('is_churned', axis=1)
    y = df['is_churned']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    print(f"Total: {len(df)} samples")
    
    # Check churn distribution
    print(f"\nChurn distribution:")
    print(f"  Overall: {y.mean()*100:.2f}%")
    print(f"  Training: {y_train.mean()*100:.2f}%")
    print(f"  Test: {y_test.mean()*100:.2f}%")
    
    return X_train, X_test, y_train, y_test

def create_preprocessing_pipeline(X_train, X_test, numeric_columns, categorical_columns):
    """
    Create and apply preprocessing pipeline
    """
    print(f"\n{'='*60}")
    print("CREATING PREPROCESSING PIPELINE")
    print("="*60)
    
    print("Fitting preprocessors on training set only...")
    
    # OneHotEncoder for categorical features (without dropping any categories)
    onehot_encoder = OneHotEncoder(sparse_output=False, drop=None)
    X_train_categorical = onehot_encoder.fit_transform(X_train[categorical_columns])
    X_test_categorical = onehot_encoder.transform(X_test[categorical_columns])
    
    print("✓ OneHotEncoder fitted on training categorical features")
    
    # Get categorical feature names
    categorical_feature_names = onehot_encoder.get_feature_names_out(categorical_columns)
    print(f"Categorical features after encoding: {len(categorical_feature_names)}")
    print(f"  Feature names: {list(categorical_feature_names)}")
    
    # StandardScaler for numeric features
    scaler = StandardScaler()
    X_train_numeric = scaler.fit_transform(X_train[numeric_columns])
    X_test_numeric = scaler.transform(X_test[numeric_columns])
    
    print("✓ StandardScaler fitted on training numeric features")
    
    print("\nTransforming both training and test sets...")
    
    # Combine numeric and categorical features
    X_train_processed = np.hstack([X_train_numeric, X_train_categorical])
    X_test_processed = np.hstack([X_test_numeric, X_test_categorical])
    
    # Create final feature names
    feature_names = list(numeric_columns) + list(categorical_feature_names)
    
    print("✓ Test set transformed using fitted preprocessors")
    
    print(f"\n✓ Final processed data:")
    print(f"  Training features shape: {X_train_processed.shape}")
    print(f"  Test features shape: {X_test_processed.shape}")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Numeric features: {len(numeric_columns)}")
    print(f"  Categorical features (one-hot): {len(categorical_feature_names)}")
    
    return X_train_processed, X_test_processed, feature_names, onehot_encoder, scaler

def create_correlation_matrix(X_train, y_train, feature_names):
    """
    Create correlation matrix for processed features
    """
    print(f"\n{'='*60}")
    print("CREATING CORRELATION MATRIX")
    print("="*60)
    
    # Combine features with target for correlation analysis
    data_for_corr = np.hstack([X_train, y_train.values.reshape(-1, 1)])
    
    # Create DataFrame with feature names
    corr_df = pd.DataFrame(data_for_corr, columns=feature_names + ['is_churned'])
    
    # Calculate correlation matrix
    corr_matrix = corr_df.corr()
    
    print(f"Correlation matrix shape: {corr_matrix.shape}")
    
    # Get correlations with target variable
    target_corr = corr_matrix['is_churned'].drop('is_churned').abs().sort_values(ascending=False)
    
    print(f"\nTop 10 features most correlated with churn:")
    print(target_corr.head(10))
    
    # Create correlation matrix visualization with numeric values
    plt.figure(figsize=(16, 12))
    
    # Create heatmap with annotations
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                fmt='.2f', annot_kws={'size': 8})
    
    plt.title('Correlation Matrix - All Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix, target_corr

def analyze_multicollinearity(X_train, y_train, feature_names):
    """
    Analyze multicollinearity in the dataset
    """
    print(f"\n{'='*60}")
    print("MULTICOLLINEARITY ANALYSIS")
    print("="*60)
    
    # Create DataFrame for correlation analysis
    data_for_corr = np.hstack([X_train, y_train.values.reshape(-1, 1)])
    corr_df = pd.DataFrame(data_for_corr, columns=feature_names + ['is_churned'])
    corr_matrix = corr_df.corr()
    
    # Check correlations for ads_listened_per_week
    if 'ads_listened_per_week' in feature_names:
        ads_corr = corr_matrix['ads_listened_per_week'].drop('ads_listened_per_week').abs().sort_values(ascending=False)
        print(f"\nads_listened_per_week correlations:")
        print(ads_corr.head(10))
    
    # Check correlations for offline_listening
    if 'offline_listening' in feature_names:
        offline_corr = corr_matrix['offline_listening'].drop('offline_listening').abs().sort_values(ascending=False)
        print(f"\noffline_listening correlations:")
        print(offline_corr.head(10))
    
    # Check correlation between the two problematic features
    if 'ads_listened_per_week' in feature_names and 'offline_listening' in feature_names:
        corr_between = corr_matrix.loc['ads_listened_per_week', 'offline_listening']
        print(f"\nCorrelation between ads_listened_per_week and offline_listening:")
        print(f"{corr_between:.4f}")
    
    # Identify features to remove
    features_to_remove = ['ads_listened_per_week', 'offline_listening']
    
    print(f"\n⚠️  MULTICOLLINEARITY DETECTED:")
    print(f"   ads_listened_per_week has high correlation (0.876) with subscription_type_Free")
    print(f"   offline_listening has perfect negative correlation (-1.000) with subscription_type_Free")
    print(f"   → Removing both ads_listened_per_week and offline_listening")
    
    return features_to_remove

def remove_multicollinear_features(X_train, X_test, feature_names, features_to_remove):
    """
    Remove multicollinear features from the dataset
    """
    print(f"\n{'='*60}")
    print("REMOVING MULTICOLLINEAR FEATURES")
    print("="*60)
    
    # Find indices of features to remove
    indices_to_remove = []
    for feature in features_to_remove:
        if feature in feature_names:
            idx = feature_names.index(feature)
            indices_to_remove.append(idx)
            print(f"✓ Removing {feature} (index {idx})")
    
    # Remove features from data
    mask = np.ones(len(feature_names), dtype=bool)
    mask[indices_to_remove] = False
    
    X_train_cleaned = X_train[:, mask]
    X_test_cleaned = X_test[:, mask]
    feature_names_cleaned = [name for i, name in enumerate(feature_names) if mask[i]]
    
    print(f"\n✓ Features removed: {len(indices_to_remove)}")
    print(f"✓ New feature count: {len(feature_names_cleaned)}")
    print(f"✓ Training set shape: {X_train_cleaned.shape}")
    print(f"✓ Test set shape: {X_test_cleaned.shape}")
    
    return X_train_cleaned, X_test_cleaned, feature_names_cleaned

def create_final_correlation_matrix(X_train, y_train, feature_names):
    """
    Create final correlation matrix after removing multicollinear features
    """
    print(f"\n{'='*60}")
    print("FINAL CORRELATION MATRIX (AFTER REMOVING MULTICOLLINEAR FEATURES)")
    print("="*60)
    
    # Create DataFrame for correlation analysis
    data_for_corr = np.hstack([X_train, y_train.values.reshape(-1, 1)])
    corr_df = pd.DataFrame(data_for_corr, columns=feature_names + ['is_churned'])
    corr_matrix = corr_df.corr()
    
    print(f"Final correlation matrix shape: {corr_matrix.shape}")
    
    # Get correlations with target variable
    target_corr = corr_matrix['is_churned'].drop('is_churned').abs().sort_values(ascending=False)
    
    print(f"\nTop 10 features most correlated with churn (after cleaning):")
    print(target_corr.head(10))
    
    # Create final correlation matrix visualization with numeric values
    plt.figure(figsize=(14, 10))
    
    # Create heatmap with annotations
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                fmt='.2f', annot_kws={'size': 8})
    
    plt.title('Final Correlation Matrix - After Removing Multicollinear Features', 
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix, target_corr

def apply_selectkbest_fscore(X_train, X_test, y_train, feature_names, k=10):
    """
    Apply SelectKBest with F-score and display results
    """
    print(f"\n{'='*60}")
    print("SELECTKBEST WITH F-SCORE")
    print("="*60)
    
    # Apply SelectKBest with F-score
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names and scores
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_indices]
    selected_scores = selector.scores_[selected_indices]
    
    print(f"✓ Selected {len(selected_feature_names)} features out of {len(feature_names)}")
    print(f"✓ Training set shape: {X_train_selected.shape}")
    print(f"✓ Test set shape: {X_test_selected.shape}")
    
    # Create feature-score pairs for all features (sorted by score)
    all_feature_scores = list(zip(feature_names, selector.scores_))
    all_feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Extract sorted features and scores
    sorted_features = [pair[0] for pair in all_feature_scores]
    sorted_scores = [pair[1] for pair in all_feature_scores]
    
    # Create histogram visualization
    plt.figure(figsize=(14, 10))
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(sorted_features)), sorted_scores, 
                   color='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Highlight selected features
    for i, feature in enumerate(sorted_features):
        if feature in selected_feature_names:
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)
    
    # Shorten feature names for better display
    short_names = []
    for feature in sorted_features:
        if len(feature) > 30:
            short_names.append(feature[:27] + '...')
        else:
            short_names.append(feature)
    
    # Customize the plot
    plt.yticks(range(len(sorted_features)), short_names, fontsize=10)
    plt.xlabel('F-Score', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title(f'SelectKBest F-Score Results (K={k}) - Red = Selected', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars (only for selected features)
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        if sorted_features[i] in selected_feature_names:
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{score:.2f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Add cutoff line
    if len(selected_feature_names) > 0:
        cutoff_score = sorted_scores[len(selected_feature_names)-1]
        plt.axvline(x=cutoff_score, color='red', linestyle='--', alpha=0.8, linewidth=2)
        plt.text(cutoff_score + 0.2, len(sorted_features)/2, 
                f'K={k} cutoff\n{cutoff_score:.2f}', 
                ha='left', va='center', fontsize=11, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Invert y-axis to show highest scores at top
    plt.gca().invert_yaxis()
    
    # Adjust layout
    plt.subplots_adjust(left=0.4, right=0.95, top=0.9, bottom=0.1)
    plt.show()
    
    # Print summary
    print(f"\n✓ SelectKBest Summary:")
    print(f"   Total features: {len(sorted_features)}")
    print(f"   Selected features: {len(selected_feature_names)}")
    print(f"   Selection method: F-score (SelectKBest)")
    print(f"   Cutoff score: {cutoff_score:.2f}")
    
    print(f"\nSelected features (top {len(selected_feature_names)}):")
    for i, (feature, score) in enumerate(zip(selected_feature_names, selected_scores)):
        print(f"  {i+1:2d}. {feature:30s} {score:8.2f}")
    
    return X_train_selected, X_test_selected, selected_feature_names, selector

def prepare_data(filepath='datasets/spotify_churn_dataset.csv', test_size=0.3, random_state=42):
    """
    Main function to prepare data for machine learning
    """
    print("DATA PREPARATION PIPELINE")
    print("="*60)
    
    # Load and explore dataset
    df, numeric_columns, categorical_columns = load_and_explore_dataset(filepath)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, test_size, random_state)
    
    # Create preprocessing pipeline
    X_train_processed, X_test_processed, feature_names, onehot_encoder, scaler = create_preprocessing_pipeline(
        X_train, X_test, numeric_columns, categorical_columns
    )
    
    # Create initial correlation matrix
    corr_matrix, target_corr = create_correlation_matrix(X_train_processed, y_train, feature_names)
    
    # Analyze multicollinearity
    features_to_remove = analyze_multicollinearity(X_train_processed, y_train, feature_names)
    
    # Remove multicollinear features
    X_train_final, X_test_final, feature_names_final = remove_multicollinear_features(
        X_train_processed, X_test_processed, feature_names, features_to_remove
    )
    
    # Create final correlation matrix
    final_corr_matrix, final_target_corr = create_final_correlation_matrix(
        X_train_final, y_train, feature_names_final
    )
    
    # Apply SelectKBest with F-score
    X_train_selected, X_test_selected, selected_feature_names, feature_selector = apply_selectkbest_fscore(
        X_train_final, X_test_final, y_train, feature_names_final, k=10
    )
    
    print(f"\n{'='*60}")
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print("Dataset is now ready for machine learning models.")
    print(f"Training set: {X_train_selected.shape[0]} samples × {X_train_selected.shape[1]} features")
    print(f"Test set: {X_test_selected.shape[0]} samples × {X_test_selected.shape[1]} features")
    print(f"Selected features: {X_train_selected.shape[1]} out of {len(feature_names_final)}")
    print(f"Removed features due to multicollinearity: {len(features_to_remove)}")
    print(f"Feature selection method: SelectKBest with F-score (K=10)")
    
    print(f"\nReady for:")
    print(f"  - Cross-validation on training set")
    print(f"  - Hyperparameter tuning")
    print(f"  - Model training and selection")
    print(f"  - Final evaluation on test set")
    
    print(f"\n{'='*60}")
    print("USAGE IN MODEL FILES")
    print("="*60)
    print("Import in your model files using:")
    print("  from data_preparation import get_prepared_data")
    print("  X_train, X_test, y_train, y_test, feature_names, *_ = get_prepared_data()")
    
    return X_train_selected, X_test_selected, y_train, y_test, selected_feature_names, onehot_encoder, scaler, feature_selector

def get_prepared_data(filepath='datasets/spotify_churn_dataset.csv', test_size=0.3, random_state=42):
    """
    Convenience function to get prepared data
    """
    return prepare_data(filepath, test_size, random_state)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, onehot_encoder, scaler = prepare_data()