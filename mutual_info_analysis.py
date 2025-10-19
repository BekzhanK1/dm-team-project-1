import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load data and apply preprocessing"""
    print("Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv('datasets/spotify_churn_dataset.csv')
    df = df.drop('user_id', axis=1)  # Remove user_id
    
    # Split features and target
    X = df.drop('is_churned', axis=1)
    y = df['is_churned']
    
    # Identify column types
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric columns: {numeric_columns}")
    print(f"Categorical columns: {categorical_columns}")
    
    # Apply preprocessing
    # OneHotEncoder for categorical features
    onehot_encoder = OneHotEncoder(sparse_output=False, drop=None)
    X_categorical_encoded = onehot_encoder.fit_transform(X[categorical_columns])
    
    # Get feature names
    categorical_feature_names = onehot_encoder.get_feature_names_out(categorical_columns)
    
    # StandardScaler for numeric features
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X[numeric_columns])
    
    # Combine features
    X_processed = np.hstack([X_numeric_scaled, X_categorical_encoded])
    all_feature_names = list(numeric_columns) + list(categorical_feature_names)
    
    print(f"Final processed data shape: {X_processed.shape}")
    print(f"Total features: {len(all_feature_names)}")
    
    return X_processed, y, all_feature_names

def calculate_mutual_information(X, y, feature_names):
    """Calculate mutual information scores for all features"""
    print(f"\n{'='*60}")
    print("CALCULATING MUTUAL INFORMATION SCORES")
    print("="*60)
    
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Create feature-score pairs
    feature_score_pairs = list(zip(feature_names, mi_scores))
    
    # Sort by score in descending order
    feature_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"✓ Calculated mutual information scores for {len(feature_names)} features")
    
    return feature_score_pairs

def display_mutual_information_histogram(feature_score_pairs):
    """Display mutual information scores as histogram in descending order"""
    print(f"\n{'='*60}")
    print("MUTUAL INFORMATION HISTOGRAM")
    print("="*60)
    
    # Extract sorted features and scores
    sorted_features = [pair[0] for pair in feature_score_pairs]
    sorted_scores = [pair[1] for pair in feature_score_pairs]
    
    # Create histogram
    plt.figure(figsize=(14, 10))
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(sorted_features)), sorted_scores, 
                   color='skyblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Shorten feature names for better display
    short_names = []
    for feature in sorted_features:
        if len(feature) > 30:
            short_names.append(feature[:27] + '...')
        else:
            short_names.append(feature)
    
    # Customize the plot
    plt.yticks(range(len(sorted_features)), short_names, fontsize=10)
    plt.xlabel('Mutual Information Score', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title('Feature Importance - Mutual Information Scores (Descending Order)', 
              fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        plt.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Invert y-axis to show highest scores at top
    plt.gca().invert_yaxis()
    
    # Adjust layout
    plt.subplots_adjust(left=0.4, right=0.95, top=0.9, bottom=0.1)
    plt.show()
    
    # Print summary
    print(f"\n✓ Mutual Information Summary:")
    print(f"   Total features: {len(sorted_features)}")
    print(f"   Highest score: {sorted_scores[0]:.4f}")
    print(f"   Lowest score: {sorted_scores[-1]:.4f}")
    print(f"   Score range: {sorted_scores[0] - sorted_scores[-1]:.4f}")
    
    # Print top 10 features
    print(f"\nTop 10 features by Mutual Information:")
    for i in range(min(10, len(sorted_features))):
        print(f"  {i+1:2d}. {sorted_features[i]:30s} {sorted_scores[i]:8.4f}")
    
    # Print bottom 5 features
    print(f"\nBottom 5 features by Mutual Information:")
    for i in range(max(0, len(sorted_features)-5), len(sorted_features)):
        print(f"  {i+1:2d}. {sorted_features[i]:30s} {sorted_scores[i]:8.4f}")

def main():
    """Main function to run mutual information analysis"""
    print("MUTUAL INFORMATION ANALYSIS")
    print("="*60)
    
    # Load and prepare data
    X, y, feature_names = load_and_prepare_data()
    
    # Calculate mutual information scores
    feature_score_pairs = calculate_mutual_information(X, y, feature_names)
    
    # Display histogram
    display_mutual_information_histogram(feature_score_pairs)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("You can now decide on the optimal number of features (K) based on:")
    print("  1. The histogram showing feature importance")
    print("  2. The elbow effect (where scores drop significantly)")
    print("  3. Your domain knowledge and model requirements")

if __name__ == "__main__":
    main()
