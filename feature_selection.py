"""
Feature Selection Analysis
==========================
This module performs feature selection to identify the most important features.
Methods used: SelectKBest, Recursive Feature Elimination, Feature Importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from data_preparation import get_prepared_data


def select_k_best_features(X_train, y_train, feature_names, k=5):
    """
    Select K best features using ANOVA F-statistic
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature names
        k: Number of features to select
    
    Returns:
        selected_features: List of selected feature names
        scores: F-scores for all features
    """
    print("\n" + "="*60)
    print("SELECT K BEST FEATURES (ANOVA F-test)")
    print("="*60)
    
    # Perform SelectKBest
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)
    
    # Get scores and selected features
    scores = selector.scores_
    feature_scores = pd.DataFrame({
        'Feature': feature_names,
        'F-Score': scores
    }).sort_values('F-Score', ascending=False)
    
    print(f"\nFeature Importance (F-Scores):")
    print(feature_scores.to_string(index=False))
    
    selected_features = feature_scores.head(k)['Feature'].tolist()
    print(f"\nTop {k} Selected Features:")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    colors = ['green' if feat in selected_features else 'lightgray' 
             for feat in feature_scores['Feature']]
    plt.barh(feature_scores['Feature'], feature_scores['F-Score'], color=colors)
    plt.xlabel('F-Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance - SelectKBest (ANOVA F-test)', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('feature_importance_selectkbest.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return selected_features, scores


def recursive_feature_elimination(X_train, y_train, feature_names, n_features=5):
    """
    Perform Recursive Feature Elimination
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature names
        n_features: Number of features to select
    
    Returns:
        selected_features: List of selected feature names
    """
    print("\n" + "="*60)
    print("RECURSIVE FEATURE ELIMINATION (RFE)")
    print("="*60)
    
    # Use Logistic Regression as the estimator
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    
    # Perform RFE
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    rfe.fit(X_train, y_train)
    
    # Get selected features and rankings
    feature_ranking = pd.DataFrame({
        'Feature': feature_names,
        'Ranking': rfe.ranking_,
        'Selected': rfe.support_
    }).sort_values('Ranking')
    
    print(f"\nFeature Rankings:")
    print(feature_ranking.to_string(index=False))
    
    selected_features = feature_ranking[feature_ranking['Selected']]['Feature'].tolist()
    print(f"\nTop {n_features} Selected Features (RFE):")
    for i, feat in enumerate(selected_features, 1):
        print(f"  {i}. {feat}")
    
    return selected_features


def feature_importance_from_model(X_train, y_train, feature_names):
    """
    Get feature importance from Random Forest
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature names
    
    Returns:
        feature_importance_df: DataFrame with feature importances
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (RANDOM FOREST)")
    print("="*60)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print(feature_importance_df.to_string(index=False))
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], 
            color='forestgreen')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('feature_importance_random_forest.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance_df


def main():
    """Main feature selection analysis"""
    print("="*60)
    print("FEATURE SELECTION ANALYSIS")
    print("="*60)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, *_ = get_prepared_data()
    
    # Method 1: SelectKBest
    selected_kbest, f_scores = select_k_best_features(X_train, y_train, feature_names, k=5)
    
    # Method 2: Recursive Feature Elimination
    selected_rfe = recursive_feature_elimination(X_train, y_train, feature_names, n_features=5)
    
    # Method 3: Feature Importance from Random Forest
    importance_df = feature_importance_from_model(X_train, y_train, feature_names)
    selected_importance = importance_df.head(5)['Feature'].tolist()
    
    # Summary
    print("\n" + "="*60)
    print("FEATURE SELECTION SUMMARY")
    print("="*60)
    print("\nTop 5 Features by Different Methods:")
    print("\n1. SelectKBest (ANOVA F-test):")
    for i, feat in enumerate(selected_kbest, 1):
        print(f"   {i}. {feat}")
    
    print("\n2. Recursive Feature Elimination:")
    for i, feat in enumerate(selected_rfe, 1):
        print(f"   {i}. {feat}")
    
    print("\n3. Random Forest Importance:")
    for i, feat in enumerate(selected_importance, 1):
        print(f"   {i}. {feat}")
    
    # Find common features
    common_features = set(selected_kbest) & set(selected_rfe) & set(selected_importance)
    print(f"\nCommon features across all methods: {common_features}")
    
    print("\n" + "="*60)
    print("DECISION: Using ALL features for modeling")
    print("="*60)
    print("Reason: All features show reasonable importance and the dataset")
    print("is not high-dimensional. Feature selection can be revisited if")
    print("overfitting is observed during model training.")


if __name__ == "__main__":
    main()

