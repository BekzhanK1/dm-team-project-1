"""
Student 2: Tree-based Models
=============================
Implements:
1. Decision Tree Classifier
2. Ensemble Methods (Random Forest or Gradient Boosting)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from data_preparation import get_prepared_data


def plot_learning_curve(param_values, train_scores, val_scores, param_name, model_name):
    """Plot training and validation scores"""
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, train_scores, 'o-', label='Training Score', linewidth=2, markersize=8)
    plt.plot(param_values, val_scores, 's-', label='Validation Score', linewidth=2, markersize=8)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title(f'{model_name}: Training vs Validation Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_")}_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    """Evaluate model on training and validation sets"""
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"\n{'='*60}")
    print(f"{model_name} - RESULTS")
    print("="*60)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"\nClassification Report (Validation):")
    print(classification_report(y_val, val_pred))
    
    return train_acc, val_acc


def model3_decision_tree():
    """
    Model 3: Decision Tree Classifier
    Hyperparameter tuning: max_depth
    """
    print("\n" + "="*60)
    print("MODEL 3: DECISION TREE CLASSIFIER")
    print("="*60)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = get_prepared_data()
    
    # Hyperparameter tuning: max_depth
    max_depth_values = [3, 5, 7, 10, 15, 20, 25, None]
    train_scores = []
    val_scores = []
    
    print("\nPerforming hyperparameter tuning...")
    for max_depth in max_depth_values:
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        
        depth_str = str(max_depth) if max_depth is not None else "None"
        print(f"max_depth={depth_str:4s}: Train={train_score:.4f}, Val={val_score:.4f}")
    
    # Plot learning curve (use string representations for x-axis)
    depth_labels = [str(d) if d is not None else "None" for d in max_depth_values]
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(depth_labels))
    plt.plot(x_pos, train_scores, 'o-', label='Training Score', linewidth=2, markersize=8)
    plt.plot(x_pos, val_scores, 's-', label='Validation Score', linewidth=2, markersize=8)
    plt.xticks(x_pos, depth_labels)
    plt.xlabel('Max Depth', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title('Decision Tree: Training vs Validation Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Decision_Tree_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find best max_depth
    best_idx = np.argmax(val_scores)
    best_max_depth = max_depth_values[best_idx]
    
    print(f"\nBest max_depth: {best_max_depth}")
    print(f"Best Validation Score: {val_scores[best_idx]:.4f}")
    
    # Train final model
    best_model = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
    best_model.fit(X_train, y_train)
    
    # Evaluate
    train_acc, val_acc = evaluate_model(best_model, X_train, X_val, y_train, y_val, 
                                        "Decision Tree")
    
    return best_model, val_acc


def model4_ensemble():
    """
    Model 4: Ensemble Method (Random Forest and Gradient Boosting)
    Hyperparameter tuning: n_estimators
    """
    print("\n" + "="*60)
    print("MODEL 4: ENSEMBLE METHODS")
    print("="*60)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = get_prepared_data()
    
    # Hyperparameter tuning for Random Forest
    n_estimators_values = [10, 50, 100, 150, 200]
    train_scores_rf = []
    val_scores_rf = []
    
    print("\n[A] Random Forest - Hyperparameter tuning...")
    for n_est in n_estimators_values:
        model = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        train_scores_rf.append(train_score)
        val_scores_rf.append(val_score)
        
        print(f"n_estimators={n_est:3d}: Train={train_score:.4f}, Val={val_score:.4f}")
    
    # Hyperparameter tuning for Gradient Boosting
    train_scores_gb = []
    val_scores_gb = []
    
    print("\n[B] Gradient Boosting - Hyperparameter tuning...")
    for n_est in n_estimators_values:
        model = GradientBoostingClassifier(n_estimators=n_est, random_state=42)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        train_scores_gb.append(train_score)
        val_scores_gb.append(val_score)
        
        print(f"n_estimators={n_est:3d}: Train={train_score:.4f}, Val={val_score:.4f}")
    
    # Plot learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Random Forest
    ax1.plot(n_estimators_values, train_scores_rf, 'o-', label='Training Score', linewidth=2, markersize=8)
    ax1.plot(n_estimators_values, val_scores_rf, 's-', label='Validation Score', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Estimators', fontsize=12)
    ax1.set_ylabel('Accuracy Score', fontsize=12)
    ax1.set_title('Random Forest', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Gradient Boosting
    ax2.plot(n_estimators_values, train_scores_gb, 'o-', label='Training Score', linewidth=2, markersize=8)
    ax2.plot(n_estimators_values, val_scores_gb, 's-', label='Validation Score', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Estimators', fontsize=12)
    ax2.set_ylabel('Accuracy Score', fontsize=12)
    ax2.set_title('Gradient Boosting', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Ensemble_Methods_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find best configuration
    best_idx_rf = np.argmax(val_scores_rf)
    best_idx_gb = np.argmax(val_scores_gb)
    
    best_val_rf = val_scores_rf[best_idx_rf]
    best_val_gb = val_scores_gb[best_idx_gb]
    
    if best_val_rf >= best_val_gb:
        best_n_est = n_estimators_values[best_idx_rf]
        best_model = RandomForestClassifier(n_estimators=best_n_est, random_state=42, n_jobs=-1)
        model_name = "Random Forest"
        best_val_score = best_val_rf
    else:
        best_n_est = n_estimators_values[best_idx_gb]
        best_model = GradientBoostingClassifier(n_estimators=best_n_est, random_state=42)
        model_name = "Gradient Boosting"
        best_val_score = best_val_gb
    
    print(f"\nBest Ensemble Method: {model_name}")
    print(f"Best n_estimators: {best_n_est}")
    print(f"Best Validation Score: {best_val_score:.4f}")
    
    # Train final model
    best_model.fit(X_train, y_train)
    
    # Evaluate
    train_acc, val_acc = evaluate_model(best_model, X_train, X_val, y_train, y_val, 
                                        f"Ensemble ({model_name})")
    
    return best_model, val_acc


if __name__ == "__main__":
    print("Student 2: Training Tree-based Models")
    print("="*60)
    
    # Model 3: Decision Tree
    model3, val_acc3 = model3_decision_tree()
    
    # Model 4: Ensemble
    model4, val_acc4 = model4_ensemble()
    
    # Summary
    print("\n" + "="*60)
    print("STUDENT 2 - MODEL SUMMARY")
    print("="*60)
    print(f"Model 3 (Decision Tree): Validation Accuracy = {val_acc3:.4f}")
    print(f"Model 4 (Ensemble): Validation Accuracy = {val_acc4:.4f}")
    
    if val_acc3 > val_acc4:
        print(f"\nBest Model: Decision Tree")
    else:
        print(f"\nBest Model: Ensemble")

