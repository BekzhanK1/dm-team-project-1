"""
Student 1: Linear Models
========================
Implements:
1. Logistic Regression
2. Logistic Regression with Regularization (Ridge/Lasso)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
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


def model1_logistic_regression():
    """
    Model 1: Standard Logistic Regression
    Hyperparameter tuning: max_iter
    """
    print("\n" + "="*60)
    print("MODEL 1: LOGISTIC REGRESSION")
    print("="*60)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = get_prepared_data()
    
    # Hyperparameter tuning: max_iter values
    max_iter_values = [100, 200, 500, 1000, 2000]
    train_scores = []
    val_scores = []
    
    print("\nPerforming hyperparameter tuning...")
    for max_iter in max_iter_values:
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        
        print(f"max_iter={max_iter:4d}: Train={train_score:.4f}, Val={val_score:.4f}")
    
    # Plot learning curve
    plot_learning_curve(max_iter_values, train_scores, val_scores, 
                       'Max Iterations', 'Logistic Regression')
    
    # Find best max_iter
    best_idx = np.argmax(val_scores)
    best_max_iter = max_iter_values[best_idx]
    
    print(f"\nBest max_iter: {best_max_iter}")
    print(f"Best Validation Score: {val_scores[best_idx]:.4f}")
    
    # Train final model with best hyperparameter
    best_model = LogisticRegression(max_iter=best_max_iter, random_state=42)
    best_model.fit(X_train, y_train)
    
    # Evaluate
    train_acc, val_acc = evaluate_model(best_model, X_train, X_val, y_train, y_val, 
                                        "Logistic Regression")
    
    return best_model, val_acc


def model2_logistic_regression_regularized():
    """
    Model 2: Logistic Regression with Regularization
    Hyperparameter tuning: C (inverse regularization strength) and penalty
    """
    print("\n" + "="*60)
    print("MODEL 2: LOGISTIC REGRESSION WITH REGULARIZATION")
    print("="*60)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = get_prepared_data()
    
    # Hyperparameter tuning: C values (inverse of regularization strength)
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    train_scores_l2 = []
    val_scores_l2 = []
    train_scores_l1 = []
    val_scores_l1 = []
    
    print("\nPerforming hyperparameter tuning for L2 (Ridge)...")
    for C in C_values:
        model = LogisticRegression(C=C, penalty='l2', max_iter=1000, random_state=42, solver='lbfgs')
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        train_scores_l2.append(train_score)
        val_scores_l2.append(val_score)
        
        print(f"C={C:7.3f}: Train={train_score:.4f}, Val={val_score:.4f}")
    
    print("\nPerforming hyperparameter tuning for L1 (Lasso)...")
    for C in C_values:
        model = LogisticRegression(C=C, penalty='l1', max_iter=1000, random_state=42, solver='saga')
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        train_scores_l1.append(train_score)
        val_scores_l1.append(val_score)
        
        print(f"C={C:7.3f}: Train={train_score:.4f}, Val={val_score:.4f}")
    
    # Plot learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # L2 Regularization
    ax1.semilogx(C_values, train_scores_l2, 'o-', label='Training Score', linewidth=2, markersize=8)
    ax1.semilogx(C_values, val_scores_l2, 's-', label='Validation Score', linewidth=2, markersize=8)
    ax1.set_xlabel('C (Inverse Regularization Strength)', fontsize=12)
    ax1.set_ylabel('Accuracy Score', fontsize=12)
    ax1.set_title('L2 Regularization (Ridge)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # L1 Regularization
    ax2.semilogx(C_values, train_scores_l1, 'o-', label='Training Score', linewidth=2, markersize=8)
    ax2.semilogx(C_values, val_scores_l1, 's-', label='Validation Score', linewidth=2, markersize=8)
    ax2.set_xlabel('C (Inverse Regularization Strength)', fontsize=12)
    ax2.set_ylabel('Accuracy Score', fontsize=12)
    ax2.set_title('L1 Regularization (Lasso)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Logistic_Regression_Regularized_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find best configuration
    best_idx_l2 = np.argmax(val_scores_l2)
    best_idx_l1 = np.argmax(val_scores_l1)
    
    best_val_l2 = val_scores_l2[best_idx_l2]
    best_val_l1 = val_scores_l1[best_idx_l1]
    
    if best_val_l2 >= best_val_l1:
        best_C = C_values[best_idx_l2]
        best_penalty = 'l2'
        best_solver = 'lbfgs'
        best_val_score = best_val_l2
        print(f"\nBest regularization: L2 (Ridge)")
    else:
        best_C = C_values[best_idx_l1]
        best_penalty = 'l1'
        best_solver = 'saga'
        best_val_score = best_val_l1
        print(f"\nBest regularization: L1 (Lasso)")
    
    print(f"Best C: {best_C}")
    print(f"Best Validation Score: {best_val_score:.4f}")
    
    # Train final model with best hyperparameters
    best_model = LogisticRegression(C=best_C, penalty=best_penalty, max_iter=1000, 
                                    random_state=42, solver=best_solver)
    best_model.fit(X_train, y_train)
    
    # Evaluate
    train_acc, val_acc = evaluate_model(best_model, X_train, X_val, y_train, y_val, 
                                        "Logistic Regression (Regularized)")
    
    return best_model, val_acc


if __name__ == "__main__":
    print("Student 1: Training Linear Models")
    print("="*60)
    
    # Model 1: Logistic Regression
    model1, val_acc1 = model1_logistic_regression()
    
    # Model 2: Regularized Logistic Regression
    model2, val_acc2 = model2_logistic_regression_regularized()
    
    # Summary
    print("\n" + "="*60)
    print("STUDENT 1 - MODEL SUMMARY")
    print("="*60)
    print(f"Model 1 (Logistic Regression): Validation Accuracy = {val_acc1:.4f}")
    print(f"Model 2 (Regularized Logistic Regression): Validation Accuracy = {val_acc2:.4f}")
    
    if val_acc1 > val_acc2:
        print(f"\nBest Model: Logistic Regression")
    else:
        print(f"\nBest Model: Regularized Logistic Regression")

