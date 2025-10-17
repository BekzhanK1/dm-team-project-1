"""
Student 3: Instance-based and Neural Network Models
===================================================
Implements:
1. k-Nearest Neighbors (kNN)
2. Multi-Layer Perceptron (MLP)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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


def model5_knn():
    """
    Model 5: k-Nearest Neighbors
    Hyperparameter tuning: n_neighbors (k)
    """
    print("\n" + "="*60)
    print("MODEL 5: K-NEAREST NEIGHBORS (kNN)")
    print("="*60)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = get_prepared_data()
    
    # Hyperparameter tuning: k values
    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
    train_scores = []
    val_scores = []
    
    print("\nPerforming hyperparameter tuning...")
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        
        print(f"k={k:2d}: Train={train_score:.4f}, Val={val_score:.4f}")
    
    # Plot learning curve
    plot_learning_curve(k_values, train_scores, val_scores, 
                       'Number of Neighbors (k)', 'k-Nearest Neighbors')
    
    # Find best k
    best_idx = np.argmax(val_scores)
    best_k = k_values[best_idx]
    
    print(f"\nBest k: {best_k}")
    print(f"Best Validation Score: {val_scores[best_idx]:.4f}")
    
    # Train final model
    best_model = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    # Evaluate
    train_acc, val_acc = evaluate_model(best_model, X_train, X_val, y_train, y_val, 
                                        "k-Nearest Neighbors")
    
    return best_model, val_acc


def model6_mlp():
    """
    Model 6: Multi-Layer Perceptron (Neural Network)
    Hyperparameter tuning: hidden_layer_sizes and alpha (L2 regularization)
    """
    print("\n" + "="*60)
    print("MODEL 6: MULTI-LAYER PERCEPTRON (MLP)")
    print("="*60)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = get_prepared_data()
    
    # Hyperparameter tuning: hidden layer sizes
    hidden_layer_configs = [
        (50,),
        (100,),
        (50, 50),
        (100, 50),
        (100, 100),
        (100, 50, 25),
    ]
    
    train_scores = []
    val_scores = []
    config_labels = []
    
    print("\nPerforming hyperparameter tuning (hidden layers)...")
    for config in hidden_layer_configs:
        model = MLPClassifier(hidden_layer_sizes=config, max_iter=500, 
                             random_state=42, early_stopping=True, 
                             validation_fraction=0.1)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        config_labels.append(str(config))
        
        print(f"hidden_layers={config}: Train={train_score:.4f}, Val={val_score:.4f}")
    
    # Plot learning curve
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(config_labels))
    plt.plot(x_pos, train_scores, 'o-', label='Training Score', linewidth=2, markersize=8)
    plt.plot(x_pos, val_scores, 's-', label='Validation Score', linewidth=2, markersize=8)
    plt.xticks(x_pos, config_labels, rotation=15)
    plt.xlabel('Hidden Layer Configuration', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title('MLP: Training vs Validation Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('MLP_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find best configuration
    best_idx = np.argmax(val_scores)
    best_config = hidden_layer_configs[best_idx]
    
    print(f"\nBest hidden_layer_sizes: {best_config}")
    print(f"Best Validation Score: {val_scores[best_idx]:.4f}")
    
    # Fine-tune with alpha (regularization)
    print("\nFine-tuning alpha (L2 regularization) with best architecture...")
    alpha_values = [0.0001, 0.001, 0.01, 0.1]
    train_scores_alpha = []
    val_scores_alpha = []
    
    for alpha in alpha_values:
        model = MLPClassifier(hidden_layer_sizes=best_config, alpha=alpha,
                             max_iter=500, random_state=42, 
                             early_stopping=True, validation_fraction=0.1)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        train_scores_alpha.append(train_score)
        val_scores_alpha.append(val_score)
        
        print(f"alpha={alpha:.4f}: Train={train_score:.4f}, Val={val_score:.4f}")
    
    # Plot alpha tuning
    plt.figure(figsize=(10, 6))
    plt.semilogx(alpha_values, train_scores_alpha, 'o-', label='Training Score', 
                linewidth=2, markersize=8)
    plt.semilogx(alpha_values, val_scores_alpha, 's-', label='Validation Score', 
                linewidth=2, markersize=8)
    plt.xlabel('Alpha (L2 Regularization)', fontsize=12)
    plt.ylabel('Accuracy Score', fontsize=12)
    plt.title('MLP: Alpha Tuning', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('MLP_alpha_tuning.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find best alpha
    best_alpha_idx = np.argmax(val_scores_alpha)
    best_alpha = alpha_values[best_alpha_idx]
    
    print(f"\nBest alpha: {best_alpha}")
    print(f"Best Validation Score: {val_scores_alpha[best_alpha_idx]:.4f}")
    
    # Train final model with best hyperparameters
    best_model = MLPClassifier(hidden_layer_sizes=best_config, alpha=best_alpha,
                               max_iter=500, random_state=42, 
                               early_stopping=True, validation_fraction=0.1)
    best_model.fit(X_train, y_train)
    
    # Evaluate
    train_acc, val_acc = evaluate_model(best_model, X_train, X_val, y_train, y_val, 
                                        "Multi-Layer Perceptron")
    
    return best_model, val_acc


if __name__ == "__main__":
    print("Student 3: Training kNN and Neural Network Models")
    print("="*60)
    
    # Model 5: kNN
    model5, val_acc5 = model5_knn()
    
    # Model 6: MLP
    model6, val_acc6 = model6_mlp()
    
    # Summary
    print("\n" + "="*60)
    print("STUDENT 3 - MODEL SUMMARY")
    print("="*60)
    print(f"Model 5 (k-Nearest Neighbors): Validation Accuracy = {val_acc5:.4f}")
    print(f"Model 6 (Multi-Layer Perceptron): Validation Accuracy = {val_acc6:.4f}")
    
    if val_acc5 > val_acc6:
        print(f"\nBest Model: k-Nearest Neighbors")
    else:
        print(f"\nBest Model: Multi-Layer Perceptron")

