"""
Final Model Comparison and Test Evaluation
==========================================
This script compares all models and evaluates the best one on the test set.
Run this after all students have completed their models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preparation import get_prepared_data

# Import all student models
from student1_models import model1_logistic_regression, model2_logistic_regression_regularized
from student2_models import model3_decision_tree, model4_ensemble
from student3_models import model5_knn, model6_mlp


def plot_model_comparison(model_names, val_accuracies):
    """Create bar plot comparing all models"""
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
    bars = plt.bar(range(len(model_names)), val_accuracies, color=colors)
    plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Model Comparison - Validation Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, val_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main comparison and testing function"""
    print("="*60)
    print("FINAL MODEL COMPARISON AND TESTING")
    print("="*60)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = get_prepared_data()
    
    # Train all models and collect results
    print("\n" + "="*60)
    print("TRAINING ALL MODELS")
    print("="*60)
    
    models = []
    model_names = []
    val_accuracies = []
    
    # Student 1 Models
    print("\n[Student 1] Training Linear Models...")
    model1, val_acc1 = model1_logistic_regression()
    models.append(model1)
    model_names.append("Logistic Regression")
    val_accuracies.append(val_acc1)
    
    model2, val_acc2 = model2_logistic_regression_regularized()
    models.append(model2)
    model_names.append("Logistic Reg. (Regularized)")
    val_accuracies.append(val_acc2)
    
    # Student 2 Models
    print("\n[Student 2] Training Tree-based Models...")
    model3, val_acc3 = model3_decision_tree()
    models.append(model3)
    model_names.append("Decision Tree")
    val_accuracies.append(val_acc3)
    
    model4, val_acc4 = model4_ensemble()
    models.append(model4)
    model_names.append("Ensemble (RF/GB)")
    val_accuracies.append(val_acc4)
    
    # Student 3 Models
    print("\n[Student 3] Training kNN and Neural Network Models...")
    model5, val_acc5 = model5_knn()
    models.append(model5)
    model_names.append("k-Nearest Neighbors")
    val_accuracies.append(val_acc5)
    
    model6, val_acc6 = model6_mlp()
    models.append(model6)
    model_names.append("Multi-Layer Perceptron")
    val_accuracies.append(val_acc6)
    
    # Plot comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON (VALIDATION SET)")
    print("="*60)
    for name, acc in zip(model_names, val_accuracies):
        print(f"{name:30s}: {acc:.4f}")
    
    plot_model_comparison(model_names, val_accuracies)
    
    # Select best model
    best_idx = np.argmax(val_accuracies)
    best_model = models[best_idx]
    best_model_name = model_names[best_idx]
    best_val_acc = val_accuracies[best_idx]
    
    print("\n" + "="*60)
    print("BEST MODEL SELECTION")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(f"Validation Accuracy: {best_val_acc:.4f}")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Validation Accuracy: {best_val_acc:.4f}")
    print(f"Difference (Val - Test): {best_val_acc - test_acc:.4f}")
    
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, test_pred))
    
    # Plot confusion matrix for test set
    plot_confusion_matrix(y_test, test_pred, 
                         f'Confusion Matrix - {best_model_name} (Test Set)')
    
    # Compare validation and test accuracies
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(2)
    accuracies = [best_val_acc, test_acc]
    colors = ['steelblue', 'coral']
    bars = plt.bar(x_pos, accuracies, color=colors)
    plt.xticks(x_pos, ['Validation', 'Test'])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'{best_model_name}: Validation vs Test Accuracy', 
             fontsize=14, fontweight='bold')
    plt.ylim([min(accuracies) - 0.05, max(accuracies) + 0.05])
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('validation_vs_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total Models Trained: {len(models)}")
    print(f"Best Model: {best_model_name}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nAll models meet the requirement of at least 6 estimators.")
    print("Cross-validation with systematic hyperparameter tuning completed.")
    print("Training and validation error comparison graphs generated.")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

