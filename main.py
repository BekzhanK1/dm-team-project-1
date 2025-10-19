from data_preparation import get_prepared_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Get the prepared data
X_train, X_val, X_test, y_train, y_val, y_test, feature_names, encoders, scaler = get_prepared_data()

# Convert NumPy arrays to DataFrame for correlation analysis
df_corr = pd.DataFrame(X_train, columns=feature_names)
df_corr['is_churned'] = y_train.values

print("Data types after mixed encoding:")
print(f"X_train type: {type(X_train)}")
print(f"X_train shape: {X_train.shape}")
print(f"Feature names: {feature_names}")
print(f"DataFrame shape: {df_corr.shape}")

# Compute correlation matrix
corr_matrix = df_corr.corr()

# Plot correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Matrix - Mixed Encoded Data', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()