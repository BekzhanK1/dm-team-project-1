# Spotify Churn Prediction - Team Project

## Problem Statement
**Type:** Binary Classification  
**Objective:** Predict whether a Spotify user will churn (cancel subscription)  
**Dataset:** 8,000 users with 11 features  
**Target Variable:** `is_churned` (0 = Not Churned, 1 = Churned)

### Features
- **Demographic:** gender, age, country
- **Subscription:** subscription_type
- **Behavioral:** listening_time, songs_played_per_day, skip_rate, offline_listening
- **Device:** device_type
- **Ads:** ads_listened_per_week

---

## Project Structure

```
dm-team-project-1/
│
├── datasets/
│   └── spotify_churn_dataset.csv      # Raw dataset
│
├── data_preparation.py                 # Common data preprocessing module
├── feature_selection.py                # Feature selection analysis
│
├── student1_models.py                  # Linear Models (2 estimators)
├── student2_models.py                  # Tree-based Models (2 estimators)
├── student3_models.py                  # kNN & MLP Models (2 estimators)
│
├── final_comparison.py                 # Model comparison & test evaluation
└── README.md                           # This file
```

---

## Team Distribution

### Student 1: Linear Models
1. **Logistic Regression** - Basic linear classifier
2. **Logistic Regression with Regularization** - L1 (Lasso) or L2 (Ridge)

### Student 2: Tree-based Models
3. **Decision Tree** - Single decision tree classifier
4. **Ensemble Methods** - Random Forest or Gradient Boosting

### Student 3: Instance-based & Neural Networks
5. **k-Nearest Neighbors (kNN)** - Instance-based learning
6. **Multi-Layer Perceptron (MLP)** - Neural network

---

## How to Use

### Step 1: Data Exploration and Preparation
```bash
python data_preparation.py
```
This will:
- Load and explore the dataset
- Generate visualizations (correlation matrix, distributions)
- Encode categorical variables
- Split into train/validation/test sets (70%/10%/20%)
- Scale numerical features
- Save visualization plots

### Step 2: Feature Selection (Optional)
```bash
python feature_selection.py
```
This will analyze feature importance using:
- SelectKBest (ANOVA F-test)
- Recursive Feature Elimination (RFE)
- Random Forest Feature Importance

### Step 3: Each Student Trains Their Models
```bash
# Student 1
python student1_models.py

# Student 2
python student2_models.py

# Student 3
python student3_models.py
```

Each script will:
- Load preprocessed data from `data_preparation.py`
- Perform systematic cross-validation
- Generate learning curves (training vs validation)
- Save plots and results
- Return the best model for each estimator

### Step 4: Final Comparison and Testing
```bash
python final_comparison.py
```
This will:
- Train all 6 models
- Compare validation accuracies
- Select the best model
- Evaluate on the test set
- Generate comparison visualizations

---

## Requirements

Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Or use the requirements file (if created):
```bash
pip install -r requirements.txt
```

---

## Key Implementation Details

### Data Preprocessing (`data_preparation.py`)
1. **Dropped columns:** user_id (not a feature)
2. **Encoding:** LabelEncoder for categorical variables
   - gender: Female, Male, Other → 0, 1, 2
   - country: AU, CA, DE, FR, IN, PK, UK, US → 0-7
   - subscription_type: Family, Free, Premium, Student → 0-3
   - device_type: Desktop, Mobile, Web → 0-2
3. **Scaling:** StandardScaler for numerical features
4. **Split:** 70% train, 10% validation, 20% test (stratified)

### Model Training
Each model performs:
- **Systematic hyperparameter tuning** to find local optimum
- **Cross-validation** comparing training and validation scores
- **Learning curve visualization** (line graphs)
- **Performance evaluation** with classification reports

### Validation Requirements ✅
- ✅ At least 6 estimators (6 models from 3 students)
- ✅ Cross-validation with systematic tuning for each estimator
- ✅ Training vs validation error comparison graphs
- ✅ Model comparison and best model selection
- ✅ Test set evaluation comparing with validation accuracy

---

## Expected Output

### Plots Generated
1. `correlation_matrix.png` - Correlation heatmap
2. `churn_by_categories.png` - Churn rate by categorical features
3. `feature_distributions.png` - Distribution histograms
4. `feature_importance_*.png` - Feature selection plots
5. `*_learning_curve.png` - Learning curves for each model (6 files)
6. `model_comparison.png` - Bar chart comparing all models
7. `validation_vs_test.png` - Best model validation vs test
8. `Confusion_Matrix_*.png` - Confusion matrix for best model

### Console Output
- Dataset overview and statistics
- Encoding details
- Train/val/test split information
- Hyperparameter tuning progress for each model
- Model evaluation metrics
- Final comparison and best model selection
- Test set performance

---

## Tips for Presentation (8-10 minutes)

### Slide Structure
1. **Problem Statement** (1 min)
   - Classification task, 8000 instances, 11 features
   - Target: Churn prediction

2. **Data Exploration** (1.5 min)
   - Show correlation matrix
   - Churn rate by features
   - Key insights

3. **Data Preprocessing** (1 min)
   - Encoding categorical variables
   - Feature scaling
   - Train/val/test split

4. **Feature Selection** (1 min)
   - Three methods used
   - Key features identified
   - Decision (use all features)

5. **Model Building** (3 min)
   - Show 1-2 learning curves from each student
   - Highlight hyperparameter tuning process
   - Brief results for each model

6. **Model Comparison** (1.5 min)
   - Show comparison bar chart
   - Best model selection
   - Validation accuracy

7. **Test Results** (1 min)
   - Test accuracy
   - Confusion matrix
   - Comparison with validation

8. **Conclusion** (0.5 min)
   - Best model and accuracy
   - Key findings

---

## Notes
- All random states are set to 42 for reproducibility
- Plots are saved with high resolution (300 dpi)
- Each student can work independently on their models
- The `data_preparation.py` ensures consistency across all models
- Run `final_comparison.py` only after all students complete their models
