# Charity Donor Predictor

The **Charity Donor Predictor** project aims to predict whether individuals earn more than $50K annually, assisting charities in identifying potential donors. By leveraging various machine learning models, the project analyzes demographic and financial data to build an efficient prediction system.

![0_ODzYjDiFGRih9iRX](https://github.com/user-attachments/assets/3367723a-83aa-487e-bbfc-e33f77989c68)

## Project Workflow

### 1. Data Exploration (EDA)
The project began with an exploration of the data to understand its structure and distributions. Visualizations were created to uncover patterns and insights, particularly for individuals earning more than $50K.

### 2. Data Cleaning and Preprocessing
Key steps in data cleaning and preprocessing:
- Addressed missing values.
- Corrected the income column format from `<=50` and `>50` to `0` and `1`.
- Stripped whitespace from categorical columns:
    ```python
    for col in obj.columns:
        data[col] = data[col].str.strip()
    ```
- Converted categorical variables into dummy variables for modeling.

### 3. Data Analysis and Insights
Key insights for individuals earning more than $50K:
- **Average age**: 44.
- **Most common work class**: Private; least common: Without-pay (2 individuals, likely retirees).
  - These individuals were aged 50 and 55.
  - Their characteristics: 
    - Relationship: Husband and Own-child.
    - Education level: HS-grad.
    - Marital status: Married-cv-spous.
    - Race: White.
    - Country: USA.
    - Occupation: Handlers-cleaners and Machine-op-inspct.
- **Top education levels**: HS-grad and Bachelor’s degree.
- **Marital status**: Most are married-cv-spous.
- **Top occupations**: Executive-managerial and Professional-specialty.
- **Predominant race**: White.
- **Top country**: USA.
- **Average weekly work hours**: Higher than those earning below $50K.
- **Higher capital gain/loss** than lower earners.
- Self-employed individuals (inc. and not inc.) show higher average work week hours and capital gain.

### 4. Data Preparation for Modeling
- Addressed skewed data distributions.
- Applied Min-Max scaling for normalization.
- Used one-hot encoding for categorical data.
- Split data into training and testing sets.

### 5. Model Selection and Tuning
Three machine learning models were tested and tuned:

#### Decision Tree
- **Training Accuracy**: 85.23%
- **Test Accuracy**: 85.17%
- **Best Parameters**: `max_depth=6`, `min_samples_leaf=6`, `min_samples_split=2`.

#### AdaBoost
- **Training Accuracy**: 87.71%
- **Test Accuracy**: 87.22%
- **F1 Score (Train)**: 72.43%
- **F1 Score (Test)**: 71.89%
- **Best Parameters**: `'learning_rate': [0.3]`, `'n_estimators': [18]`, `'base_estimator': DecisionTreeClassifier`, `max_depth=6`, `min_samples_leaf=6`, `min_samples_split=2`.

#### SVM (Poly Kernel)
- **Training Accuracy**: 84.77%
- **Test Accuracy**: 84.31%
- **Best Parameters**: `C=0.7`, `kernel='poly'`, `degree=3`, `random_state=42`.

#### SVM (RBF Kernel with Grid Search)
- **Best Parameters**: `C=1`, `kernel='rbf'`, `gamma=0.1`.
- **Training Accuracy**: 84.42%
- **Test Accuracy**: 84.01%
- **F1 Score (Train)**: 64.39%
- **F1 Score (Test)**: 64.08%.

### 6. Final Model Selection
The **AdaBoost** model was chosen as the best-performing model for this dataset, achieving the highest accuracy and F1 scores.

## Technologies Used
- **Data Analysis and Visualization**: Pandas, Numpy, Matplotlib, Seaborn
- **Data Preprocessing**: Handling missing data, Min-Max Scaler, One-hot Encoding
- **Machine Learning**: Decision Tree, AdaBoost, SVM (Poly and RBF kernels)
- **Model Evaluation**: Cross-validation, GridSearch, Model tuning
