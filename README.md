# Charity Donor Predictor and Insights

![0_ODzYjDiFGRih9iRX](https://github.com/user-attachments/assets/3367723a-83aa-487e-bbfc-e33f77989c68)

The **Charity Donor Predictor** project aims to predict whether individuals earn more than $50K annually, assisting charities in identifying potential donors. By leveraging various machine learning models, the project analyzes demographic and financial data to build an efficient prediction system.

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

### 5. KMeans Clustering
To improve model accuracy, KMeans clustering was applied to group the data into clusters, and a new column indicating cluster membership was created. The **Elbow Method** was used to identify the optimal number of clusters, with **4 clusters** selected as the best choice. This improved both accuracy and F1 scores across all models.

### 6. Model Selection and Tuning
Four machine learning models were tested and tuned:

#### Decision Tree
- **Training Accuracy**: 88.97%
- **Test Accuracy**: 89.08%
- **Best Parameters**: `max_depth=6`, `min_samples_leaf=6`, `min_samples_split=2`.

#### AdaBoost (Best Model)
- **Training Accuracy**: 90.79%
- **Test Accuracy**: 90.47%
- **F1 Score (Train)**: 79.87%
- **F1 Score (Test)**: 79.76%
- **Best Parameters**: `'learning_rate': [0.3]`, `'n_estimators': [18]`, `'base_estimator': DecisionTreeClassifier`, `max_depth=6`, `min_samples_leaf=6`, `min_samples_split=2`.

#### SVM (Poly Kernel)
- **Training Accuracy**: 88.05%
- **Test Accuracy**: 87.73%
- **Best Parameters**: `C=0.7`, `kernel='poly'`, `degree=3`, `random_state=42`.

#### SVM (RBF Kernel with Grid Search)
- **Training Accuracy**: 89.50%
- **Test Accuracy**: 88.04%
- **F1 Score (Train)**: 77.07%
- **F1 Score (Test)**: 74.02%
- **Best Parameters**: `C=1`, `kernel='rbf'`, `gamma=0.4`.

### 7. Precision and F-Beta Score Considerations
Since the dataset is imbalanced, with fewer individuals earning more than $50K compared to those earning less, **precision** was prioritized to ensure that predictions of potential high-earning donors were accurate.

To give more weight to precision, I calculated the **F-Beta score** with a beta value of 0.5, which emphasizes precision over recall. Below are the results for the **AdaBoost** model, which was the best-performing model:

- **Training Accuracy**: 90.79%
- **Test Accuracy**: 90.47%
- **Training F1 Score**: 79.87%
- **Test F1 Score**: 79.76%
- **Training F-Beta Score (β = 0.5)**: 83.72%
- **Test F-Beta Score (β = 0.5)**: 83.79%

These scores indicate that the model not only performs well in terms of overall accuracy but also balances precision and recall effectively, especially in predicting individuals likely to earn more than $50K.

### 8. Final Model Selection
The **AdaBoost** model was chosen as the best-performing model for this dataset, achieving the highest accuracy and F1 scores.

## Technologies Used
- **Data Analysis and Visualization**: Pandas, Numpy, Matplotlib, Seaborn
- **Data Preprocessing**: Handling missing data, Min-Max Scaler, One-hot Encoding
- **Machine Learning**: Decision Tree, AdaBoost, SVM (Poly and RBF kernels)
- **Clustering**: KMeans for data segmentation
- **Model Evaluation**: Cross-validation, GridSearch, Model tuning
