# Employee Turnover Prediction

ML pipeline to predict employee turnover for Portobello Tech's HR department — using EDA, K-Means clustering, SMOTE for class imbalance, and cross-validated classification models to identify at-risk employees and guide retention strategy.

## Objective

Predict which employees are likely to leave (`left`) based on satisfaction, performance, workload, and tenure data, and translate model output into actionable, risk-tiered retention strategies.

## Dataset

~15,000 employee records ([HR Analytics dataset](https://www.kaggle.com/liujiaqi/hr-comma-sepcsv)) with features including:

- `satisfaction_level`, `last_evaluation` — employee satisfaction and performance scores
- `number_project`, `average_montly_hours`, `time_spend_company` — workload and tenure
- `Work_accident`, `promotion_last_5years` — binary flags
- `salary`, `sales` (department) — categorical fields
- `left` — **target**, whether the employee left the company

## Approach

1. **Data Quality Check** — Verified no missing values; identified column types (numeric vs. categorical)
2. **EDA**
   - Correlation heatmap across numerical features
   - Distribution plots for satisfaction, evaluation score, and average monthly hours
   - Project-count comparison between employees who left vs. stayed
3. **Clustering** — K-Means (k=3) on employees who left, using `satisfaction_level` and `last_evaluation`, to profile *why* they left
4. **Class Imbalance Handling**
   - One-hot encoded categorical variables (`salary`, `sales`)
   - Stratified 80:20 train/test split (`random_state=123`)
   - Upsampled the training set using **SMOTE**
5. **Modeling** — Trained and evaluated 3 classifiers with 5-fold cross-validation:
   - Logistic Regression
   - Random Forest Classifier
   - Gradient Boosting Classifier
6. **Model Selection** — Compared via classification report, ROC/AUC curves, and confusion matrices
7. **Retention Strategy** — Scored test employees by turnover probability and bucketed into 4 risk zones

## Key Findings

- `satisfaction_level` is the strongest negative predictor of turnover (r = -0.35); `time_spend_company` is positively correlated (r = 0.17)
- Employees who left clustered into 3 distinct profiles:
  - **Burnout:** high evaluation scores but low satisfaction — overworked top performers
  - **Disengaged:** medium satisfaction, low evaluation, low hours — stuck with no growth path
  - *(see notebook for full cluster breakdown)*
- Employees who left tended to handle a slightly higher average project count than those who stayed

### Model Performance (cross-validated accuracy)

| Model | Accuracy |
|---|---|
| Logistic Regression | 83.38% |
| **Gradient Boosting Classifier** | **98.07%** |
| Random Forest Classifier | 98.39% |

**Random Forest and Gradient Boosting both substantially outperform Logistic Regression.** Given the business cost of missing an at-risk employee (a false negative) versus a false alarm, **Recall** was prioritized over Precision when selecting the final model — flagging too many safe employees as at-risk is a far smaller cost than failing to catch someone who's about to leave.

## Retention Strategy: Risk Zones

Using the best model's predicted turnover probability, employees are segmented into:

| Zone | Probability Range | Action |
|---|---|---|
| 🟢 Safe | < 20% | Standard engagement |
| 🟡 Low Risk | 20–60% | Monitor, light-touch check-ins |
| 🟠 Medium Risk | 60–90% | Proactive manager conversations, targeted support |
| 🔴 High Risk | > 90% | Immediate retention intervention |

## Tech Stack

- **Python 3**, Jupyter Notebook
- `pandas`, `numpy` for data manipulation
- `scikit-learn` — `KMeans`, `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, cross-validation, ROC/AUC
- `imbalanced-learn` (SMOTE) for class balancing
- `matplotlib`, `seaborn` for visualization

## Project Structure

```
├── notebooks/
│   └── Employee_Turnover_Analytics.ipynb   # Full EDA, clustering, SMOTE & model comparison
├── data/
│   └── hr_comma_sep.xlsx                    # HR Analytics employee dataset
├── docs/
│   └── problem_statement.docx               # Original course project brief
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn openpyxl
jupyter notebook notebooks/Employee_Turnover_Analytics.ipynb
```

## Author

Sanjay Gosai
*Machine Learning — Course-End Project*
