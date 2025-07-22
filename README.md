# Diabetes Risk Analysis with Machine Learning

This project aims to predict whether an individual is at risk of diabetes using machine learning techniques. The dataset is derived from the NHANES (National Health and Nutrition Examination Survey), and includes demographic, physical examination, laboratory, dietary, and questionnaire data. The goal is to identify individuals at high risk based on features such as age, BMI, glucose levels, and physical activity.

The project involves data preprocessing, feature engineering, model training (Logistic Regression, Random Forest, XGBoost), and evaluation of model performance.

## Table of Contents

1. Project Overview  
2. Dataset  
3. Model Performance Summary  
4. Observations  
5. How to Run the Project  
6. Development and Future Work  
7. Dependencies  
8. License  

## Project Overview

The goal of this project is to develop a machine learning model that can predict whether an individual is at risk of diabetes. Due to the structured and multi-dimensional nature of health data, the project combines multiple datasets from NHANES, including demographic, physical examination, laboratory, dietary, and questionnaire data.

Several classification algorithms were implemented and compared, including Logistic Regression, Random Forest, and XGBoost, in order to identify the most effective model. The complete workflow includes data collection, cleaning, preprocessing, feature engineering, model training, and performance evaluation.

## Dataset

The dataset used in this project is composed of the following components from NHANES:
- **[Kaggle: National Health and Nutrition Examination Survey](https://www.kaggle.com/datasets/cdc/national-health-and-nutrition-examination-survey)**
- **Demographic Data:** Includes age, gender, race/ethnicity, and other personal details.  
- **Physical Examination Data:** Contains measurements such as Body Mass Index (BMI), waist circumference, and blood pressure.  
- **Laboratory Data:** Includes biochemical test results like glucose and insulin levels.  
- **Dietary Data:** Information about participants' dietary intake and nutrition.  
- **Questionnaire Data:** Self-reported health information, including diabetes diagnosis.

These components were integrated using a unique participant identifier to create a unified dataset for machine learning analysis.

## Model Performance Summary

Three classification models were trained and evaluated on the dataset: Logistic Regression, Random Forest, and XGBoost. Below are their performance metrics on the test set:

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.9158   | 0.8655    | 0.7322 | 0.7933   | 0.9503  |
| Random Forest       | 0.9856   | 0.9925    | 0.9419 | 0.9666   | 0.9942  |
| XGBoost             | 0.9885   | 0.9902    | 0.9573 | 0.9735   | 0.9899  |

### Detailed Classification Reports

- **Logistic Regression:**  
  Shows solid performance but lower recall compared to ensemble models.

- **Random Forest:**  
  High overall performance with good balance between precision and recall.

- **XGBoost:**  
  Best performance overall with strong precision and recall metrics.

## Observations

- The near-perfect accuracy of some models may indicate potential overfitting or that the training and test sets are very similar.
- The dataset might be imbalanced, with a dominance of negative (non-diabetic) or positive cases, which can inflate accuracy scores.
- Models like Logistic Regression showed lower recall compared to ensemble methods, suggesting some challenges in detecting all positive cases.
- The high performance of Random Forest and XGBoost suggests ensemble methods are more effective for this task.

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**:
   - Place the dataset files (`demographic.csv`, `examination.csv`, `laboratory.csv`, `dietary.csv`, `questionnaire.csv`) into the project directory.
   - Alternatively, update the file paths in the notebook to point to your local dataset locations.

4. **Run the Notebook**:
   - Open `diabetes.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute the cells sequentially to preprocess the data, train the models, and evaluate the results.

5. **View the Results**:
   - Performance metrics such as accuracy, precision, recall, and F1-score will be printed.
   - A bar chart visualizing model performance will also be displayed.

## Development and Future Work

- **Model Validation**: Use techniques like k-fold cross-validation to ensure the model generalizes well.
- **Handling Class Imbalance**: Apply techniques such as SMOTE or adjusting class weights to address imbalanced data.
- **Hyperparameter Tuning**: Utilize GridSearchCV or RandomizedSearchCV for better model optimization.
- **Explainability**: Integrate SHAP or LIME to interpret the model’s predictions and improve trustworthiness.
- **User Interface**: Build a web interface using Streamlit or Flask for user-friendly deployment.
- **Deployment**: Package the model using Docker and deploy it on a cloud platform (e.g., Heroku, AWS).

## Dependencies

```bash
pandas==2.2.2  
numpy==1.26.4  
matplotlib==3.9.2  
seaborn==0.13.2  
scikit-learn==1.5.2  
xgboost==2.0.3
```

To install all dependencies, run:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## License

This project is licensed under the MIT License.

## Final Note

This README provides a comprehensive summary of the project, explains the rationale behind each analysis and modeling step, and presents actionable suggestions for future improvements. By implementing the recommended enhancements, the model can become more robust, generalizable, and suitable for real-world applications in predicting diabetes risk using NHANES data.

