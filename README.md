# Predicting Bank Customer Churn with Gradient Boosting

## Project Overview
- This project focuses on analyzing and predicting customer churn for a bank. The primary objective was to build a machine learning model that accurately identifies customers most likely to leave, allowing the bank to implement targeted retention strategies.
- The analysis involved extensive Exploratory Data Analysis (EDA), testing multiple models, and diagnosing data leakage to deliver a high-value, realistic predictive solution.

## Key Finding & Model Diagnosis
- Initial Challenge : The first goal was predicting Customer Tenure (years with the bank) using various regression models. The models consistently failed (r^2 ~ 0), indicating no simple linear or non-linear relationship between features and tenure.
- The Pivot: The objective was successfully shifted to the binary classification task: predicting Customer Exited (Churn).
- Feature Leakage Detected: Initial classification models (Logistic Regression, Random Forest) achieved near-perfect accuracy (~ 99.9%). Diagnosis revealed a strong collinearity (0.996 correlation) between the target variable (Exited) and the feature Complain. This implied the "Complain" feature was likely recorded at the time of exit, rendering the perfect model useless for pre-exit prediction.
- Final, Robust Model: After dropping the confounding Complain feature, the Gradient Boosting Classifier delivered a strong, realistic performance:
- Accuracy: ~86%
- Precision (of predicting churn): ~ 75%
- This model provides high, actionable insight for the bank.

## Repository
- Final_Churn_Prediction.ipynb: Primary Jupyter analysis notebook (EDA, data cleaning, model validation, and the final 86% predictive model.
- Customer-Churn-Records.csv: The raw customer data used for the analysis.
- Presentation_Slides.pdf: High-level summary of findings, technical process, and actionable business insights.

## Additional Links

## Actionable Business Insights
- Customers in Germany churn at double the rate of customers in France and Spain
  - Targeted Action: Develop Germany-specific retention programs
- Inactive Members show a significantly higher rate of churn compared to active members
  - Targeted Action: Re-engage inactive members before they leave
- Customers with 3 or 4 products have an extremely high churn rate (up to 100%)
  - Targeted Action: Investigate friction points for multi-product customers (e.g., complexity, fees)
- The 45-65 age bracket represents the highest turnover risk
  - Targeted Action: Offer specialized support or loyalty rewards to mid-career/older clients
## Technology Stack
- Python(Jupyter Notebook): Data manipulation and modeling
- Libraries: Pandas, NumPy, Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting, SVR, OLS, Lasso)
- Visualization: Tableau (For Visual Analysis)
- Project Delivery: Structured presentation using Wix and PowerPoint for final stakeholder review

## Conclusion
This project demonstrates a full end-to-end data science process, from EDA and model tuning to feature validation and business storytelling which highlights both technical skill and real-world application.
