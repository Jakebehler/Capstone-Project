# Machine Learning Capstone Project: Bank Customer Churn Predictions

This project focuses on analyzing and predicting customer churn for a bank. The primary objective was to build a machine learning model that accurately identifies customers most likely to leave, allowing the bank to implement targeted retention strategies.

The analysis involved extensive Exploratory Data Analysis (EDA), testing multiple models, and diagnosing data leakage to deliver a high-value, realistic predictive solution.

## Key Finding & Model Diagnosis

Initial Challenge : The first goal was predicting Customer Tenure (years with the bank) using various regression models. The models consistently failed (r^2 ~ 0), indicating no simple linear or non-linear relationship between features and tenure.

The Pivot: The objective was successfully shifted to the binary classification task: predicting Customer Exited (Churn).

Feature Leakage Detected: Initial classification models (Logistic Regression, Random Forest) achieved near-perfect accuracy (~ 99.9%). Diagnosis revealed a strong collinearity (0.996 correlation) between the target variable (Exited) and the feature Complain. This implied the "Complain" feature was likely recorded at the time of exit, rendering the perfect model useless for pre-exit prediction.

Final, Robust Model: After dropping the confounding Complain feature, the Gradient Boosting Classifier delivered a strong, realistic performance:

Accuracy: ~86%

Precision (of predicting churn): ~ 75%

This model provides high, actionable insight for the bank.

## Repository 
- Final_Churn_Prediction.ipynb: Primary Analysis Notebook. Contains all data cleaning, preprocessing pipelines (ColumnTransformer), model testing (Regression and Classification), OLS analysis, and the final ~86% churn model.
- Customer-Churn-Records.csv: The raw customer data used for the analysis.
- Presentation_Slides.pdf 

## Additinal Links
Tableau Interactive Dashboard
- https://public.tableau.com/app/profile/jake.behler/vizzes
PowerPoint Presentation
- https://docs.google.com/presentation/d/1R22e3wNlkY9wL_KnU_M1mPMPR_h9Wkqw/edit?usp=share_link&ouid=106356259358977809628&rtpof=true&sd=true
Project Overview
- https://docs.google.com/document/d/17lfmUc0khKmJ3Gxash4rblpGUCPMCENLmlbJ7ue2VdQ/edit?usp=share_link

## Technology Stack
- Python(Jupyter Notbook): Data manipulation and modeling 
- Libraries: Pandas, NumPy, Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting, SVR, OLS, Lasso)
- Visualization: Tableau (For Visual Analysis)
- Project Delivery: Structured presentation using Wix and PowerPoint for final stakeholder review
- Wix

## Results Summary
- Achieved 99.9% accuracy (F1-score: 0.997) using Logistic Regression  
- Identified “Complain” as the key churn driver (r = 0.996)  
- Built Tableau dashboard highlighting churn by demographics, geography, and activity level  


