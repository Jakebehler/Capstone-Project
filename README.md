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
- With 75% precision (Predicting Churn), the bank can accurately target high-risk customers with retention offers while minimizing resources spent on customers who weren't actually going to leave.
- This model provides high, actionable insight for the bank.

## Actionable Business Insights
**Geography** 
- Germany churns at 2x the rate of France/Spain.
  - Targeted Action: Localize retention offers for the German market.
**Activity**
- Inactive members are significantly higher risk.
  - Targeted Action: Trigger re-engagement campaigns for dormant accounts.
**Product Usage**
- 3-4 product holders have nearly 100% churn.
  - Targeted Action: Investigate fee structures or friction for multi-product users. (e.g., complexity, fees)
**Demographics**
- The 45-65 age bracket is the highest risk.
  - Targeted Action: Targeted loyalty rewards for mid-career/older clients.
## Technical Skills Demonstrated
- Machine Learning: Gradient Boosting, Random Forest, Logistic Regression, Feature Leakage Diagnosis.
- Deployment: Streamlit (Web App), GitHub.
- BI & Analytics: Tableau (Interactive Dashboards), EDA with Python (Pandas/NumPy).
- Communication: Stakeholder-ready PowerPoint & Wix project delivery.

## Additional Links
- [Tableau Interactive Dashboard](https://public.tableau.com/app/profile/jake.behler/vizzes)
- [PowerPoint Presentation](https://docs.google.com/presentation/d/1R22e3wNlkY9wL_KnU_M1mPMPR_h9Wkqw/edit?slide=id.p1#slide=id.p1): Summary of findings, technical approach, and business insights
- [Project Overview](https://docs.google.com/document/d/17lfmUc0khKmJ3Gxash4rblpGUCPMCENLmlbJ7ue2VdQ/edit?tab=t.0): The key findings, model diagnosis, and business insights]
- [Streamlit Application](https://capstoneproject-jakebehler.streamlit.app)

## Repository
- Final_Churn_Prediction.ipynb: Primary Jupyter analysis notebook (EDA, data cleaning, model validation, and the final 86% predictive model.
- Customer-Churn-Records.csv: The raw customer data used for the analysis.
- Presentation_Slides.pdf: High-level summary of findings, technical process, and actionable business insights.

## Conclusion
This project demonstrates a full end-to-end data science process, from EDA and model tuning to feature validation and business storytelling which highlights both technical skill and real-world application.
