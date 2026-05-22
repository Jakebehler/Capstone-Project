# Bank Customer Churn Analysis: Predicting At-Risk Customers Before They Leave

**🔴 Live App: [Try the Customer Churn Predictor](https://capstoneproject-jakebehler.streamlit.app)**

## Project Overview
Developed an end-to-end machine learning and business intelligence pipeline to identify high-risk banking customers, highlighting the importance of data integrity and feature validation. Key deliverables: a deployed Streamlit prediction tool and Tableau dashboards.

## Key Finding & Model Diagnosis
- The Strategic Pivot: Discovered that predicting "Tenure" was not statistically predictable (R^2 ≈0), leading to a strategic pivot toward Binary Classification (Churn).
- Feature Leakage Diagnosis: Identified a 0.996 correlation between Exited and Complain. Recognizing this as data leakage (complaints logged at exit) prevented the deployment of a "perfect" but useless model.
- Final Model: Deployed a Gradient Boosting Classifier achieving ~86% Accuracy and 75% Precision, enabling the bank to target high-risk customers with retention offers while minimizing wasted outreach.

## Actionable Business Insights
### **Regional Interventions (Germany)** 
- Customers in Germany churn at 2x the rate of those in France or Spain.
  - Targeted Action: Localize retention offers for the German market.
### **Activity**
- Inactive members are significantly higher risk.
  - Targeted Action: Trigger re-engagement campaigns for dormant accounts.
### **Product Friction Audit (3+ Products)**
- Customers with 3 or 4 products have nearly a 100% churn rate, likely due to high fees or account complexity.
  - Targeted Action: Investigate fee structures or friction for multi-product users.
### **High Risk Demographic Outreach**
- The 45-65 age bracket represents the highest turnover risk.
  - Targeted Action: Design loyalty rewards for mid-career/older clients.

## Technical Skills Demonstrated
- Machine Learning: Gradient Boosting, Random Forest, Logistic Regression (scikit-learn), and Feature Leakage Diagnosis.
- Deployment: Streamlit (Web App), GitHub.
- BI & Analytics: Tableau (Interactive Dashboards), EDA with Python (Pandas/NumPy).
- Communication: Stakeholder-ready PowerPoint & Wix project delivery.

## Resources
- [Streamlit Application](https://capstoneproject-jakebehler.streamlit.app): Deployed prediction tool: enter a customer profile and get a real-time churn risk assessment.
- [Tableau Interactive Dashboard](https://public.tableau.com/app/profile/jake.behler/vizzes): Interactive dashboard visualizing churn patterns by geography, activity, and product usage.
- [Jupyter Notebook](Final_Churn_Prediction.ipynb): Primary Jupyter analysis notebook (EDA, data cleaning, model validation, and the final 86% predictive model.
- [Dataset](Customer-Churn-Records.csv): The raw customer data used for the analysis.

## Conclusion
This project delivers a deployable churn prediction tool and targeted retention strategy that enables a bank to proactively identify at-risk customers before they leave, reducing customer acquisition costs and protecting long-term revenue.
