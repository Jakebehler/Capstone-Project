# Predicting Bank Customer Churn with Gradient Boosting

## Project Overview
- Developed an end-to-end machine learning pipeline to identify high-risk bank customers. This project demonstrates a strategic transition from raw data exploration to a deployed predictive tool, specifically highlighting the importance of data integrity and feature validation in a financial services context.

## Key Finding & Model Diagnosis
- The Strategic Pivot: Discovered that predicting "Tenure" was not statistically viable (R^2 ≈0), leading to a strategic pivot toward Binary Classification (Churn).
- Feature Leakage Diagnosis: Identified a 0.996 correlation between Exited and Complain. Recognizing this as data leakage (complaints logged at exit) prevented the deployment of a "perfect" but useless model.
- Final Model: Deployed a Gradient Boosting Classifier achieving ~86% Accuracy and 75% Precision for churn prediction.
- With 75% precision (Predicting Churn), the bank can accurately target high-risk customers with retention offers while minimizing resources spent on customers who weren't actually going to leave.

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
This project delivers a deployable churn prediction tool and targeted retention strategy that enables a bank to proactively identify at-risk customers before they leave, reducing customer acquisition costs and protecting long-term revenue..
