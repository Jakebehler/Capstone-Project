# Executive Summary: Bank Customer Churn Analysis

## Project Overview
- Developed an end-to-end machine learning and business intelligence pipeline to identify high-risk banking customers. Key focus areas: data integrity, feature validation, and a deployed Streamlit prediction tool backed by Tableau dashboards.

## Key Finding & Model Diagnosis
- The Model: Final model achieved 86% accuracy and 75% precision on churn prediction, enabling the bank to target high-risk customers with retention offers while minimizing wasted outreach.
- The Findings: Initial testing showed that predicting how long a customer stays (Tenure) was not reliable. I pivoted the strategy to focus on a binary Stay vs. Leave model, which provides much higher business value.
- Leakage Warning: I identified that Customer Complaints were likely being recorded at the moment of exit. To ensure the model remains predictive, I adjusted the analysis to focus on Pre-Exit indicators.

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
