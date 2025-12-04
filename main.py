import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error, classification_report, root_mean_squared_error, roc_auc_score, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.svm import SVC, SVR


header = st.container()
overview = st.container()
dataset = st.container()
findings = st.container()
repository = st.container()
links = st.container()
insight = st.container()
stack = st.container()
conclusion = st.container()
model_training = st.container()


with header:
	st.title('Predicting Bank Customer Churn with Gradient Boosting')

with overview:
	st.header('Project Overview')
	st.markdown('* This project focuses on analyzing and predicting customer churn for a bank. The primary objective was to build a machine learning model that accurately identifies customers most likely to leave, allowing the bank to implement targeted retention strategies.')
	st.markdown('* The analysis involved extensive Exploratory Data Analysis (EDA), testing multiple models, and diagnosing data leakage to deliver a high-value, realistic predictive solution.')

with dataset:
	st.header('Customer Churn Dataset')
	capstone = pd.read_csv('data/Customer-Churn-Records.csv')
	st.write(capstone.head())

with findings:
	st.header('Key Finding & Model Diagnosis')
	st.markdown('* Initial Challenge : The first goal was predicting Customer Tenure (years with the bank) using various regression models. The models consistently failed (r^2 ~ 0), indicating no simple linear or non-linear relationship between features and tenure.')
	st.markdown('* The Pivot: The objective was successfully shifted to the binary classification task: predicting Customer Exited (Churn).')
	st.markdown('* Feature Leakage Detected: Initial classification models (Logistic Regression, Random Forest) achieved near-perfect accuracy (~ 99.9%). Diagnosis revealed a strong collinearity (0.996 correlation) between the target variable (Exited) and the feature Complain. This implied the "Complain" feature was likely recorded at the time of exit, rendering the perfect model useless for pre-exit prediction.')
	st.markdown('* Final, Robust Model: After dropping the confounding Complain feature, the Gradient Boosting Classifier delivered a strong, realistic performance:')
	st.markdown('* Accuracy: ~86%')
	st.markdown('* Precision (of predicting churn): ~ 75%')
	st.markdown('* This model provides high, actionable insight for the bank.')

with repository:
	st.header('Repository')
	st.markdown('* Final_Churn_Prediction.ipynb: Primary Jupyter analysis notebook (EDA, data cleaning, model validation, and the final 86% predictive model.')
	st.markdown('* Customer-Churn-Records.csv: The raw customer data used for the analysis.')
	st.markdown('* Presentation_Slides.pdf: High-level summary of findings, technical process, and actionable business insights.')

with links:
	st.header('Additional Links')
	st.link_button("Tableau Interactive Dashboard", "https://public.tableau.com/app/profile/jake.behler/vizzes")
	st.link_button("PowerPoint Presentation: Summary of findings, technical approach, and business insights", "https://docs.google.com/presentation/d/1R22e3wNlkY9wL_KnU_M1mPMPR_h9Wkqw/edit?slide=id.p1#slide=id.p1")
	st.link_button("Project Overview: The key findings, model diagnosis, and business insights", "https://docs.google.com/document/d/17lfmUc0khKmJ3Gxash4rblpGUCPMCENLmlbJ7ue2VdQ/edit?tab=t.0")
	
with insight:
	st.header('Actionable Business Insights')
	st.markdown("""
		* Customers in Germany churn at double the rate of customers in France and Spain
			* Targeted Action: Develop Germany-specific retention programs
			""")
	st.markdown("""
		* Inactive Members show a significantly higher rate of churn compared to active members
			* Targeted Action: Re-engage inactive members before they leave
			""")
	st.markdown("""
		* Customers with 3 or 4 products have an extremely high churn rate (up to 100%)
			* Targeted Action: Investigate friction points for multi-product customers (e.g., complexity, fees)
			""")
	st.markdown("""
		* The 45-65 age bracket represents the highest turnover risk
			* Targeted Action: Offer specialized support or loyalty rewards to mid-career/older clients
			""")

with stack:
	st.header('Technology Stack')
	st.markdown('* Python(Jupyter Notebook): Data manipulation and modeling')
	st.markdown('* Libraries: Pandas, NumPy, Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting, SVR, OLS, Lasso)')
	st.markdown('* Visualization: Tableau (For Visual Analysis)')
	st.markdown('Project Delivery: Structured presentation using Wix and PowerPoint for final stakeholder review')

with conclusion:
	st.header('Conclusion')
	st.markdown('This project demonstrates a full end-to-end data science process, from EDA and model tuning to feature validation and business storytelling which highlights both technical skill and real-world application.')

with model_training:
	st.header('Model Training')
	st. text( 'Choose hyperparameters for model')

	sel_col, disp_col = st.columns(2)

	max_depth = sel_col.slider ('Max depth of model?', min_value=10, max_value=100, value=20, step=10)

	n_estimators = sel_col.selectbox('Number of estimators', options= [100,200,300, 'No limit'], index = 0)

	capstone_complain = capstone.copy()
	capstone_complain = capstone_complain.drop(columns = ['Complain'])

	sel_col.text('List of features in Data')
	sel_col.write(capstone_complain.columns)

	input_feature = sel_col.text_input('Which features to include', 'All')

	if n_estimators == 'No limit':
		GBR_complain = GradientBoostingClassifier(max_depth=max_depth)
	else:
		GBR_complain= GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators)

	capstone_complain = capstone.copy()
	capstone_complain = capstone_complain.drop(columns = ['Complain'])
	numerical_pipeline = Pipeline([
	    ('scaler', StandardScaler())
	])

	categorical_pipeline = Pipeline([
	    ('onehot', OneHotEncoder())
	])

	ordinal_pipeline = Pipeline([
	    ('ordinal', OrdinalEncoder()),
	    ('scaler', StandardScaler())
	])

	categorical_complain = ['Geography', 'Gender', 'Card Type']
	numerical_complain = ['NumOfProducts','CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Point Earned','HasCrCard', 'IsActiveMember', 'Tenure']
	ordinal_complain = ['Satisfaction Score']

	X_complain = capstone_complain.drop(columns = ['Exited'])
	y_complain = capstone_complain['Exited']

	np.random.seed(42)

	X_train_complain, X_test_complain, y_train_complain, y_test_complain = train_test_split(X_complain, y_complain, test_size = 0.2, random_state = 42)

	CT3 = ColumnTransformer(
	    transformers=[
	        ('num', numerical_pipeline, numerical_complain),
	        ('cat', categorical_pipeline, categorical_complain),
	        ('ord', ordinal_pipeline, ordinal_complain),       
	    ])

	X_train_CT_com = CT3.fit_transform(X_train_complain)
	X_test_CT_com = CT3.fit_transform(X_test_complain)

	GBR_complain = GradientBoostingClassifier()
	GBR_complain.fit(X_train_CT_com, y_train_complain)
	y_pred_com = GBR_complain.predict(X_test_CT_com)
	accuracy = accuracy_score(y_test_complain, y_pred_com)


	disp_col.subheader('Accuracy of Model')
	disp_col.write(accuracy)

	disp_col.subheader('Precision of Model')
	disp_col.write(precision_score(y_test_complain, y_pred_com, zero_division= 0))

	disp_col.subheader('Recall of Model')
	disp_col.write(recall_score(y_test_complain, y_pred_com))

	disp_col.subheader('F1 Score of Model')
	disp_col.write(f1_score(y_test_complain, y_pred_com))

	disp_col.subheader('Confusion Matrix of Model')
	disp_col.write(confusion_matrix(y_test_complain, y_pred_com))

















