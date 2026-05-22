import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
 
 
# ── Cache data load ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('Customer-Churn-Records.csv')
    df['Age'] = np.sqrt(df['Age'])
    return df
 
# ── Cache model training ─────────────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    capstone_complain = df.copy().drop(columns=['Complain'])
 
    numerical_complain = ['NumOfProducts', 'CreditScore', 'Age', 'Balance',
                          'EstimatedSalary', 'Point Earned', 'HasCrCard',
                          'IsActiveMember', 'Tenure']
    categorical_complain = ['Geography', 'Gender', 'Card Type']
    ordinal_complain = ['Satisfaction Score']
 
    numerical_pipeline = Pipeline([('scaler', StandardScaler())])
    categorical_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    ordinal_pipeline = Pipeline([('ordinal', OrdinalEncoder()), ('scaler', StandardScaler())])
 
    CT3 = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_complain),
        ('cat', categorical_pipeline, categorical_complain),
        ('ord', ordinal_pipeline, ordinal_complain),
    ])
 
    X = capstone_complain.drop(columns=['Exited'])
    y = capstone_complain['Exited']
 
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
    X_train_t = CT3.fit_transform(X_train)
    X_test_t  = CT3.transform(X_test)
 
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train_t, y_train)
 
    y_pred = model.predict(X_test_t)
 
    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'cm':        confusion_matrix(y_test, y_pred),
    }
 
    return model, CT3, metrics
 
 
# ── Load & train ─────────────────────────────────────────────────────────────
capstone = load_data()
model, transformer, metrics = train_model(capstone)
 
 
# ── Layout containers ────────────────────────────────────────────────────────
header        = st.container()
overview      = st.container()
dataset       = st.container()
findings      = st.container()
repository    = st.container()
links         = st.container()
insight       = st.container()
stack         = st.container()
conclusion    = st.container()
predictor     = st.container()
model_metrics = st.container()
 
 
# ── Header ───────────────────────────────────────────────────────────────────
with header:
    st.title('Predicting Bank Customer Churn with Gradient Boosting')
 
# ── Overview ─────────────────────────────────────────────────────────────────
with overview:
    st.header('Project Overview')
    st.markdown('* This project focuses on analyzing and predicting customer churn for a bank. The primary objective was to build a machine learning model that accurately identifies customers most likely to leave, allowing the bank to implement targeted retention strategies.')
    st.markdown('* The analysis involved extensive Exploratory Data Analysis (EDA), testing multiple models, and diagnosing data leakage to deliver a high-value, realistic predictive solution.')
 
# ── Dataset ──────────────────────────────────────────────────────────────────
with dataset:
    st.header('Customer Churn Dataset')
    st.write(pd.read_csv('Customer-Churn-Records.csv').head())
 
# ── Findings ─────────────────────────────────────────────────────────────────
with findings:
    st.header('Key Finding & Model Diagnosis')
    st.markdown('* **Initial Challenge:** The first goal was predicting Customer Tenure using regression models. Every model failed (r² ~ 0), indicating no meaningful relationship between the features and tenure.')
    st.markdown('* **The Pivot:** The objective was shifted to predicting Customer Churn (Exited).')
    st.markdown('* **Feature Leakage Detected:** Initial models achieved near-perfect accuracy (~99.9%). Diagnosis revealed a 0.996 correlation between the target variable (Exited) and the feature Complain — likely recorded at the time of exit, making it useless for pre-exit prediction.')
    st.markdown('* **Final Robust Model:** After dropping Complain, the Gradient Boosting Classifier delivered strong, realistic performance: **~86% Accuracy** and **~75% Precision**.')
 
# ── Repository ───────────────────────────────────────────────────────────────
with repository:
    st.header('Repository')
    st.markdown('* **Final_Churn_Prediction.ipynb:** Primary Jupyter analysis notebook (EDA, data cleaning, model validation, and the final 86% predictive model).')
    st.markdown('* **Customer-Churn-Records.csv:** The raw customer data used for the analysis.')
    st.markdown('* **Presentation_Slides.pdf:** High-level summary of findings, technical process, and actionable business insights.')
 
# ── Links ────────────────────────────────────────────────────────────────────
with links:
    st.header('Additional Links')
    st.link_button("Tableau Interactive Dashboard", "https://public.tableau.com/app/profile/jake.behler/vizzes")
    st.link_button("PowerPoint Presentation: Summary of findings, technical approach, and business insights", "https://docs.google.com/presentation/d/1R22e3wNlkY9wL_KnU_M1mPMPR_h9Wkqw/edit?slide=id.p1#slide=id.p1")
    st.link_button("Project Overview: The key findings, model diagnosis, and business insights", "https://docs.google.com/document/d/17lfmUc0khKmJ3Gxash4rblpGUCPMCENLmlbJ7ue2VdQ/edit?tab=t.0")
 
# ── Insights ─────────────────────────────────────────────────────────────────
with insight:
    st.header('Actionable Business Insights')
    st.markdown("""
* Customers in Germany churn at double the rate of customers in France and Spain
    * **Targeted Action:** Develop Germany-specific retention programs
""")
    st.markdown("""
* Inactive members show a significantly higher churn rate compared to active members
    * **Targeted Action:** Re-engage inactive members before they leave
""")
    st.markdown("""
* Customers with 3 or 4 products have an extremely high churn rate (up to 100%)
    * **Targeted Action:** Investigate friction points for multi-product customers (e.g., complexity, fees)
""")
    st.markdown("""
* The 45-65 age bracket represents the highest turnover risk
    * **Targeted Action:** Offer specialized support or loyalty rewards to mid-career/older clients
""")
 
# ── Technology Stack ─────────────────────────────────────────────────────────
with stack:
    st.header('Technology Stack')
    st.markdown('* **Python (Jupyter Notebook):** Data manipulation and modeling')
    st.markdown('* **Libraries:** Pandas, NumPy, Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting, SVR, OLS, Lasso)')
    st.markdown('* **Visualization:** Tableau')
    st.markdown('* **Project Delivery:** Structured presentation using Wix and PowerPoint for final stakeholder review')
 
# ── Conclusion ───────────────────────────────────────────────────────────────
with conclusion:
    st.header('Conclusion')
    st.markdown('This project demonstrates a full end-to-end data science process, from EDA and model tuning to feature validation and business storytelling which highlights both technical skill and real-world application.')
 
# ── Customer Churn Predictor ─────────────────────────────────────────────────
with predictor:
    st.header('Customer Churn Predictor')
    st.markdown('Enter a customer profile below to see whether they are at risk of leaving the bank.')
 
    col1, col2, col3 = st.columns(3)
 
    with col1:
        geography     = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
        gender        = st.selectbox('Gender', ['Male', 'Female'])
        card_type     = st.selectbox('Card Type', ['DIAMOND', 'GOLD', 'SILVER', 'PLATINUM'])
        sat_score     = st.selectbox('Satisfaction Score', [1, 2, 3, 4, 5])
 
    with col2:
        age           = st.slider('Age', 18, 92, 38)
        credit_score  = st.slider('Credit Score', 350, 850, 650)
        tenure        = st.slider('Tenure (years)', 0, 10, 5)
        num_products  = st.selectbox('Number of Products', [1, 2, 3, 4])
 
    with col3:
        balance       = st.number_input('Account Balance ($)', min_value=0.0, max_value=300000.0, value=75000.0, step=1000.0)
        salary        = st.number_input('Estimated Salary ($)', min_value=0.0, max_value=200000.0, value=100000.0, step=1000.0)
        points        = st.slider('Points Earned', 100, 1000, 600)
        has_cc        = st.selectbox('Has Credit Card?', ['Yes', 'No'])
        is_active     = st.selectbox('Is Active Member?', ['Yes', 'No'])
 
    if st.button('Predict Churn Risk', type='primary'):
        input_data = pd.DataFrame({
            'CreditScore':      [credit_score],
            'Geography':        [geography],
            'Gender':           [gender],
            'Age':              [np.sqrt(age)],
            'Tenure':           [tenure],
            'Balance':          [balance],
            'NumOfProducts':    [num_products],
            'HasCrCard':        [1 if has_cc == 'Yes' else 0],
            'IsActiveMember':   [1 if is_active == 'Yes' else 0],
            'EstimatedSalary':  [salary],
            'Satisfaction Score': [sat_score],
            'Card Type':        [card_type],
            'Point Earned':     [points],
        })
 
        input_transformed = transformer.transform(input_data)
        churn_prob = model.predict_proba(input_transformed)[0][1]
        churn_pred = model.predict(input_transformed)[0]
 
        st.divider()
        result_col1, result_col2 = st.columns(2)
 
        with result_col1:
            if churn_pred == 1:
                st.error('### HIGH CHURN RISK')
                st.markdown('This customer profile is predicted to **leave the bank**.')
            else:
                st.success('### LOW CHURN RISK')
                st.markdown('This customer profile is predicted to **stay with the bank**.')
 
        with result_col2:
            st.metric(label='Churn Probability', value=f'{churn_prob:.1%}')
 
            if churn_prob >= 0.7:
                st.markdown('**Risk Level:** 🔴 High')
            elif churn_prob >= 0.4:
                st.markdown('**Risk Level:** 🟡 Medium')
            else:
                st.markdown('**Risk Level:** 🟢 Low')
 
        st.divider()
        st.markdown('**Key risk factors to consider based on this profile:**')
        flags = []
        if geography == 'Germany':
            flags.append('German customers churn at twice the rate of French and Spanish customers.')
        if is_active == 'No':
            flags.append('Inactive members are significantly more likely to churn.')
        if num_products >= 3:
            flags.append(f'Customers with {num_products} products have extremely high churn rates.')
        if 45 <= age <= 65:
            flags.append('This customer falls in the highest-risk age bracket (45-65).')
        if flags:
            for flag in flags:
                st.markdown(f'- {flag}')
        else:
            st.markdown('- No major individual risk flags detected for this profile.')
# ── High Risk Customer List ──────────────────────────────────────────────────
high_risk = st.container()

with high_risk:
    st.header('High Risk Customer List')
    st.markdown('Customers from the dataset ranked by predicted churn probability. These are the accounts the bank should prioritize for retention outreach.')

    @st.cache_data
    def get_high_risk_customers(_model, _transformer, df):
        data = df.copy().drop(columns=['Complain', 'Exited', 'RowNumber', 'CustomerId', 'Surname'])
        probs = _model.predict_proba(_transformer.transform(data))[:, 1]
        result = df[['CustomerId', 'Surname', 'Geography', 'Gender', 'Age', 'NumOfProducts',
                      'IsActiveMember', 'Balance', 'Tenure']].copy()
        result['Age'] = (result['Age'] ** 2).round().astype(int)
        result['Churn Probability'] = probs
        result['Risk Level'] = result['Churn Probability'].apply(
            lambda x: '🔴 High' if x >= 0.7 else ('🟡 Medium' if x >= 0.4 else '🟢 Low')
        )
        return result.sort_values('Churn Probability', ascending=False).reset_index(drop=True)

    risk_df = get_high_risk_customers(model, transformer, capstone)

    threshold = st.slider('Minimum Churn Probability', min_value=0, max_value=99, value=70, step=5, format='%d%%')
    threshold = threshold / 100
    filtered = risk_df[risk_df['Churn Probability'] >= threshold].copy()
    filtered['Churn Probability'] = filtered['Churn Probability'].apply(lambda x: f'{x:.1%}')

    st.markdown(f"**{len(filtered)} customers** flagged at or above {threshold:.0%} churn probability.")
    st.dataframe(filtered, use_container_width=True)
 
# ── Model Performance Metrics ────────────────────────────────────────────────
with model_metrics:
    st.header('Model Performance')
    st.markdown('Performance of the final Gradient Boosting model on the held-out test set:')
 
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Accuracy',  f"{metrics['accuracy']:.1%}")
    m2.metric('Precision', f"{metrics['precision']:.1%}")
    m3.metric('Recall',    f"{metrics['recall']:.1%}")
    m4.metric('F1 Score',  f"{metrics['f1']:.1%}")
 
    with st.expander('Confusion Matrix'):
        cm_df = pd.DataFrame(
            metrics['cm'],
            index=['Actual: Stayed', 'Actual: Left'],
            columns=['Predicted: Stayed', 'Predicted: Left']
        )
        st.dataframe(cm_df)
