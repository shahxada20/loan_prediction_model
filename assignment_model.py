# Import Libraries and dependencies
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Streamlit Page Setup
st.set_page_config(page_title="Loan Approval Portal", layout='wide', initial_sidebar_state='expanded')
col1, col2 =st.columns(2)

with col1:
    st.title('📊 Instant Risk Assessment')
    st.subheader('Loan approval prediction for personal loan applications')
    st.caption('AI-powered risk assessment utilizing Logistic Regression classification, \
                to predict loan eligibility based on applicant\'s financial profile')    

with col2:
    st.image("https://www.cashe.co.in/wp-content/uploads/2024/06/What-is-a-Loan.png", width=560)


# Load Dataset in cache
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


# Trained model saved in cache
@st.cache_resource
def train_model(df: pd.DataFrame):
    target = "approved" # our target is approved column containing labels
    drop_cols = [target]

    # appending applicant_name to drop_cols
    if "applicant_name" in df.columns:
        drop_cols.append("applicant_name")
    
    # dropping applicant_name column
    x = df.drop(columns = drop_cols)
    y = df[target]

    # segregating categorical and numerical columns
    cat_col = [c for c in ['gender', 'city', 'employment_type', 'bank'] if c in x.columns]
    num_col = [c for c in x.columns if c not in cat_col]

    # applying transformation to numerical columns
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), # applying median transformation to null values
        ("scaler", StandardScaler()) # standard scaling for convergence
    ])

    # applying transformation to categorical columns
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")), # applying mode transformation to null values
        ("onehot", OneHotEncoder(handle_unknown="ignore")) # applying oneHOTEncoder to categorical columns, ignore unknown category on prediction
    ])

    # preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_col), # label encoder for numerical columns
            ("cat", categorical_transformer, cat_col) # oneHotEncoder for categorical columns
        ]
    )

    # Logistic Regression model with 2000 iterations
    model = LogisticRegression(max_iter=2000)

    # classifier pipeline
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)

    # model training with defined labels
    clf.fit(x_train, y_train)

    # prediction by model on test data
    y_pred = clf.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)), 
        "precision": float(precision_score(y_test, y_pred, zero_division=0)), # when we predict how often is it correct?
        "recall": float(recall_score(y_test, y_pred, zero_division=0)), # out of all truly approved, how many did we catch?
        "f1": float(f1_score(y_test, y_pred, zero_division=0)), # balance b/w precision and recall
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist() # TP, TN, FP, FN (2x2)
    }

    return clf, metrics, x_train.columns.tolist()


# streamlit sidebar
st.sidebar.header('load dataset')
csv_path = st.sidebar.text_input(
    "CSV Path",
    value="loan_dataset.csv",
    help="put the path to the dataset csv"
)

# try loading dataset
try:
    df = load_data(csv_path)
except Exception as e:
    st.error(f"could not load csv: {e}")
    st.stop()

st.sidebar.success(f"loaded {len(df):,} rows")



# Train model
st.sidebar.header('Train the model')
train_now = st.sidebar.button("Train / Re-Train")

if train_now:
    st.cache_resource.clear()

clf, metrics, feature_order = train_model(df)

# Layout
colA, colB = st.columns([1,1])

with colA:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Model Metrics")
    st.write({
        "accuracy": round(metrics["accuracy"], 4), 
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4)
    })
    st.write("Confusion Matrix (row: actual [0,1], cols: predicted [0,1])")

with colB:
    cm = np.array(metrics["confusion_matrix"])

    st.subheader("Confusion Matrix Heatmap")
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Pred 0", "Pred 1"], 
                yticklabels=["Actual 0", "Actual 1"], ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)



    st.dataframe(
        pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"]), use_container_width=True
    )

    st.divider()

# Try a prediction
st.title("Try a Prediction")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.subheader("Applicant's Details")
    applicant_name = st.text_input("Applicant Name", value='John Doe')
    gender = st.radio("Gender", ["M", "F"], horizontal=True)
    age = st.slider("Age", 18, 65, 25)
    city = st.selectbox("City", sorted(df["city"].unique().tolist()))

with c2:
    st.subheader("Employment Details")
    employment_type = st.selectbox("Employment Type", sorted(df["employment_type"].unique().tolist()))
    bank = st.selectbox("Bank", sorted(df["bank"].unique().tolist()))
    monthly_income_pkr = st.number_input("Monthly Income", min_value=15000, step=1000)

with c3:
    st.subheader("Loan Details")
    credit_score = st.slider("Credit Score", 300, 900, 500)
    loan_amount_pkr = st.number_input("Loan Amount", min_value=50000, max_value=10000000, step=10000)
    loan_tenure_months = st.number_input("Tenure (months)", min_value=12, max_value=60, step=6)
    #default_history = st.radio("Default History", [0, 1], horizontal=True, format_func=lambda x: "No" if x==0 else "Yes")
    #has_credit_card = st.radio("Credit Card", [0, 1], horizontal=True, format_func=lambda x: "No" if x==0 else "Yes")

with c4:
    st.subheader("Credit History")
    existing_loans = st.number_input("Existing Loan", min_value=0, max_value=3)
    default_history = st.selectbox("Default History", [0, 1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)", index=0)
    has_credit_card = st.selectbox("Credit Card", [0, 1], format_func=lambda x: "No (0)" if x==0 else "Yes (1)", index=0)


#Building model input rows
input_rows = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "city": city,
    "employment_type": employment_type,
    "bank": bank,
    "monthly_income_pkr": monthly_income_pkr,
    "credit_score": credit_score,
    "loan_amount_pkr": loan_amount_pkr,
    "loan_tenure_months": loan_tenure_months,
    "existing_loans": existing_loans,
    "default_history": default_history,
    "has_credit_card": has_credit_card,
}])

input_rows = input_rows[feature_order]
st.divider()

# prediction_button
if st.button("Predict Approval", type="primary"):
    prob = float(clf.predict_proba(input_rows)[:,1][0])
    pred = int(prob >= 0.5)

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if pred == 1:
            st.success(f"**Result for {applicant_name}: APPROVED**")
        else:
            st.error(f"**Result for {applicant_name}: REJECTED**")
    
    with res_col2:
        st.metric(label="Approval Confidence", value=f"{prob:.2%}")

    st.divider()

colA, colB = st.columns(2)

with colA:
    prob = float(clf.predict_proba(input_rows)[:,1][0])
    pred = int(prob >= 0.5)
    
    st.subheader("Prediction Summary")
    st.write(f"Applicant: {applicant_name}")
    st.write(f"Prediction: {'Approved' if pred == 1 else 'Rejected'}")
    st.write(f"Confidence: {prob:.2%}")

with colB:
    st.subheader("Assignment Project Designed by ")
    st.write("Shahzad")
