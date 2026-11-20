import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Smartphone Usage Prediction", page_icon="📱", layout="wide")

st.title("📱 Smartphone Usage Prediction Model")
st.write("Upload a smartphone-usage CSV file to train a machine-learning model that predicts the Usage Category.")

uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded is None:
    st.info("Upload your CSV file to begin. The dataset must contain a Usage_Category column.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

target = "Usage_Category"
if target not in df.columns:
    st.error("The uploaded dataset must contain a column named Usage_Category.")
    st.write("Available columns:", list(df.columns))
    st.stop()

df = df.dropna(subset=[target]).copy()
df[target] = df[target].astype(str).str.strip()
df = df[df[target] != ""]
if len(df) < 10:
    st.error("At least 10 rows with a Usage_Category value are recommended.")
    st.stop()

X = df.drop(columns=[target])
# Exclude ID-like columns because they do not describe usage behaviour.
id_cols = [c for c in X.columns if c.lower() in {"user_id", "id", "index"}]
X = X.drop(columns=id_cols, errors="ignore")
y = df[target]

# Remove columns that are completely empty or contain only one value.
usable = [c for c in X.columns if X[c].notna().any() and X[c].nunique(dropna=True) > 1]
X = X[usable]
if X.shape[1] == 0:
    st.error("No useful input columns were found.")
    st.stop()

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]
transformers = []
if numeric_cols:
    transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols))
if categorical_cols:
    transformers.append(("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols))

preprocessor = ColumnTransformer(transformers=transformers)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"))
])

class_counts = y.value_counts()
stratify = y if class_counts.min() >= 2 else None
test_count = max(1, round(len(df) * 0.2))
if len(df) - test_count < 2:
    st.error("Not enough rows for a train/test split.")
    st.stop()

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

accuracy = accuracy_score(y_test, predictions)
a, b, c = st.columns(3)
a.metric("Dataset rows", len(df))
b.metric("Usage categories", y.nunique())
c.metric("Test accuracy", f"{accuracy:.1%}")

st.subheader("Model Evaluation")
st.caption("Accuracy is measured on the held-out test split; results can vary with small datasets.")
st.text(classification_report(y_test, predictions, zero_division=0))

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, colorbar=False)
ax.set_title("Confusion Matrix")
st.pyplot(fig)
plt.close(fig)

st.subheader("Make a Prediction")
st.write("Enter values for one user. The model will estimate the usage category.")
user_input = {}
for col in X.columns:
    if col in numeric_cols:
        vals = pd.to_numeric(X[col], errors="coerce").dropna()
        default = float(vals.median()) if not vals.empty else 0.0
        user_input[col] = st.number_input(col.replace("_", " "), value=default)
    else:
        choices = sorted(X[col].dropna().astype(str).unique().tolist())
        user_input[col] = st.selectbox(col.replace("_", " "), choices)

if st.button("Predict Usage Category", type="primary"):
    input_df = pd.DataFrame([user_input], columns=X.columns)
    predicted = model.predict(input_df)[0]
    st.success(f"Predicted Usage Category: {predicted}")
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        probability_df = pd.DataFrame({
            "Category": model.named_steps["classifier"].classes_,
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)
        st.dataframe(probability_df, use_container_width=True)

st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)
st.download_button("Download dataset preview", df.head(20).to_csv(index=False), "dataset_preview.csv", "text/csv")

st.caption("Educational demonstration only. Predictions depend on the quality and size of the uploaded dataset.")
