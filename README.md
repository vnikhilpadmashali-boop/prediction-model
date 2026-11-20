# VINIX Task 4 – Create a Prediction Model

## Project
Smartphone Usage Prediction Model

## Objective
Build a machine-learning classification application that predicts a user's `Usage_Category` from smartphone usage information.

## Technologies
- Python
- Streamlit
- Pandas
- Scikit-learn
- Random Forest Classifier
- Matplotlib

## Dataset format
Upload a CSV file containing a target column named `Usage_Category`. The app uses the remaining suitable columns as model inputs and excludes ID columns such as `User_ID`.

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Features
- CSV dataset upload
- Data preview and validation
- Missing-value imputation
- Random Forest classification
- Train/test evaluation and accuracy
- Classification report and confusion matrix
- Interactive prediction form
- Predicted category and class probabilities

## Note
This is an educational model. Model performance depends on dataset size, class balance, and data quality. The app reports test accuracy rather than promising a fixed accuracy.
