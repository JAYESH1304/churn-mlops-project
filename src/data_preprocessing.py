import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(df):

    df = df.drop("customerID", axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn_Yes", axis=1)
    y = df["Churn_Yes"]

    return train_test_split(X, y, test_size=0.2, random_state=42)