import mlflow

from src.data_ingestion import load_data
from src.data_preprocessing import preprocess
from src.model_training import train_model
from src.model_evaluation import evaluate_model

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def run_pipeline():

    with mlflow.start_run():

        df = load_data(DATA_PATH)

        X_train, X_test, y_train, y_test = preprocess(df)

        model = train_model(X_train, y_train)

        acc = evaluate_model(model, X_test, y_test)

        print("Model Accuracy:", acc)


if __name__ == "__main__":
    run_pipeline()