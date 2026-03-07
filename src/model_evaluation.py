from sklearn.metrics import accuracy_score
import mlflow
import joblib

def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)

    joblib.dump(model, "models/model.pkl")

    return acc