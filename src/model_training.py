from xgboost import XGBClassifier

def train_model(X_train, y_train):

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    model.fit(X_train, y_train)

    return model