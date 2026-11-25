"""Tests unitaires de model_pipeline.py."""

from model_pipeline import prepare_data, train_model


def test_prepare_data_split():
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_test) > 0


def test_train_model_predicts():
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    model = train_model(X_train, y_train, n_estimators=10)
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
