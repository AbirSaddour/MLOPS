"""Pipeline ML pour la prédiction du churn (Churn_Modelling)."""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def prepare_data(filepath="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """Charge et prétraite les données.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(filepath)

    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    df = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Geography"])

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un RandomForestClassifier."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle et affiche les métriques. Retourne l'accuracy."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy : {accuracy * 100:.2f} %")
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))

    return accuracy


def save_model(model, filename="classifier.joblib"):
    """Sauvegarde le modèle avec joblib."""
    joblib.dump(model, filename)
    print(f"Modèle sauvegardé dans {filename}")


def load_model(filename="classifier.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    model = joblib.load(filename)
    print(f"Modèle chargé depuis {filename}")
    return model
