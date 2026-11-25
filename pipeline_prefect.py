"""Pipeline Prefect : orchestration des étapes du projet ML (Churn)."""

import argparse
import subprocess
import sys

from prefect import flow, task

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "classifier.joblib"
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run_command(command):
    """Exécute une commande et fait échouer la task en cas d'erreur."""
    subprocess.run([sys.executable, "-m", *command], check=True)


# ---------------------------- TASKS : code ----------------------------
@task(name="Installer les dépendances")
def install_dependencies():
    run_command(["pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code")
def format_code():
    run_command(["black", *CODE_FILES])


@task(name="Qualité du code")
def check_quality():
    run_command(["flake8", "--max-line-length", "88", *CODE_FILES])


@task(name="Sécurité du code")
def check_security():
    run_command(["bandit", "-ll", "-q", *CODE_FILES])


@task(name="Tests unitaires")
def run_tests():
    run_command(["pytest", "-q", "test_model_pipeline.py"])


# ---------------------------- TASKS : modèle ----------------------------
@task(name="Préparation des données")
def prepare():
    return prepare_data(DATA_PATH)


@task(name="Entraînement du modèle")
def train(data):
    X_train, X_test, y_train, y_test = data
    return train_model(X_train, y_train)


@task(name="Sauvegarder le modèle")
def save(model):
    save_model(model, MODEL_PATH)


@task(name="Charger le modèle")
def load():
    return load_model(MODEL_PATH)


@task(name="Évaluer le modèle")
def evaluate(model, data):
    X_train, X_test, y_train, y_test = data
    return evaluate_model(model, X_test, y_test)


# ---------------------------- FLOWS ----------------------------
@flow(name="code")
def code_flow():
    install_dependencies()
    format_code()
    check_quality()
    check_security()
    run_tests()


@flow(name="all")
def all_flow():
    code_flow()
    data = prepare()
    model = train(data)
    save(model)
    evaluate(model, data)


@flow(name="train")
def train_flow():
    data = prepare()
    model = train(data)
    save(model)


@flow(name="evaluate")
def evaluate_flow():
    data = prepare()
    model = load()
    evaluate(model, data)


# ---------------------------- LANCEMENT ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument(
        "--flow",
        choices=["all", "code", "train", "evaluate"],
        default="all",
        help="Flow à exécuter",
    )
    args = parser.parse_args()

    flows = {
        "all": all_flow,
        "code": code_flow,
        "train": train_flow,
        "evaluate": evaluate_flow,
    }
    flows[args.flow]()
