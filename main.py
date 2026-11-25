"""Point d'entrée : exécute les étapes du pipeline via la ligne de commande."""

import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument(
        "--train", action="store_true", help="Entraîner et sauvegarder le modèle"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Évaluer le modèle sauvegardé"
    )
    parser.add_argument("--data", default="Churn_Modelling.csv", help="Chemin du CSV")
    parser.add_argument("--model", default="classifier.joblib", help="Chemin du modèle")
    args = parser.parse_args()

    if args.prepare:
        X_train, X_test, y_train, y_test = prepare_data(args.data)
        print(f"Données prêtes : train={X_train.shape}, test={X_test.shape}")

    if args.train:
        X_train, X_test, y_train, y_test = prepare_data(args.data)
        model = train_model(X_train, y_train)
        save_model(model, args.model)

    if args.evaluate:
        X_train, X_test, y_train, y_test = prepare_data(args.data)
        model = load_model(args.model)
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
