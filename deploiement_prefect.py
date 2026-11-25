"""Déploiement Prefect : planification quotidienne des flows."""

from prefect import serve

from pipeline_prefect import all_flow, code_flow, train_flow, evaluate_flow

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",  # tous les jours à 02:00 (UTC)
            tags=["full-pipeline", "mlops"],
        ),
        code_flow.to_deployment(
            name="ml-pipeline-code",
            tags=["code-quality", "mlops"],
        ),
        train_flow.to_deployment(
            name="ml-pipeline-train",
            tags=["training", "mlops"],
        ),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate",
            tags=["evaluation", "mlops"],
        ),
    )
