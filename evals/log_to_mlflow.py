import json, mlflow

mlflow.set_experiment("ragbench-ablations")

results = json.load(open("results/ablation_results.json"))

for result in results:
    cfg = result["config"]
    with mlflow.start_run(run_name=cfg["name"]):
        mlflow.log_params({
            "chunk_size": cfg["chunk_size"],
            "overlap":    cfg["overlap"],
            "model":      cfg["model"]
        })
        mlflow.log_metrics({
            "recall_at_1":  result["recall@1"],
            "recall_at_3":  result["recall@3"],
            "recall_at_5":  result["recall@5"],
            "recall_at_10": result["recall@10"],
            "mrr":       result["mrr"]
        })

print("Logged all runs to MLflow.")