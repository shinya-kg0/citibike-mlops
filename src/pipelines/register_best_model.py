import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import yaml


def load_config(config_path: str = "config/register_best_model.yaml") -> dict:
    """
    YAML設定ファイルを読み込んで辞書として返す。
    """
    project_root = Path(__file__).resolve().parents[2]  # ← src/ の2つ上がルート
    config_file = project_root / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def get_best_run(experiment_name: str, metric: str = "test_f1_score"):
    """
    Experimentから指定メトリクスが最も高いRunを取得する。
    """
    clinet = MlflowClient()
    experiment = clinet.get_experiment_by_name(experiment_name)
    
    if not experiment:
        raise ValueError(f"Experiment not found: {experiment_name}")
    
    runs = clinet.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError(f"No runs found for experiment: {experiment_name}")
    
    best_run = runs[0]
    print(f"Best run: {best_run.info.run_id} (metric={metric}: {best_run.data.metrics.get(metric):.4f})")
    
    return best_run.info.run_id


def register_model_from_run(run_id: str, model_name: str):
    """
    指定したRun IDのモデルをModel Registryに登録する。
    """
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    
    print(f"Registering model from run: {run_id}")
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    client.set_model_version_tag(
        name=model_name,
        version=result.version,
        key="registered_from_run",
        value=run_id,
    )
    
    print(f"Registered as {result.name} (version={result.version})")
    return result.version   

def update_alias(model_name: str, new_version: str, alias: str = "production"):
    """
    既存のProductionエイリアスを削除し、新しいモデルに付与する。（他のエイリアスの付け替えも対応）
    """
    client = MlflowClient()
    
    # 既存のProductionモデルを確認
    try:
        latest_model = client.get_model_version_by_alias(name=model_name, alias=alias)
        latest_version = latest_model.version
        print(f"Removing existing {alias} alias (v{latest_version})")
        client.delete_registered_model_alias(name=model_name, alias=alias)
    except Exception:
        print(f"No existing {alias} alias found")

    # 新しいモデルにProductionエイリアスを付与
    client.set_registered_model_alias(
        name=model_name,
        version=new_version,
        alias=alias,
    )
    print(f"Set alias {alias} → {model_name} (v{new_version})")


def register_best_model():
    """メイン処理：Experiment内で最良Runを探してRegistryを更新"""
    CONFIG = load_config()
    
    experiment_name = CONFIG["experiment_name"]
    model_name = CONFIG["model_name"]
    metric = CONFIG.get("metric", "test_f1_score")
    alias = CONFIG.get("alias", "production")
    
    print(f"Searching best run from experiment '{experiment_name}'...")
    best_run_id = get_best_run(experiment_name, metric)

    client = MlflowClient()
    try:
        current_prod = client.get_model_version_by_alias(name=model_name, alias=alias)
        current_prod_run_id = current_prod.tags.get("registered_from_run", "")
        current_prod_version = current_prod.version
        print(f"Current Production model: v{current_prod_version} (run_id={current_prod_run_id})")    
    except Exception:
        print("No production model found.")
        current_prod_run_id = None
    
    # 同じrun_idならスキップ
    if current_prod_run_id == best_run_id:
        print("Best run is same as current Production model. No update needed.")
        return
    
    print("Registering best model...")
    new_version = register_model_from_run(best_run_id, model_name)

    print(f"Updating alias '{alias}'...")
    update_alias(model_name, new_version, alias)

    print("Model registration and alias update complete.")      

    
    