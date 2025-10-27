import mlflow
from mlflow.tracking import MlflowClient # type: ignore
from src.utils.io import load_month_data, load_config
from src.utils.preprocess import preprocess_pipeline
from src.train.evaluator import evaluate_model
from src.train.experiment import run_experiment
from src.pipelines.register_best_model import register_best_model


def load_production_model(client: MlflowClient, model_name: str):
    """Model RegistryからProductionモデルを取得"""
    try:
        prod_model = client.get_model_version_by_alias(model_name, "production")
        prod_run_id = prod_model.tags.get("registered_from_run")
        prod_uri = f"runs:/{prod_run_id}/model"
        prod_model_loaded = mlflow.pyfunc.load_model(prod_uri)
        print(f"Loaded current production model (v{prod_model.version})")
        return prod_model_loaded, prod_run_id
    except Exception as e:
        print(f"No production model found, retraining from scratch. ({e})")
        return None, None
    

def inherit_training_params(client: MlflowClient, prod_run_id: str):
    """現行モデルのハイパーパラメータとモデル種別を引き継ぐ"""
    model_name = "logistic_regression"
    default_params = {"max_iter": 500}
    if not prod_run_id:
        return model_name, default_params
    try:
        prod_run = client.get_run(prod_run_id)
        prod_params = prod_run.data.params      # dict[str, str]
        model_name = prod_run.data.tags.get("model_type", model_name)
        
        # 型変換（MLflowはparamsをstrで保存する）
        for k, v in prod_params.items():
            if v.isdigit():
                prod_params[k] = int(v)
            else:
                try:
                    prod_params[k] = float(v)
                except ValueError:
                    pass
        print(f"Inherited params: {prod_params}")
        print(f"Model type: {model_name}")
    except Exception as e:
        print(f"Failed to load inherited params: {e}")
        prod_params = {"max_iter": 500}
        model_name = "logistic_regression"
    
    return model_name, prod_params
        

def compare_performance(old_metrics: dict, new_metrics: dict, threshold: float = 0.01):
    """精度改善を判定"""
    old_f1 = old_metrics.get("f1_score", 0.0)
    new_f1 = new_metrics["test_f1_score"]
    improvement = new_f1 - old_f1
    print(f"Old F1={old_f1:.4f} → New F1={new_f1:.4f} (+{improvement:.4f})")
    return improvement >= threshold, improvement


def retrain_if_needed(year: int, month: int, threshold: float = 0.01):
    """新しいデータで再学習を実施し、精度が改善した場合のみ更新"""
    config = load_config()
    model_name = config["model_name"]
    expriment_name = config["experiment_name"]
    
    client = MlflowClient()
    prod_model, prod_run_id = load_production_model(client, model_name)
    
    df_raw = load_month_data(year, month)
    df = preprocess_pipeline(df_raw)
    X = df.drop("is_member", axis=1)
    y = df["is_member"]
    
    old_metrics = evaluate_model(prod_model, X, y) if prod_model else {"f1_score": 0.0}
    
    model_type, params = inherit_training_params(client, prod_run_id) # type: ignore
    new_metrics = run_experiment(
        data_info=[year, month],
        model_name=model_type,
        params=params,
        experiment_name=expriment_name,
    )

    improved, delta = compare_performance(old_metrics, new_metrics, threshold)
    if improved:
        print(f"Improvement detected (+{delta:.4f}), registering new model...")
        register_best_model()
    else:
        print(f"No significant improvement (+{delta:.4f}), keeping current model.")