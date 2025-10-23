import mlflow
import json
from datetime import datetime
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb


def log_model_to_mlflow(model, X_train, model_name: str):
    """モデルをMLflowに保存（フレームワーク自動判定）"""
    if isinstance(model, lgb.LGBMClassifier):
        mlflow.lightgbm.log_model(model, artifact_path="model", input_example=X_train.iloc[:5])
    elif isinstance(model, xgb.XGBClassifier):
        mlflow.xgboost.log_model(model, artifact_path="model", input_example=X_train.iloc[:5])
    else:
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train.iloc[:5])


def log_experiment_to_mlflow(model, df, dataset_params, metrics, params, dataset_info, X_train, experiment_name, model_name):
    """MLflowへの統合的なログ処理"""

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # パラメータ & メトリクス
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if not isinstance(v, list)})
        
        # 特徴量名をファイルとして保存
        feature_path = Path("interim/features.json")
        feature_path.parent.mkdir(exist_ok=True, parents=True)
        with open(feature_path, "w") as f:
            json.dump(dataset_params["feature_names"], f, indent=2)           
        mlflow.log_artifact(str(feature_path), "dataset_info")
        
        # クラス分布を記録
        class_dist_path = Path("interim/class_distribution.json")
        class_dist_path.parent.mkdir(exist_ok=True, parents=True)
        with open(class_dist_path, "w") as f:
            json.dump({
                "train": dataset_params["class_distribution_train"],
                "test": dataset_params["class_distribution_test"]
            }, f, indent=2)
        mlflow.log_artifact(str(class_dist_path), "dataset_info")        
        
        # データセットの追跡（スナップショットを記録）
        try:
            mlflow.log_input(
                mlflow.data.from_pandas(            # type: ignore
                    df,
                    name=f"citibike_data_{datetime.now().strftime('%Y%m%d')}",
                ),
                context="training",
            )
        except Exception as e:
            print(f"Error logging dataset to MLflow: {e}")

        # 混同行列
        cm_path = Path("data/interim/confusion_matrix.json")
        cm_path.parent.mkdir(exist_ok=True, parents=True)
        with open(cm_path, "w") as f:
            json.dump(metrics["confusion_matrix"], f, indent=2)
        mlflow.log_artifact(str(cm_path), "evaluation")

        # データ情報
        feature_path = Path("data/interim/features.json")
        with open(feature_path, "w") as f:
            json.dump(dataset_info["feature_names"], f, indent=2)
        mlflow.log_artifact(str(feature_path), "dataset_info")

        # モデル登録
        log_model_to_mlflow(model, X_train, model_name)

        mlflow.set_tags({
            "model_type": model_name,
            "framework": "sklearn",
            "data_source": dataset_info["data_info"],
        })