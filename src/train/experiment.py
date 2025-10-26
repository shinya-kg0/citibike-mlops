from src.utils.preprocess import preprocess_pipeline
from src.utils.io import load_month_data
from src.train.trainer import get_model, train_model
from src.train.evaluator import evaluate_model_train_test
from src.train.mlflow_logger import log_experiment_to_mlflow
from sklearn.model_selection import train_test_split

def run_experiment(data_info, model_name, params, experiment_name, random_state=42):
    """実験全体の統合関数
    
    実行例）
    metrics = run_experiment(
    data_info=[2014, 1],
    model_name="logistic_regression",
    params={"max_iter": 500, "random_state": 42},
    experiment_name="citibike_membership"
    )
    
    """
    df_org = load_month_data(*data_info)
    df = preprocess_pipeline(df_org)

    X = df.drop("is_member", axis=1)
    y = df["is_member"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    dataset_info = {
        "data_info": data_info,
        "feature_names": X_train.columns.tolist(),
    }
    
    dataset_params = {
        "data_info": data_info,
        "test_size": 0.2,
        "random_state": random_state,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": X_train.shape[1],
        "feature_names": X_train.columns.tolist(),
        "class_distribution_train": y_train.value_counts().to_dict(),
        "class_distribution_test": y_test.value_counts().to_dict(),
    }

    model = get_model(model_name, params)
    model = train_model(model, X_train, y_train)
    metrics = evaluate_model_train_test(model, X_train, X_test, y_train, y_test)
    log_experiment_to_mlflow(model, df, dataset_params, metrics, params, dataset_info, X_train, experiment_name, model_name)

    return metrics