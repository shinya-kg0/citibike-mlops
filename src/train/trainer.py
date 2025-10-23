import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from typing import Any

def get_model(model_name: str, params: dict | None = None) -> Any:
    """
    モデル名に応じて学習器インスタンスを返す

    Parameters
    ----------
    model_name : str
        モデル名（"logistic_regression" / "decision_tree" / "random_forest" / "lgbm" / "xgboost"）
    params : dict
        モデルのハイパーパラメータ

    Returns
    -------
    model : estimator
    """
    if params is None:
        params = {}

    model_name = model_name.lower()

    if model_name == "logistic_regression":
        return LogisticRegression(**params)

    elif model_name == "decision_tree":
        return DecisionTreeClassifier(**params)

    elif model_name == "random_forest":
        return RandomForestClassifier(**params)

    elif model_name == "lgbm":
        return lgb.LGBMClassifier(**params)

    elif model_name == "xgboost":
        return xgb.XGBClassifier(**params)

    else:
        raise ValueError(f"Unsupported model: {model_name}")
def train_model(model, X_train, y_train):
    """モデル学習"""
    model.fit(X_train, y_train)
    return model