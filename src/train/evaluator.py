from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """学習・テスト両方のメトリクスを計算"""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "train_f1_score": f1_score(y_train, y_pred_train, zero_division=0),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "test_precision": precision_score(y_test, y_pred_test, zero_division=0),
        "test_recall": recall_score(y_test, y_pred_test, zero_division=0),
        "test_f1_score": f1_score(y_test, y_pred_test, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_test).tolist(),
    }
    return metrics