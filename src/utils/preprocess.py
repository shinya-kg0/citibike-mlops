import pandas as pd
from enum import Enum, auto

class TimeOfDay(Enum):
    MORNING = auto()
    AFTERNOON = auto()
    NIGHT = auto()


def load_and_clean_data(df_org: pd.DataFrame, max_duration_min: int = 360) -> pd.DataFrame:
    """
    CSVを読み込み、日付変換と基本クリーニングを行う。

    Parameters
    ----------
    df_org : pd.DataFrame
        生データ
    max_duration_min : int, optional
        利用時間の上限（分）。外れ値除外に使用。

    Returns
    -------
    pd.DataFrame
        前処理済みデータ
    """
    df = df_org.copy()

    # 型変換
    df["starttime"] = pd.to_datetime(df["starttime"])
    df["stoptime"] = pd.to_datetime(df["stoptime"])

    # 利用時間（分）
    df["tripduration_min"] = df["tripduration"] / 60
    df = df[df["tripduration_min"] < max_duration_min]

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """時間に関する特徴量を追加"""
    df["start_hour"] = df["starttime"].dt.hour
    df["weekday"] = df["starttime"].dt.weekday

    def categorize_time(hour: int) -> int:
        if 6 <= hour < 12:
            return TimeOfDay.MORNING.value
        elif 12 <= hour < 18:
            return TimeOfDay.AFTERNOON.value
        else:
            return TimeOfDay.NIGHT.value

    df["time_category"] = df["start_hour"].apply(categorize_time)
    return df


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """駅や自転車単位の集約特徴量を追加"""
    # 駅の人気度
    station_usage = df.groupby("start station name").size().reset_index(name="station_usage_count")
    df = df.merge(station_usage, on="start station name", how="left")

    # 自転車の稼働回数
    bike_usage = df.groupby("bikeid").size().reset_index(name="bike_usage_count")
    df = df.merge(bike_usage, on="bikeid", how="left")

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """ターゲット変数 is_member を作成"""
    df["is_member"] = (df["usertype"] == "Subscriber").astype(int)
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """モデル学習用の特徴量を抽出"""
    features = [
        "start_hour",
        "weekday",
        "time_category",
        "tripduration_min",
        "station_usage_count",
        "bike_usage_count",
        "gender",
    ]
    return df[features + ["is_member"]]


def convert_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    MLflowの警告を避けるため、整数型の特徴量をfloat64に変換する。
    ターゲット変数 'is_member' は変換しない。
    """
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]) and col != "is_member":
            df[col] = df[col].astype("float64")
    return df


def preprocess_pipeline(df_org: pd.DataFrame) -> pd.DataFrame:
    """
    CitiBikeデータの前処理を一括で実行する。

    Parameters
    ----------
    df_org : pd.DataFrame
        生データ

    Returns
    -------
    pd.DataFrame
        前処理済みかつ学習用に整形されたDataFrame
    """
    df = load_and_clean_data(df_org)
    df = add_time_features(df)
    df = add_aggregate_features(df)
    df = add_target(df)
    df = select_features(df)
    df_final = convert_to_float(df)
    return df_final