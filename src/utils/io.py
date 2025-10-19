from pathlib import Path
import pandas as pd
import re

DATA_DIR = "data"
RAW_DIR = "raw"
INTERIM_DIR = "interim"


def get_project_root() -> Path:
    """
    プロジェクトのルートディレクトリ（data/ や src/ がある階層）を返す。
    Docker内では /app を返す想定。
    """
    cwd = Path.cwd()
    for parent in cwd.parents:
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return cwd


def get_year_dir(year: int) -> Path:
    """
    指定年の生データディレクトリパスを返す。
    """
    root = get_project_root()
    data_dir = root / DATA_DIR / RAW_DIR / f"{year}-citibike-tripdata"
    if not data_dir.exists():
        raise FileNotFoundError(f"Year directory not found: {data_dir}")
    return data_dir


def find_month_dir(year_dir: Path, month: int) -> Path:
    """
    年ディレクトリ内から、指定した月に対応するフォルダを正規表現で検索する。
    例: month=1 → フォルダ名が '1_' で始まるもの（例: '1_January'）
    """
    pattern = re.compile(rf"^{month}_", re.IGNORECASE)
    matches = [p for p in year_dir.iterdir() if p.is_dir() and pattern.match(p.name)]
    
    if not matches:
        raise FileNotFoundError(f"No folder found for month={month} in {year_dir}")
    if len(matches) > 1:
        print(f"複数フォルダが該当しました。最初の1つを使用します: {matches[0].name}")
        
    return matches[0]


def load_month_data(year: int, month: int) -> pd.DataFrame:
    """
    特定の年・月のCSVを読み込む。
    例: load_month_data(2014, 1)
    """
    year_dir = get_year_dir(year)
    month_dir = find_month_dir(year_dir, month)
    
    csv_files = list(month_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {month_dir}")
    
    if len(csv_files) > 1:
        print(f"複数CSVが見つかりました。最初の1件を読み込みます: {csv_files[0].name}")
        
    csv_path = csv_files[0]
    print(f"Loading: {csv_files}")
    return pd.read_csv(csv_path)
