"""
EV Purchases — Playground Series S6E9
可复用的数据清洗模块 (Reusable cleaning module)
================================================
提供 load_clean_data()，直接返回**带正确 dtype** 的 DataFrame，
避免 CSV round-trip 丢失 int8 / category / 有序属性。

用法:
    from ev_clean import load_clean_data

    train, test, sub = load_clean_data()           # 默认在当前目录找 csv
    train, test, sub = load_clean_data("path/to/") # 指定数据目录

列类型约定:
    - 二值列 (Home_Charging_Possible / Subsidy_Available / Will_Buy_EV) -> int8 (0/1)
    - 有序数值 Environmental_Concern_Level -> int8 (1~5)
    - 有序分类 Range_Anxiety_Level -> ordered category (Low < Medium < High)
    - 名义分类 Gender / City_Type / Current_Car_Type -> category
    - 计数类整数 (Age / Number_of_Cars_Owned / 充电桩数) -> int8
    - Annual_Income_USD / Daily_Commute_km -> float64 (原始即 float)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# 列清单
ID_COL = "id"
TARGET = "Will_Buy_EV"

BINARY_COLS = ["Home_Charging_Possible", "Subsidy_Available"]

NOMINAL_CATS = ["Gender", "City_Type", "Current_Car_Type"]
ORDINAL_CAT = "Range_Anxiety_Level"
ORDINAL_CAT_ORDER = ["Low", "Medium", "High"]

ORDINAL_NUM = "Environmental_Concern_Level"  # 取值 1~5

# 可安全下压到 int8 的计数类整数列
SMALL_INT_COLS = [
    "Age",  # 25~69
    "Number_of_Cars_Owned",  # 1~4
    "Charging_Stations_Near_Home",  # 0~14
    "Charging_Stations_Near_Work",  # 0~19
]

FLOAT_COLS = ["Annual_Income_USD", "Daily_Commute_km"]

# 全特征列（建模时用，不含 id 与 target）
FEATURE_COLS = (
    ["Age"]
    + FLOAT_COLS
    + ["Number_of_Cars_Owned", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work"]
    + [ORDINAL_NUM]
    + NOMINAL_CATS
    + BINARY_COLS
    + [ORDINAL_CAT]
)


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    """就地做类型修正与编码，返回同一 DataFrame（已拷贝）。"""
    df = df.copy()

    # 二值 Yes/No -> 0/1
    for c in BINARY_COLS:
        df[c] = df[c].map({"Yes": 1, "No": 0}).astype("int8")

    # 目标 (仅 train 有)
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype("int8")

    # 有序数值 -> int8
    df[ORDINAL_NUM] = df[ORDINAL_NUM].astype("int8")

    # 计数类整数 -> int8
    for c in SMALL_INT_COLS:
        df[c] = df[c].astype("int8")

    # 名义分类 -> category
    for c in NOMINAL_CATS:
        df[c] = df[c].astype("category")

    # 有序分类 -> ordered category
    df[ORDINAL_CAT] = pd.Categorical(
        df[ORDINAL_CAT], categories=ORDINAL_CAT_ORDER, ordered=True
    )

    return df


def load_clean_data(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载并清洗 train/test/sample_submission，返回带正确 dtype 的 (train, test, sub)。

    data_dir: 数据目录（含 train.csv / test.csv / sample_submission.csv）。
              默认取本模块所在目录。
    """
    data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sub = pd.read_csv(data_dir / "sample_submission.csv")

    train = _encode(train)
    test = _encode(test)
    # sample_submission 无需编码，仅保留原样（id + 概率）

    return train, test, sub


if __name__ == "__main__":
    # 自测：打印 dtype 与目标分布，并做一致性断言
    train, test, sub = load_clean_data()

    print("[dtypes] train")
    print(train.dtypes)
    print("\n[target] positive rate = %.4f" % train[TARGET].mean())
    print("[shapes]", train.shape, test.shape, sub.shape)

    # 断言关键 dtype
    assert train[TARGET].dtype == "int8"
    assert train[ORDINAL_CAT].dtype.ordered is True
    assert train["Gender"].dtype == "category"
    assert (train["Charging_Stations_Near_Home"].max() <= 14)
    print("\n[check] all dtype assertions passed.")
