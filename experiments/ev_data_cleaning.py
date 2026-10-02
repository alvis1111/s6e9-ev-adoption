"""
EV Purchases — Playground Series S6E9
数据清洗第一步 (Data Cleaning Step 1)
================================================
对原始 train.csv / test.csv / sample_submission.csv 做基础清洗：

  1. 加载原始数据
  2. 缺失值检查（本数据无缺失）
  3. 重复行检查（本数据无重复）
  4. 类型修正与编码
       - 目标 Will_Buy_EV:  Yes/No -> 1/0 (int8)
       - 二值列 Home_Charging_Possible / Subsidy_Available: Yes/No -> 1/0 (int8)
       - 有序分类 Range_Anxiety_Level: Low < Medium < High -> ordered category
       - 名义分类 Gender / City_Type / Current_Car_Type -> category
       - 有序数值 Environmental_Concern_Level: float -> int8 (取值 1~5)
  5. 训练/测试 类别一致性校验
  6. 输出清洗后的 CSV: train_clean.csv / test_clean.csv

运行:  python ev_data_cleaning.py
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. 加载
# ---------------------------------------------------------------------------
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

print(f"[load] train {train.shape} | test {test.shape} | sub {sub.shape}")

# ---------------------------------------------------------------------------
# 2. 缺失值
# ---------------------------------------------------------------------------
print("\n[missing]")
print("train nulls:", train.isnull().sum().sum())
print("test  nulls:", test.isnull().sum().sum())

# ---------------------------------------------------------------------------
# 3. 重复行
# ---------------------------------------------------------------------------
print("\n[duplicates]")
print("train dup rows:", train.duplicated().sum())
print("test  dup rows:", test.duplicated().sum())

# ---------------------------------------------------------------------------
# 4. 类型修正与编码
# ---------------------------------------------------------------------------
BINARY_COLS = ["Home_Charging_Possible", "Subsidy_Available"]
TARGET = "Will_Buy_EV"

yes_no = lambda s: s.map({"Yes": 1, "No": 0}).astype("int8")

# 目标编码
train[TARGET] = yes_no(train[TARGET])

# 二值列编码
for c in BINARY_COLS:
    train[c] = yes_no(train[c])
    test[c] = yes_no(test[c])

# 名义分类 -> category
NOMINAL = ["Gender", "City_Type", "Current_Car_Type"]
for c in NOMINAL:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")

# 有序分类 -> ordered category
train["Range_Anxiety_Level"] = pd.Categorical(
    train["Range_Anxiety_Level"], categories=["Low", "Medium", "High"], ordered=True
)
test["Range_Anxiety_Level"] = pd.Categorical(
    test["Range_Anxiety_Level"], categories=["Low", "Medium", "High"], ordered=True
)

# 有序数值 Environmental_Concern_Level: float -> int8 (1~5)
for df in (train, test):
    df["Environmental_Concern_Level"] = df["Environmental_Concern_Level"].astype("int8")

print("\n[dtypes after cleaning]")
print(train.dtypes)

# ---------------------------------------------------------------------------
# 5. 训练/测试 类别一致性校验
# ---------------------------------------------------------------------------
print("\n[category consistency]")
ok = True
for c in NOMINAL + ["Range_Anxiety_Level"]:
    tr, te = set(train[c].unique()), set(test[c].unique())
    match = tr == te
    ok &= match
    print(f"  {c}: {'OK' if match else 'MISMATCH ' + str(tr ^ te)}")
print("  ->", "ALL CONSISTENT" if ok else "INCONSISTENT")

# ---------------------------------------------------------------------------
# 6. 输出
# ---------------------------------------------------------------------------
train.to_csv(DATA_DIR / "train_clean.csv", index=False)
test.to_csv(DATA_DIR / "test_clean.csv", index=False)
print("\n[save] train_clean.csv / test_clean.csv written.")

# ---------------------------------------------------------------------------
# 快速校验目标分布
# ---------------------------------------------------------------------------
print("\n[target distribution]")
print(train[TARGET].value_counts(normalize=True).round(4).to_string())
print("positive rate: %.4f" % train[TARGET].mean())
