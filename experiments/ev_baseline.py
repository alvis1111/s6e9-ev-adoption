"""
EV Purchases — Playground Series S6E9
Baseline 模型 (LightGBM + 分层 5-Fold CV, 评估 AUC)
================================================
步骤:
  1. load_clean_data() 载入带正确 dtype 的数据
  2. 特征编码:
       - 名义分类 (Gender/City_Type/Current_Car_Type) -> LightGBM 原生 category
       - 有序分类 Range_Anxiety_Level -> 整数编码 (Low=0 < Medium=1 < High=2)
       - 其余为数值 int8/float
  3. StratifiedKFold(5) 计算 out-of-fold AUC
  4. 全量训练 -> 预测 test -> 生成 submission.csv

运行:  python ev_baseline.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from ev_clean import load_clean_data, FEATURE_COLS, TARGET

DATA_DIR = Path(__file__).resolve().parent
SEED = 42
N_FOLDS = 5

# LightGBM 原生 category 列
LGB_CATEGORY_COLS = ["Gender", "City_Type", "Current_Car_Type"]
# 有序分类 -> 整数编码 (保留顺序)
ORDINAL_MAP = {"Low": 0, "Medium": 1, "High": 2}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """返回可直接喂给 LightGBM 的特征矩阵。"""
    X = df[FEATURE_COLS].copy()
    # 有序分类 -> 整数
    X["Range_Anxiety_Level"] = X["Range_Anxiety_Level"].map(ORDINAL_MAP).astype("int8")
    # 名义分类 -> LightGBM 原生 category
    for c in LGB_CATEGORY_COLS:
        X[c] = X[c].astype("category")
    return X


def main() -> None:
    train, test, sub = load_clean_data(DATA_DIR)

    X = prepare_features(train)
    y = train[TARGET]
    X_test = prepare_features(test)

    print(f"[data] X {X.shape} | y {y.shape} | X_test {X_test.shape}")
    print(f"[target] positive rate = {y.mean():.4f}\n")

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    feature_importance = np.zeros(len(X.columns))

    print("[CV] stratified 5-fold ...")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        d_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=LGB_CATEGORY_COLS)
        d_va = lgb.Dataset(X_va, y_va, reference=d_tr)
        model = lgb.train(
            params,
            d_tr,
            num_boost_round=3000,
            valid_sets=[d_va],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        oof[va_idx] = model.predict(X_va, num_iteration=model.best_iteration)
        test_pred += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
        feature_importance += model.feature_importance("gain") / N_FOLDS

        auc = roc_auc_score(y_va, oof[va_idx])
        print(f"  fold {fold}: AUC = {auc:.5f}  (best_iter = {model.best_iteration})")

    oof_auc = roc_auc_score(y, oof)
    print(f"\n[OOF] mean AUC = {oof_auc:.5f}")

    # 特征重要性
    imp = pd.Series(feature_importance, index=X.columns).sort_values(ascending=False)
    print("\n[feature importance (gain, avg)]")
    print(imp.round(1).to_string())

    # 生成提交
    sub["Will_Buy_EV"] = test_pred
    out = DATA_DIR / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"\n[save] {out}  (n={len(sub)}, pred range [{test_pred.min():.4f}, {test_pred.max():.4f}])")


if __name__ == "__main__":
    main()
