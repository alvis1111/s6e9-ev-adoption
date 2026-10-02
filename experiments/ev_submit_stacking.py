"""
EV Purchases — Playground Series S6E9
Stacking 集成 (无伪标签)
================================================
第一层 5 个 base (刻意多样化):
  1. LGB            (GBDT,  leaves31 lr0.05)
  2. XGB            (GBDT,  depth6)
  3. CatBoost       (GBDT,  depth6)
  4. RandomForest   (bagging 系, 和 boosting 差异大)
  5. LogisticReg    (线性系, 和树模型差异最大)

第二层 meta: LogisticRegression (在 5 个 base 的 OOF 上, 简单稳健)
  - meta 也做 5 折 CV, 报告真实 OOF (避免过拟合假象)

全部无伪标签 -> 不引入 test 信息。限线程防卡死, 实时输出。
输出: submission_stacking.csv
运行: python -u ev_submit_stacking.py
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ev_clean import load_clean_data, TARGET
from ev_features import build_features, LGB_CATEGORY_COLS

DATA_DIR = Path(__file__).resolve().parent
SEED = 42
N_FOLDS = 5
CAT_COLS = LGB_CATEGORY_COLS + ["Income_bin"]
OUT_NAME = "submission_stacking.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


def lgb_params() -> dict:
    return dict(objective="binary", metric="auc", learning_rate=0.05,
                num_leaves=31, max_depth=-1, min_child_samples=100,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=SEED, n_jobs=8, verbose=-1)


def xgb_params() -> dict:
    return dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.05,
                max_depth=6, min_child_weight=50, subsample=0.9,
                colsample_bytree=0.9, reg_lambda=1.0,
                tree_method="hist", seed=SEED, n_jobs=4)


def cb_params() -> dict:
    return dict(loss_function="Logloss", eval_metric="AUC", learning_rate=0.05,
                depth=6, l2_leaf_reg=1.0, random_seed=SEED,
                verbose=False, allow_writing_files=False, thread_count=8)


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """category -> 整数编码, 供 sklearn 树/线性模型使用。"""
    out = df.copy()
    for c in CAT_COLS:
        if hasattr(out[c], "cat"):
            out[c] = out[c].cat.codes.astype(float)
        else:
            out[c] = out[c].astype(float)
    return out


def main() -> None:
    t0 = time.time()
    train, test, sub = load_clean_data(DATA_DIR)
    X = build_features(train)
    y = train[TARGET].astype(int)
    X_test = build_features(test)
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    n_tr = len(X)
    log(f"[data] X {X.shape} | X_test {X_test.shape}\n")

    X_num = to_numeric(X)
    Xt_num = to_numeric(X_test)

    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y))

    # 复用 test 结构
    dtest_xgb = xgb.DMatrix(X_test)
    ptest_cb = Pool(X_test, cat_features=CAT_COLS)

    # base 的 OOF 和 test 预测
    base = {
        "lgb": {"oof": np.zeros(n_tr), "tp": np.zeros(len(X_test))},
        "xgb": {"oof": np.zeros(n_tr), "tp": np.zeros(len(X_test))},
        "cb": {"oof": np.zeros(n_tr), "tp": np.zeros(len(X_test))},
        "rf": {"oof": np.zeros(n_tr), "tp": np.zeros(len(X_test))},
        "lr": {"oof": np.zeros(n_tr), "tp": np.zeros(len(X_test))},
    }

    for fi, (tr, va) in enumerate(folds, start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        Xn_tr, Xn_va = X_num.iloc[tr], X_num.iloc[va]
        log(f"--- fold {fi}/{N_FOLDS}  (train {len(tr)}, val {len(va)}) ---")

        # LGB
        t = time.time()
        d_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS)
        d_va = lgb.Dataset(X_va, y_va, reference=d_tr)
        m = lgb.train(lgb_params(), d_tr, num_boost_round=3000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        base["lgb"]["oof"][va] = m.predict(X_va, num_iteration=m.best_iteration)
        base["lgb"]["tp"] += m.predict(X_test, num_iteration=m.best_iteration) / N_FOLDS
        del d_tr, d_va, m; gc.collect()
        log(f"    LGB {time.time()-t:5.0f}s")

        # XGB
        t = time.time()
        d_tr = xgb.DMatrix(X_tr, y_tr); d_va = xgb.DMatrix(X_va, y_va)
        m = xgb.train(xgb_params(), d_tr, num_boost_round=3000,
                      evals=[(d_va, "va")], early_stopping_rounds=100, verbose_eval=False)
        base["xgb"]["oof"][va] = m.predict(d_va, iteration_range=(0, m.best_iteration + 1))
        base["xgb"]["tp"] += m.predict(dtest_xgb, iteration_range=(0, m.best_iteration + 1)) / N_FOLDS
        del d_tr, d_va, m; gc.collect()
        log(f"    XGB {time.time()-t:5.0f}s")

        # CatBoost
        t = time.time()
        p_tr = Pool(X_tr, y_tr, cat_features=CAT_COLS); p_va = Pool(X_va, y_va, cat_features=CAT_COLS)
        m = CatBoostClassifier(**cb_params(), iterations=3000, early_stopping_rounds=100)
        m.fit(p_tr, eval_set=p_va, use_best_model=True)
        base["cb"]["oof"][va] = m.predict_proba(p_va)[:, 1]
        base["cb"]["tp"] += m.predict_proba(ptest_cb)[:, 1] / N_FOLDS
        del p_tr, p_va, m; gc.collect()
        log(f"    CB  {time.time()-t:5.0f}s")

        # RandomForest (bagging)
        t = time.time()
        m = RandomForestClassifier(n_estimators=150, max_features="sqrt",
                                   n_jobs=8, random_state=SEED, verbose=0)
        m.fit(Xn_tr, y_tr)
        base["rf"]["oof"][va] = m.predict_proba(Xn_va)[:, 1]
        base["rf"]["tp"] += m.predict_proba(Xt_num)[:, 1] / N_FOLDS
        del m; gc.collect()
        log(f"    RF  {time.time()-t:5.0f}s")

        # LogisticRegression (线性)
        t = time.time()
        sc = StandardScaler()
        Xs_tr = sc.fit_transform(Xn_tr)
        Xs_va = sc.transform(Xn_va)
        Xs_te = sc.transform(Xt_num)
        m = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", n_jobs=4)
        m.fit(Xs_tr, y_tr)
        base["lr"]["oof"][va] = m.predict_proba(Xs_va)[:, 1]
        base["lr"]["tp"] += m.predict_proba(Xs_te)[:, 1] / N_FOLDS
        del sc, m; gc.collect()
        log(f"    LR  {time.time()-t:5.0f}s")

        fa = {k: roc_auc_score(y.iloc[va], base[k]["oof"][va]) for k in base}
        log(f"    fold AUC  " + "  ".join(f"{k}={v:.5f}" for k, v in fa.items()))

    # ---- 各 base 整体 OOF ----
    log("\n[base OOF] ...")
    for k in base:
        log(f"    {k:3s} = {roc_auc_score(y, base[k]['oof']):.5f}")

    # ---- 第二层 meta (LR, 5 折 CV 评估) ----
    log("\n[meta] LogisticRegression on 5 base OOFs ...")
    names = list(base.keys())
    X_meta = np.column_stack([base[k]["oof"] for k in names])
    X_meta_test = np.column_stack([base[k]["tp"] for k in names])

    meta_oof = np.zeros(n_tr)
    meta_tp = np.zeros(len(X_test))
    for tr, va in folds:
        mm = LogisticRegression(C=1.0, max_iter=2000)
        mm.fit(X_meta[tr], y.iloc[tr])
        meta_oof[va] = mm.predict_proba(X_meta[va])[:, 1]
        meta_tp += mm.predict_proba(X_meta_test)[:, 1] / N_FOLDS
    meta_auc = roc_auc_score(y, meta_oof)
    log(f"    meta OOF AUC = {meta_auc:.5f}")
    log(f"    (对照: 5-base 等权 = {roc_auc_score(y, X_meta.mean(axis=1)):.5f})")

    # ---- 生成提交 ----
    sub["Will_Buy_EV"] = meta_tp
    out = DATA_DIR / OUT_NAME
    sub.to_csv(out, index=False)
    log(f"\n[save] {out}  pred [{meta_tp.min():.4f}, {meta_tp.max():.4f}]  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
