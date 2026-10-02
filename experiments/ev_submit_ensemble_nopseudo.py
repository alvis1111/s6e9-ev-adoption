"""
EV Purchases — Playground Series S6E9
三模型集成 (无伪标签) 提交生成器
================================================
复现已验证的 0.94218 (LGB+XGB+CB 等权平均, 66 万行原始 train, 无伪标签):
  - LGB   0.94201
  - XGB   0.94184
  - CB    0.94201
  - 等权平均 0.94218  |  加权(0.4/0.2/0.4) 0.94220

关键: 无伪标签 -> 不引入 test 信息, 不过拟合 test 分布 (伪标签已证伪)。
限线程防卡死 (LGB 8 / XGB 4 / CB 8), 实时输出, 每折释放内存。

输出:
  submission_ensemble_nopseudo.csv     (等权平均, 主推)
  submission_ensemble_nopseudo_w.csv   (加权 0.4/0.2/0.4, 备选)

运行: python -u ev_submit_ensemble_nopseudo.py
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

from ev_clean import load_clean_data, TARGET
from ev_features import build_features, LGB_CATEGORY_COLS

DATA_DIR = Path(__file__).resolve().parent
SEED = 42
N_FOLDS = 5
CAT_COLS = LGB_CATEGORY_COLS + ["Income_bin"]


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


def main() -> None:
    t0 = time.time()
    train, test, sub = load_clean_data(DATA_DIR)
    X = build_features(train)
    y = train[TARGET].astype(int)
    X_test = build_features(test)
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    log(f"[data] X {X.shape} | X_test {X_test.shape}\n")

    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    oof = {"lgb": np.zeros(len(X)), "xgb": np.zeros(len(X)), "cb": np.zeros(len(X))}
    tp = {"lgb": np.zeros(len(X_test)), "xgb": np.zeros(len(X_test)), "cb": np.zeros(len(X_test))}

    dtest_xgb = xgb.DMatrix(X_test)
    ptest_cb = Pool(X_test, cat_features=CAT_COLS)

    for fi, (tr, va) in enumerate(folds, start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        log(f"--- fold {fi}/{N_FOLDS}  (train {len(tr)}, val {len(va)}) ---")

        # LGB
        t = time.time()
        d_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS)
        d_va = lgb.Dataset(X_va, y_va, reference=d_tr)
        m = lgb.train(lgb_params(), d_tr, num_boost_round=3000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        bi = m.best_iteration
        oof["lgb"][va] = m.predict(X_va, num_iteration=bi)
        tp["lgb"] += m.predict(X_test, num_iteration=bi) / N_FOLDS
        del d_tr, d_va, m; gc.collect()
        log(f"    LGB {time.time()-t:5.0f}s  best_iter={bi}")

        # XGB
        t = time.time()
        d_tr = xgb.DMatrix(X_tr, y_tr)
        d_va = xgb.DMatrix(X_va, y_va)
        m = xgb.train(xgb_params(), d_tr, num_boost_round=3000,
                      evals=[(d_va, "va")], early_stopping_rounds=100, verbose_eval=False)
        bi = m.best_iteration
        oof["xgb"][va] = m.predict(d_va, iteration_range=(0, bi + 1))
        tp["xgb"] += m.predict(dtest_xgb, iteration_range=(0, bi + 1)) / N_FOLDS
        del d_tr, d_va, m; gc.collect()
        log(f"    XGB {time.time()-t:5.0f}s  best_iter={bi}")

        # CatBoost
        t = time.time()
        p_tr = Pool(X_tr, y_tr, cat_features=CAT_COLS)
        p_va = Pool(X_va, y_va, cat_features=CAT_COLS)
        m = CatBoostClassifier(**cb_params(), iterations=3000, early_stopping_rounds=100)
        m.fit(p_tr, eval_set=p_va, use_best_model=True)
        bi = m.get_best_iteration()
        oof["cb"][va] = m.predict_proba(p_va)[:, 1]
        tp["cb"] += m.predict_proba(ptest_cb)[:, 1] / N_FOLDS
        del p_tr, p_va, m; gc.collect()
        log(f"    CB  {time.time()-t:5.0f}s  best_iter={bi}")

        fa = {k: roc_auc_score(y.iloc[va], oof[k][va]) for k in oof}
        fb = (oof["lgb"][va] + oof["xgb"][va] + oof["cb"][va]) / 3
        log(f"    fold AUC  lgb={fa['lgb']:.5f} xgb={fa['xgb']:.5f} cb={fa['cb']:.5f} "
            f"blend={roc_auc_score(y.iloc[va], fb):.5f}")

    # ---- 汇总 ----
    log("\n[OOF] individual & blend ...")
    for k in oof:
        log(f"    {k:3s} = {roc_auc_score(y, oof[k]):.5f}")
    blend_eq = (oof["lgb"] + oof["xgb"] + oof["cb"]) / 3
    log(f"    blend equal = {roc_auc_score(y, blend_eq):.5f}")
    blend_w = 0.4 * oof["lgb"] + 0.2 * oof["xgb"] + 0.4 * oof["cb"]
    log(f"    blend w(0.4/0.2/0.4) = {roc_auc_score(y, blend_w):.5f}")

    # ---- 生成提交 ----
    tp_eq = (tp["lgb"] + tp["xgb"] + tp["cb"]) / 3
    sub["Will_Buy_EV"] = tp_eq
    out = DATA_DIR / "submission_ensemble_nopseudo.csv"
    sub.to_csv(out, index=False)
    log(f"\n[save] {out}  pred [{tp_eq.min():.4f}, {tp_eq.max():.4f}]")

    tp_w = 0.4 * tp["lgb"] + 0.2 * tp["xgb"] + 0.4 * tp["cb"]
    sub["Will_Buy_EV"] = tp_w
    out2 = DATA_DIR / "submission_ensemble_nopseudo_w.csv"
    sub.to_csv(out2, index=False)
    log(f"[save] {out2}  pred [{tp_w.min():.4f}, {tp_w.max():.4f}]")

    log(f"[total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
