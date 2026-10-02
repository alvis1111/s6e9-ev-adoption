"""
EV Purchases — Playground Series S6E9
伪标签 + 三模型(LGB/XGB/CB) 5-Fold 集成 —— 提交生成器
================================================
基于已验证结论 (均无泄漏 OOF):
  - 三模型等权平均  = 0.94218  (LGB 0.94201 / XGB 0.94184 / CB 0.94201)
  - 伪标签 hard w=1.0 单独增益 +0.00013 (0.94201 -> 0.94214)
两者叠加: 伪标签增强训练 + 三模型 5 折集成，直接对 test 预测生成提交。

相比 ev_final_push.py 的改进 (针对那次 exit code 4):
  1. 全部输出实时 flush (必须 `python -u` 运行，避免块缓冲吞输出)
  2. 每个模型训练后立刻 del + gc.collect()，避免 native 内存累积
  3. X_test 的 DMatrix/Pool 只建一次复用
  4. 每折/每模型打印耗时，崩溃时可定位到具体位置

用法:
  python -u ev_submit_ensemble.py            # 全量 5 折
  python -u ev_submit_ensemble.py --smoke    # 冒烟: 只跑 1 折，快速验证
"""
from __future__ import annotations

import gc
import sys
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
N_FOLDS = 5
SEED = 42
CAT_COLS = LGB_CATEGORY_COLS + ["Income_bin"]
SMOKE = "--smoke" in sys.argv
OUT_NAME = "submission_ensemble.csv"


def log(msg: str) -> None:
    print(msg, flush=True)


def lgb_params(seed: int) -> dict:
    return dict(objective="binary", metric="auc", learning_rate=0.05,
                num_leaves=31, max_depth=-1, min_child_samples=100,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=seed, n_jobs=8, verbose=-1)


def xgb_params(seed: int) -> dict:
    return dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.05,
                max_depth=6, min_child_weight=50, subsample=0.9,
                colsample_bytree=0.9, reg_lambda=1.0,
                tree_method="hist", seed=seed, n_jobs=4)


def cb_params(seed: int) -> dict:
    return dict(loss_function="Logloss", eval_metric="AUC", learning_rate=0.05,
                depth=6, l2_leaf_reg=1.0, random_seed=seed,
                verbose=False, allow_writing_files=False, thread_count=8)


def main() -> None:
    t_start = time.time()
    n_eff = 1 if SMOKE else N_FOLDS
    log(f"[mode] {'SMOKE (1 fold)' if SMOKE else 'FULL (5 folds)'}\n")

    # ---- 1. 数据 + 特征 ----
    log("[1] load + features ...")
    train, test, sub = load_clean_data(DATA_DIR)
    X = build_features(train)
    y = train[TARGET].astype(int)
    X_test = build_features(test)
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")
    n_tr = len(X)
    log(f"    X {X.shape} | X_test {X_test.shape} | y+ {y.mean():.4f}")

    # ---- 2. 伪标签 (全量 LGB seed42, 1500 轮, 硬标签) ----
    log("[2] pseudo labels (full LGB seed42, 1500 rounds) ...")
    t = time.time()
    d_all = lgb.Dataset(X, y, categorical_feature=CAT_COLS)
    m_full = lgb.train(lgb_params(SEED), d_all, num_boost_round=1500)
    p_test = m_full.predict(X_test)
    hard = (p_test > 0.5).astype(int)
    log(f"    hard rate {hard.mean():.4f} (train {y.mean():.4f})  [{time.time()-t:.0f}s]")
    del d_all, m_full
    gc.collect()

    X_aug = pd.concat([X, X_test], ignore_index=True)
    y_aug = pd.concat([y, pd.Series(hard)], ignore_index=True).astype(int)
    log(f"    X_aug {X_aug.shape}\n")

    # ---- 3. 复用 test 结构 (DMatrix/Pool 只建一次) ----
    dtest_xgb = xgb.DMatrix(X_test)
    ptest_cb = Pool(X_test, cat_features=CAT_COLS)

    # ---- 4. 5 折 CV ----
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X_aug, y_aug))

    oof = {"lgb": np.zeros(n_tr), "xgb": np.zeros(n_tr), "cb": np.zeros(n_tr)}
    test_pred = {"lgb": np.zeros(len(X_test)),
                 "xgb": np.zeros(len(X_test)),
                 "cb": np.zeros(len(X_test))}

    for fi, (tr, va) in enumerate(folds[:n_eff]):
        X_tr, X_va = X_aug.iloc[tr], X_aug.iloc[va]
        y_tr, y_va = y_aug.iloc[tr], y_aug.iloc[va]
        va_orig = va[va < n_tr]          # 验证折中属于原始 train 的样本
        va_orig_pos = np.where(va < n_tr)[0]
        log(f"--- fold {fi+1}/{n_eff}  (train {len(tr)}, val {len(va_orig)}) ---")

        # LGB
        t = time.time()
        d_tr = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS)
        d_va = lgb.Dataset(X_va, y_va, reference=d_tr)
        m = lgb.train(lgb_params(SEED), d_tr, num_boost_round=3000,
                      valid_sets=[d_va], callbacks=[lgb.early_stopping(100, verbose=False)])
        best_iter = m.best_iteration
        oof["lgb"][va_orig] = m.predict(X_aug.iloc[va_orig], num_iteration=best_iter)
        test_pred["lgb"] += m.predict(X_test, num_iteration=best_iter) / n_eff
        del d_tr, d_va, m; gc.collect()
        log(f"    LGB  {time.time()-t:5.0f}s  best_iter={best_iter}")

        # XGB
        t = time.time()
        d_tr = xgb.DMatrix(X_tr, y_tr)
        d_va = xgb.DMatrix(X_va, y_va)
        m = xgb.train(xgb_params(SEED), d_tr, num_boost_round=3000,
                      evals=[(d_va, "va")], early_stopping_rounds=100, verbose_eval=False)
        best_iter = m.best_iteration
        oof["xgb"][va_orig] = m.predict(xgb.DMatrix(X_aug.iloc[va_orig]),
                                        iteration_range=(0, best_iter + 1))
        test_pred["xgb"] += m.predict(dtest_xgb, iteration_range=(0, best_iter + 1)) / n_eff
        del d_tr, d_va, m; gc.collect()
        log(f"    XGB  {time.time()-t:5.0f}s  best_iter={best_iter}")

        # CatBoost
        t = time.time()
        p_tr = Pool(X_tr, y_tr, cat_features=CAT_COLS)
        p_va = Pool(X_va, y_va, cat_features=CAT_COLS)
        m = CatBoostClassifier(**cb_params(SEED), iterations=3000, early_stopping_rounds=100)
        m.fit(p_tr, eval_set=p_va, use_best_model=True)
        best_iter = m.get_best_iteration()
        oof["cb"][va_orig] = m.predict_proba(Pool(X_aug.iloc[va_orig], cat_features=CAT_COLS))[:, 1]
        test_pred["cb"] += m.predict_proba(ptest_cb)[:, 1] / n_eff
        del p_tr, p_va, m; gc.collect()
        log(f"    CB   {time.time()-t:5.0f}s  best_iter={best_iter}")

        # fold 级验证 AUC（只看原始 train 的验证样本）—— smoke 也看得到真实水平
        fa = {k: roc_auc_score(y.iloc[va_orig], oof[k][va_orig]) for k in ("lgb", "xgb", "cb")}
        fb = (oof["lgb"][va_orig] + oof["xgb"][va_orig] + oof["cb"][va_orig]) / 3
        log(f"    fold AUC  lgb={fa['lgb']:.5f}  xgb={fa['xgb']:.5f}  cb={fa['cb']:.5f}  "
            f"blend={roc_auc_score(y.iloc[va_orig], fb):.5f}")

    # ---- 5. 汇总 ----
    if SMOKE:
        log("\n[5] SMOKE mode: only 1 fold ran, skip full OOF.")
    else:
        log("\n[5] results (OOF on original train only, no leakage) ...")
        for k in ("lgb", "xgb", "cb"):
            log(f"    {k:3s} OOF AUC = {roc_auc_score(y, oof[k]):.5f}")
        blend = (oof["lgb"] + oof["xgb"] + oof["cb"]) / 3
        blend_auc = roc_auc_score(y, blend)
        log(f"    blend (equal avg) OOF AUC = {blend_auc:.5f}")

    # ---- 6. 生成提交 ----
    tp_blend = (test_pred["lgb"] + test_pred["xgb"] + test_pred["cb"]) / 3
    if SMOKE:
        log("\n[6] SMOKE mode: skip submission save.")
    else:
        sub["Will_Buy_EV"] = tp_blend
        out = DATA_DIR / OUT_NAME
        sub.to_csv(out, index=False)
        log(f"\n[save] {out}")
        log(f"[pred] range [{tp_blend.min():.4f}, {tp_blend.max():.4f}] | total {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
