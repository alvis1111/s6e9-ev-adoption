"""S6E9 —— 严谨流程套 Naji 配方：10% 审计留出（只用 y 分层）+ 5 折 OOF + 稳健 seed。

审计集放在 test 侧（和真实测试集同样待遇：标签不参与训练/TE/选特征，只参与
无标签统计），只在最后测一次，作为「本地 OOF 会不会在未见数据上翻车」的诚实估计。

用法：
    python rigorous.py               # 单 seed 42
    python rigorous.py --seeds 42,137,2026   # 稳健性三 seed
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from naji_features import build_features, make_lgb, te_transform

BASE = Path(__file__).parent
TARGET = "Will_Buy_EV"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--audit", type=float, default=0.10)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    tr = pd.read_csv(BASE / "train.csv")
    y = (tr[TARGET] == "Yes").astype(int).values
    # 只用目标标签分层切 10% 审计
    dev_idx, audit_idx = train_test_split(np.arange(len(tr)), test_size=args.audit, stratify=y, random_state=42)
    dev = tr.iloc[dev_idx].reset_index(drop=True)
    audit = tr.iloc[audit_idx].reset_index(drop=True)
    audit_y = (audit[TARGET] == "Yes").astype(int).values
    test_csv = pd.read_csv(BASE / "test.csv")
    test_df = pd.concat([audit, test_csv], ignore_index=True)  # audit 放最前
    n_audit = len(audit)

    X, y_dev, X_test, ids, TE_COLS = build_features(train_df=dev, test_df=test_df)
    n_test = len(X_test) - n_audit
    print(f"dev {len(X)} / audit {n_audit} / test {n_test}，特征 {X.shape[1]}", flush=True)

    oof_all = np.zeros((len(seeds), len(X)))
    audit_all = np.zeros((len(seeds), n_audit))
    test_all = np.zeros((len(seeds), n_test))
    t0 = time.time()

    for si, seed in enumerate(seeds):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = np.zeros(len(X))
        te_pred = np.zeros(len(X_test))
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_dev), 1):
            X_tr, y_tr = X.iloc[tr_idx].copy(), y_dev.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx].copy(), y_dev.iloc[va_idx]
            X_te = X_test.copy()
            X_tr, X_va, X_te = te_transform(X_tr, y_tr, X_va, X_te, TE_COLS)
            clf = make_lgb(seed)
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(500, verbose=False)])
            oof[va_idx] = clf.predict_proba(X_va)[:, 1]
            te_pred += clf.predict_proba(X_te)[:, 1] / 5
        oof_all[si] = oof
        audit_all[si] = te_pred[:n_audit]
        test_all[si] = te_pred[n_audit:]
        print(f"  seed {seed}: dev OOF {roc_auc_score(y_dev, oof):.5f} | audit AUC {roc_auc_score(audit_y, te_pred[:n_audit]):.5f}  [{time.time()-t0:.0f}s]", flush=True)

    oof_m = oof_all.mean(axis=0)
    audit_m = audit_all.mean(axis=0)
    test_m = test_all.mean(axis=0)
    print(f"\n=== {len(seeds)} seed 平均 ===")
    print(f"dev OOF AUC (90% dev, 5折): {roc_auc_score(y_dev, oof_m):.5f}")
    print(f"audit AUC (10% 留出，标签从未参与): {roc_auc_score(audit_y, audit_m):.5f}")

    out = Path("results_rigorous")
    out.mkdir(exist_ok=True)
    np.save(out / "oof_dev.npy", oof_m)
    np.save(out / "audit_pred.npy", audit_m)
    test_ids = ids[n_audit:].reset_index(drop=True)
    pd.DataFrame({"id": test_ids, TARGET: test_m}).to_csv(out / "submission.csv", index=False)
    print(f"已保存到 {out}/", flush=True)


if __name__ == "__main__":
    main()
