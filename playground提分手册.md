# Playground 数据提分手册：准类别检测 + 目标编码

> 来源：Kaggle Playground S6E9（EV Purchases）实战，LB 从 0.94198 → 0.94516（+0.00318）。
> 核心一句话：**Playground 合成数据的 numeric 列往往是「准类别」，用 fold 内目标编码 + KNN 编码吃干净它，比堆模型有效得多。**

---

## 0. 先做数据探查（最关键的一步，5 分钟）

决定后面要不要做目标编码，就看这一步。检查每个列的唯一值比例：

```python
import pandas as pd
train = pd.read_csv("train.csv")
for c in train.columns:
    if c in ("id", "Will_Buy_EV"):   # 跳过 id 和目标
        continue
    r = train[c].nunique() / len(train)
    flag = "  <<< 准类别!" if (r < 0.05 and train[c].nunique() > 50) else ""
    print(f"{c:32s} nunique={train[c].nunique():8d} ratio={r:.4f}{flag}")
```

**判断标准：**
- `ratio` 极低（< 0.001）→ 真类别，LightGBM 原生 category 处理即可，**不需要**额外编码
- `ratio` 在 0.001~0.05 且 nunique > 50 → **「准类别」列**，做目标编码会有大收益（S6E9 的 `Annual_Income_USD` 13214 唯一值 = ratio 0.02，目标编码直接 +0.002）
- `ratio` 接近 1 → 真连续，做 KNN 编码（相似值）可能有效

> ⚠️ 先看整体：如果**完整特征组合 100% 唯一**（每行都独一无二），说明是 VAE/diffusion 生成，没有「重复模板行」，别浪费时间做整行聚合。

---

## 1. 可复用的提分流程（5 步）

### Step 1 — baseline（单 LGB，10 分钟）
先跑一个干净的 LightGBM 5 折 OOF，锚定起点。参数用通用默认：`lr=0.05, num_leaves=31, min_child_samples=100`。

### Step 2 — 同值目标编码（主突破，10 分钟）
对「准类别」列做 **fold 内平滑目标编码**（严格防泄漏）：

```python
def te_fit(key_tr, y_tr, gm, smoothing):
    agg = y_tr.groupby(key_tr).agg(["mean", "count"])
    return (agg["mean"] * agg["count"] + gm * smoothing) / (agg["count"] + smoothing)

# fold 内：只用 train fold 的 y 算编码，map 到 train/val/test
enc = te_fit(X_tr[c], y_tr, gm, smoothing)
X_tr[f"TE_{c}"] = X_tr[c].map(enc).fillna(gm).astype(float)
X_va[f"TE_{c}"] = X_va[c].map(enc).fillna(gm).astype(float)
```

然后 **smoothing 细扫**（5/7/10/20/50），这个数据最优在 **s=7** 附近（太小会欠平滑、太大丢信号）。

### Step 3 — KNN 目标编码（二次增益，10 分钟）
「同值」只聚合完全相同的值，但 VAE 数据里「相近值 → 相近模板」。对关键列做 KNN 编码，捕捉连续性：

```python
from sklearn.neighbors import NearestNeighbors
def knn_enc(X_tr, X_q, y_np, k):
    nn = NearestNeighbors(n_neighbors=k).fit(X_tr)
    _, idx = nn.kneighbors(X_q)
    return y_np[idx].mean(axis=1)
```

k 细扫（50/100/150/200/300），S6E9 最优 **k=150**（Income 一维）。

### Step 4 — 三模型集成（10 分钟）
LGB + XGB + CatBoost 等权 blend。增益不大（+0.0002，因为三个 GBDT 相关性 0.997），但稳定、白拿。

### Step 5 — 收尾（可选）
多 seed bagging + 权重搜索。**注意：OOF→LB 会饱和**，多 seed 的 OOF 增益可能 LB 不兑现，别花提交配额在微小差异上。

---

## 2. 踩过的坑（别再犯，省你几小时）

| 方向 | 结果 | 原因 |
|------|------|------|
| 伪标签 | ❌ LB 下降 | 伪标签正例率偏差 + 过拟合 test 分布，OOF 虚高但 LB 掉 |
| Stacking（5 base + LR meta） | ❌ 无效 | RF/LR 单独太弱，meta 挤不出正贡献 |
| 组合列目标编码 | ❌ 0 增益 | 单列 TE 已吃干净信号 |
| 特征扩展（比率/三阶/分箱） | ❌ 0 或负 | 信号已被现有特征捕获 |
| 神经网络（MLP） | ❌ 0.9437 弱于 GBDT | 合成 tabular 数据 NN 打不过调好的 GBDT |
| 参数调优（leaves 15） | ⚠️ +0.00006 | 但 LB 已饱和，不兑现 |

---

## 3. 环境坑（Windows 上跑 GBDT 的必知）

- **三个 GBDT 库在增强数据上会随机卡死**（XGB 曾卡 6809s、CB 曾 1415s）。必须限线程：`LGB n_jobs=8 / XGB n_jobs=4 / CB thread_count=8`。
- **必须用 `python -u` 运行**，否则 print 被块缓冲吞掉，崩溃时看不到输出（S6E9 第一次 exit code 4 就这么死的）。
- 每个模型训练后 `del` + `gc.collect()`，避免 native 内存累积。

---

## 4. 本次 S6E9 的成绩单

| 版本 | OOF | LB |
|------|-----|-----|
| 单 LGB baseline | 0.94201 | 0.94198 |
| + 同值 TE (s=7) | 0.94441 | 0.94491 |
| + KNN (k=150) | 0.94483 | 0.94514 |
| + 多 seed 加权 | 0.94513 | **0.94516** |

第一名 0.94674，差 0.0016。靠 GBDT 体系已追不动，剩余差距需 NN/AutoGluon 级投入且大概率追不平。

---

## 5. 一句话总结

**拿到 playground 数据 → 先查 nunique 找「准类别」→ fold 内目标编码（s≈7）→ KNN 编码（k 细扫）→ 三模型 blend。** 这套流程在绝大多数 playground 合成数据上都能稳定拿 +0.002 以上的提升，且全程无泄漏、不过拟合。
