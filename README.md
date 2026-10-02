# Kaggle Playground S6E9 — Predicting Electric Vehicle Interest

预测客户是否购买电动汽车（二分类，ROC-AUC 评估）。

**最终成绩：Public LB 0.94643**（训练 OOF 0.94645），公开榜第一名 ~0.94674。

## 成绩演进（三个阶段）

| 阶段 | 关键手段 | OOF | LB |
|------|---------|-----|-----|
| ① 旧方案 | fold 内目标编码(s=7) + KNN 目标编码(k=150) + 三模型加权 | 0.94513 | 0.94516 |
| ② Naji 配方复现 | 整数位分解 + 收入魔法特征 + 三套目标编码 + 多 seed | 0.94626 | ≈0.94604 |
| ③ 抄公开 OOF（drafting） | 9 个公开模型贪心 rank 融合 | 0.94645 | **0.94643** |

## 核心方法论

Playground 合成数据的 numeric 列往往是「准类别」（大量重复值，如 `Annual_Income_USD` 13214 个唯一值各重复 ~50 次），这是 VAE/diffusion 生成器「模板行」的痕迹。

- **阶段①**：fold 内平滑目标编码（smoothing 细扫最优 s=7）+ KNN 目标编码（k=150，捕捉「相近 Income→相近模板」的连续性），直接 +0.003。
- **阶段②**：复现头部公开配方（`strong_lgbm`/Naji）——数字位特征、收入魔法特征（30k 尖峰/38-42k 死区/170537+ 悬崖）、原始数据集逐列目标均值、频率编码、平滑分箱、三套目标编码，多 seed 平均。
- **阶段③**：下载公开 OOF 库（Naji s6e9-oof / six-views / residual-stack，共 21 模型），训练集上贪心 rank 融合。

完整方法论与踩坑记录见 [`playground提分手册.md`](playground提分手册.md)。

## 仓库结构

```
├── src/                      # 核心代码（最终方案）
│   ├── naji_features.py       # 特征工程（Naji 配方，核心）
│   ├── train_final.py         # 生成最终提交
│   ├── rigorous.py            # 严谨验证框架（10% 审计留出 + 5 折）
│   ├── longtail_rigorous.py   # 长尾/收入魔法特征验证
│   ├── blend_oof.py           # 公开 OOF 贪心 rank 融合（drafting）
│   └── reconstruct_blend.py   # 反解 Nina v4 融合配方
├── experiments/              # 全部实验脚本（23 个 ev_* 系列）
├── notebooks/
│   ├── jazivxt_zoom.ipynb     # 头部选手 jazivxt 完整源码
│   └── s6e9-does-breaking-ties-help.ipynb  # Nina v4 配方来源验证
├── data/                     # 数据（需自行下载，见 data/README.md）
└── playground提分手册.md      # 方法论与踩坑记录
```

## 复现

```bash
pip install -r requirements.txt
# 从 Kaggle 下载 train.csv / test.csv / sample_submission.csv（及 EV_orig.csv）放到 data/
python -u src/train_final.py        # 阶段② 最终提交（OOF ≈ 0.94626）
python -u src/blend_oof.py          # 阶段③ drafting 融合（需公开 OOF 库）
```

## 核心结论（已充分验证）

**0.946 是这道题「公开特征 + 常规建模」的天花板。** 头部所有人用同一套 Naji 整数除法特征（互相 Spearman 0.999），0.94657 只是「54% 抄社区 top 提交 + 46% 同款配方」。

已验证**无效**：树堆叠、神经网络（与 LGBM 相关 0.98）、KNN 目标编码（Naji 细粒度 TE 已覆盖）、聚类、交互目标编码、扩展魔法特征、超参扫描、伪标签、生成器指纹映射（`original_label` AUC 0.513≈随机）。唯一确定收益是多 seed 平均（+0.0002）。

要破 0.94674 只剩一条路：**Naji 整数除法家族之外的真·新特征**（逆向 VAE 生成器的 latent 结构）——这是开放式研究，翻代码/堆模型/指纹映射都走不通。

## 环境注意（Windows）

- LGB/XGB/CB 在增强数据（~95 万行）上会随机卡死，必须限线程（LGB `n_jobs=8` / XGB `n_jobs=4` / CB `thread_count=8`）。
- 必须用 `python -u` 运行，保证实时输出。
