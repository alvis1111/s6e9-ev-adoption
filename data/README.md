# 数据说明

本目录用于存放数据，**数据文件不提交到 Git**（见 `.gitignore`）。

## 需要的文件

从 Kaggle 竞赛页下载：

1. **train.csv / test.csv / sample_submission.csv** — 竞赛数据
   - 下载：`kaggle competitions download -c playground-series-s6e9`
2. **EV_orig.csv** — 原始 EV 数据集（`naji_features.py` 计算目标均值用）
   - 来源：Kaggle Dataset「Predicting Electric Vehicle Interest」的原始数据

下载后放到本目录（`data/`）下即可，脚本按此路径读取。
