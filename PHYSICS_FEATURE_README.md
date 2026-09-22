# PhysicsFeatureExtractor：非侵入式物理預估特徵外掛

這個升級刻意不修改既有 256D / V23 Core，也不修改目前 residual_bias_runtime.js 的 7D 生產流程。

現有主線仍是：

~~~
牌路歷史 -> 256D / V23 Core -> Core P(B) -> 固定 7D
-> XGBoost Residual -> Delta clip ±10% -> Final P(B)
~~~

新增 Python 實驗管線：

~~~
                         +------------------------------+
B/P/T history ----------> PhysicsFeatureExtractor 48D  |
                         +------------------------------+
                                      |
Core P(B) + 原固定 7D ----------------+--> 56D --> Physics XGB Residual
~~~

## 重要物理限制

只輸入 B/P/T 路徑時，模型無法知道真實未翻開的 A-K、花色或實際剩餘牌組成。
因此這個模組輸出的是離線 8 副牌模擬下，對下一局與當前 shoe state 的條件期望 / 概率分佈。

花色不參與標準百家樂勝負與補牌規則，所以只靠 B/P/T 對花色的可辨識資訊極低；
模型輸出的花色通常應接近 25% / 25% / 25% / 25% 的條件期望。
它不是實體驗牌器，也不聲稱能重建 unseen cards。

## 48D Physics Features

固定輸出 48 維：

- 3D：下一局 4 / 5 / 6 張牌概率。
- 10D：閒家最終 0-9 點概率。
- 10D：莊家最終 0-9 點概率。
- 3D：莊 / 閒 / 和概率。
- 13D：下一局 A-K 各 rank 預期消耗張數。
- 4D：下一局四花色預期消耗比例。
- 3D：已消耗牌數比例、剩餘低 rank 密度、剩餘高 rank 密度。
- 2D：莊減閒點數差期望、絕對點數差期望。

總計 48D。

### 大 / 小牌密度定義

此模組使用明確的 physical-rank 定義：

- low-rank：A-5
- high-rank：9-K
- 6-8 不放入上述兩個 density

## 離線 8 副牌模擬

physics_feature_extractor.py 使用：

- 8 副 × 52 張 = 416 張實體牌。
- 標準 Player / Banker 第三張牌規則。
- 可設定 cut cards。
- 每個 supervision row 只把當下已出現的 B/P/T history 當輸入。
- simulated card state 只作為 training target，不會在線上 inference 偷看。

產生資料：

~~~
python physics_feature_extractor.py simulate \
  --shoes 1000 \
  --cut-cards 60 \
  --output physics_simulation_dataset.npz
~~~

直接訓練 multi-output XGBoost：

~~~
python physics_feature_extractor.py train \
  --shoes 2000 \
  --cut-cards 60 \
  --output physics_multitask_model.ubj
~~~

模型旁邊會建立 physics_multitask_model.ubj.meta.json，
記錄 schema、feature names 與 validation metrics。

## prepare_xgboost_input

線上 Python 對接：

~~~
import numpy as np
from physics_feature_extractor import (
    PhysicsFeatureExtractor,
    prepare_xgboost_input,
)

physics = PhysicsFeatureExtractor.load("physics_multitask_model.ubj")

core_pb = 0.534
original_7d = np.asarray([
    0.534,
    18.0,
    60.0,
    0.7167,
    0.52,
    2.0,
    1.0,
], dtype=np.float32)

x = prepare_xgboost_input(
    core_pb,
    original_7d,
    "BPPBTBBPPTB",
    extractor=physics,
)

assert x.shape == (56,)
~~~

實作只做：

~~~
np.hstack([
    [core_pb],          # 1D
    original_7d,        # 原值、原順序，不修改
    physics_48d,        # 新外掛
])
~~~

注意：目前 repository 的舊 7D schema 本身含有 core_p_b。
為了完全遵守「原 7D 不動」，56D 會同時保留最前面的 external Core P(B)
與舊 7D 裡原本的 core_p_b，不偷偷去重。

## Physics XGBoost Residual

新的實驗 trainer 是 xgb_residual_physics.py。
它沒有取代原本的 xgb_residual_bias.py。

先收集原 browser runtime 的標籤資料：

~~~
__BGS_RESIDUAL_BIAS__.downloadTrainingData()
~~~

再訓練：

~~~
python xgb_residual_physics.py train \
  --input bgs_xgb_residual_training.json \
  --physics-model physics_multitask_model.ubj \
  --output residual_bias_physics.joblib \
  --min-samples 500
~~~

Residual 定義維持：

~~~
Residual = actual_B - Core P(B)
~~~

Inference 邏輯維持：

~~~
raw_delta = residual_model.predict(x56)
delta = np.clip(raw_delta, -0.10, +0.10)
final_pb = np.clip(core_pb + delta, 0.0, 1.0)
direction = "B" if final_pb > 0.50 else "P"
~~~

仍然沒有 PASS。

## 為什麼沒有直接改 residual_bias_runtime.js

目前網站是純 GitHub Pages + JavaScript；Python/XGBoost multi-output 模型不能直接在 browser 裡執行。

為避免「看起來已上線、其實 browser 沒有跑 Physics」的假整合，本分支先完成：

1. 完整 Python simulator。
2. 完整 multi-task Physics predictor。
3. 完整 56D feature concatenation。
4. 完整 Physics residual trainer / predictor。
5. 保留原 browser 7D runtime 不動。

若要正式在 GitHub Pages 啟用，下一階段需二選一：

- 把 Physics multi-output XGBoost 轉成 browser 可讀 tree bundle，再新增純 JS inference。
- 或把 56D residual inference 移到 Python API / server endpoint。

在那之前，main 的網站行為不會被這個實驗分支暗中改變。
