## モデル訓練の実行方法（Pythonスクリプト）

`notebook/Origin_chex_AIME.ipynb` の訓練部分のみを抽出し、CLIから実行できるスクリプト `train.py` を追加しました。

### 使い方

基本:

```bash
python /home/kosukeyano/workspace/huber_aime/train.py
```

主なオプション:

- `--dataset-base`: Kaggle NIH Chest X-rays の展開先ルート。未指定時は自動解決を試みます（`kagglehub` が使えない場合、既定パスにフォールバック）。
- `--use-final-csv`: 既に作成済みの前処理済みCSV（`final.csv` など）を指定すると、それをそのまま使用します。
- `--save-final-csv`: 前処理後のCSVを書き出すパス（例: `/home/kosukeyano/workspace/huber_aime/data/final.csv`）。
- `--epochs`: 学習エポック数（デフォルト: 20）。
- `--train-batch-size` / `--eval-batch-size`: バッチサイズ（デフォルト: 8 / 4）。
- `--device`: `auto` | `cpu` | `cuda` | `cuda:0` など（デフォルト: `auto`）。
- `--best-model-path`: ベストモデルの保存先（デフォルト: `/home/kosukeyano/workspace/huber_aime/notebook/models/best_model.pt`）。
- `--num-workers`: DataLoader ワーカ数（デフォルト: CPUコア数-1）。

### 実行例

1) KaggleHub による自動解決（ダウンロード/キャッシュ済み前提）で訓練、前処理CSVも保存:

```bash
python /home/kosukeyano/workspace/huber_aime/train.py \
  --save-final-csv /home/kosukeyano/workspace/huber_aime/data/final.csv
```

2) 手元のデータセットルートを明示して訓練:

```bash
python /home/kosukeyano/workspace/huber_aime/train.py \
  --dataset-base /home/kosukeyano/.cache/kagglehub/datasets/nih-chest-xrays/data/versions/3
```

3) すでに作ってある `final.csv` を使って訓練（前処理をスキップ）:

```bash
python /home/kosukeyano/workspace/huber_aime/train.py \
  --use-final-csv /home/kosukeyano/workspace/huber_aime/data/final.csv
```

4) CPUで小さくデバッグ（件数制限あり）:

```bash
python /home/kosukeyano/workspace/huber_aime/train.py \
  --device cpu \
  --epochs 1 \
  --limit-samples 2000
```

### 出力物

- モデル: `--best-model-path` で指定したパスに、検証AUROCが改善した都度ベストを上書き保存
- ログ: `train.log`
- エポック毎の要約表: `results.txt`

### 補足

- 乱数固定（`--seed`）と `DataLoader` の `generator` / `worker_init_fn` を設定し、分割と学習の再現性を担保しています。
- データ分割は 80%（train+val）:20%（test）、さらに train を 80:20 で train:val に分割します（ノートブックと同等）。

## huber-aime — AIME (Approximate Inverse Model Explanations)

### 概要
本リポジトリは AIME（Approximate Inverse Model Explanations）のデモ/ユーティリティ実装とノートブックを含みます。近似逆作用素を用いて、モデル出力から入力特徴量方向へ説明量を逆伝播させ、グローバル/ローカルの特徴重要度を算出します。CPU/Numpy 実装に加え、PyTorch、CuPy、cupynumeric（Legate）など複数バックエンドに対応しています。

- **主な実装**: `aime_xai/core_gpu.py` の `AIMEEx`
- **ノートブック**: `notebook/compare.ipynb`, `notebook/Origin_chex_AIME.ipynb`
- **関連資料**: `PDF/Approximate_Inverse_Model_Explanations_AIME_...pdf`

### 対応環境
- **Python**: >= 3.10, < 3.13
- **OS**: Linux 推奨（GPU/CUDA を使う場合は必須）

### インストール
依存関係は `pyproject.toml` で管理しています。以下のいずれかでインストールできます。

- uv（推奨）
```bash
uv sync
# または 開発インストール
uv pip install -e .
```

- pip
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

#### GPU オプション（任意）
- CuPy / RAPIDS など GPU 周辺は Linux + CUDA 環境向けです。
- 付属の extras は環境差が大きく、事前に CUDA ドライバ/ランタイムの整備が必要です。

例（Linux + CUDA 12.x 想定）:
```bash
# CuPy（CUDA 12.x）
pip install cupy-cuda12x

# PyTorch（CUDA 12.1 の例）
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# 必要に応じて（環境依存）RAPIDS 系を extras で試すことも可能
# pip install ".[rapids]"
```

### クイックスタート（分類 × 重要度）
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from aime_xai.core_gpu import AIMEEx

# データ用意
X, y = load_iris(return_X_y=True)
feature_names = [
    "sepal_length", "sepal_width", "petal_length", "petal_width"
]
class_names = ["setosa", "versicolor", "virginica"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 学習（任意のモデルで OK。ここでは分類の確率を Y として使用）
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
Y = clf.predict_proba(X_test)  # (n_samples, n_classes)

# 近似逆作用素の作成（CPU/Numpy 例）
ex = AIMEEx(method="numpy_solve", epsilon=1e-6)
ex.create_explainer(X_test, Y, normalize=True)

# グローバル重要度（可視化なし DataFrame）
df_global = ex.global_feature_importance_without_viz(
    feature_names=feature_names,
    class_names=class_names,
    top_k=4,
)
print(df_global)

# ローカル重要度（1 サンプル、可視化なし）
df_local = ex.local_feature_importance_without_viz(
    x=X_test[0],
    y=Y[0],
    feature_names=feature_names,
    top_k=4,
)
print(df_local)

# GPU（任意）：CUDA が使える場合の例
# import torch
# if torch.cuda.is_available():
#     ex_gpu = AIMEEx(method="torch_solve", device="cuda", epsilon=1e-6)
#     ex_gpu.create_explainer(X_test, Y, normalize=True)
#     print("CUDA time (s):", ex_gpu.computation_time)
```

### 主要 API（抜粋）
- **`AIMEEx(method, epsilon=1e-6, device="cuda", batch_size=100)`**: 近似逆作用素の計算器
  - **method**: `"numpy_solve" | "numpy_pinv" | "torch_solve" | "torch_pinv" | "torch_solve_batch" | "torch_pinv_batch" | "cupy_solve" | "cupy_pinv" | "cupynumeric_solve"`
  - **device**: `"cuda" | "cpu"`（PyTorch 利用時）
  - **computation_time**: 直近の計算に要した秒数（バックエンドにより計測方法が異なります）
- **`create_explainer(X, Y, normalize=True)`**: 近似逆作用素を学習し内部に保持
- **`global_feature_importance(...)` / `global_feature_importance_without_viz(...)`**: 出力次元（クラス）ごとのグローバル重要度
- **`global_feature_importance_each(..., class_num=0)`**: 指定クラスのみのグローバル重要度
- **`local_feature_importance(...)` / `local_feature_importance_without_viz(...)`**: 単一サンプルのローカル重要度
- **`plot_rep_instance_similarity(...)`**: 代表ベクトルとサンプルの類似度プロット（RBF + 次元削減オプション）

### ノートブック
- `notebook/compare.ipynb`
- `notebook/Origin_chex_AIME.ipynb`

### よくある注意点
- `Y` は形状 `(n_samples, n_outputs)`（例: 分類なら予測確率）を想定しています。
- `normalize=True` の場合、内部で `StandardScaler` により `X` を正規化します。
- GPU 実行時は、データ転送と同期タイミングを除いた計算時間を計測しています（実装依存）。

### 引用・参考
理論や詳細は付属 PDF（`PDF/Approximate_Inverse_Model_Explanations_AIME_...pdf`）およびノートブックをご参照ください。

### ライセンス
未定（必要に応じて追記してください）。



