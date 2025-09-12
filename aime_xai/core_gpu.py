import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# PyTorchとCuPyが利用可能かチェック
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import cupynumeric as cpn
    from legate.timing import time as legate_time

    CUPYNUMERIC_AVAILABLE = True
except ImportError:
    CUPYNUMERIC_AVAILABLE = False

from sklearn.preprocessing import StandardScaler


class AIMEEx:
    """
    AIMEクラスの拡張版。
    pinv_solve_example.pyで実装された計算手法を統合し、
    近似逆作用素の計算方法を選択可能にする。
    """

    def __init__(
        self, method="numpy_solve", epsilon=1e-6, device="cuda", batch_size=100
    ):
        """
        Parameters
        ----------
        method : str
            計算に使用する手法。以下から選択可能:
            'numpy_solve', 'numpy_pinv',
            'torch_solve', 'torch_pinv', 'torch_solve_batch', 'torch_pinv_batch',
            'cupy_solve', 'cupy_pinv',
            'cupynumeric_solve'
        epsilon : float
            ティホノフ正則化のための微小な値。逆行列計算を安定させる。
        device : str
            PyTorchを使用する場合のデバイス ('cuda' or 'cpu')。
        batch_size : int
            PyTorchバッチ版で使用するバッチサイズ。
        """
        self.method = method
        self.epsilon = epsilon
        self.device = device
        self.batch_size = batch_size
        self.A_dagger = None
        self.scaler = None
        self.computation_time = None

        # ライブラリの利用可能性をチェック
        if "torch" in self.method and not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorchがインストールされていません。'torch_solve'または'torch_pinv'は使用できません。"
            )
        if "cupy" in self.method and not CUPY_AVAILABLE:
            raise ImportError(
                "CuPyがインストールされていません。'cupy_solve'または'cupy_pinv'は使用できません。"
            )
        if (
            device == "cuda"
            and "torch" in self.method
            and not torch.cuda.is_available()
        ):
            print("警告: CUDAが利用できません。PyTorchはCPUで実行されます。")
            self.device = "cpu"

    def create_explainer(self, X, Y, normalize=True):
        """
        入力Xと出力Yから近似逆作用素を計算して説明器を作成する。
        """
        if X is None or Y is None:
            raise ValueError("XとYの両方を提供してください。")

        # データの正規化
        X_np = np.asarray(X, dtype=float)
        Y_np = np.asarray(Y, dtype=float)
        if normalize:
            self.scaler = StandardScaler()
            X_prime = self.scaler.fit_transform(X_np)
        else:
            self.scaler = None
            X_prime = X_np

        # 計算手法に応じて逆作用素を計算
        if self.method == "numpy_solve":
            start_time = time.time()
            self.A_dagger = self._calculate_numpy_solve(X_prime, Y_np)
            self.computation_time = time.time() - start_time
        elif self.method == "numpy_pinv":
            start_time = time.time()
            self.A_dagger = self._calculate_numpy_pinv(X_prime, Y_np)
            self.computation_time = time.time() - start_time
        elif self.method == "torch_solve":
            self.A_dagger = self._calculate_torch_solve(X_prime, Y_np)
        elif self.method == "torch_pinv":
            self.A_dagger = self._calculate_torch_pinv(X_prime, Y_np)
        elif self.method == "torch_solve_batch":
            self.A_dagger = self._calculate_torch_batch_solve(X_prime, Y_np)
        elif self.method == "torch_pinv_batch":
            self.A_dagger = self._calculate_torch_batch_pinv(X_prime, Y_np)
        elif self.method == "cupy_solve":
            self.A_dagger = self._calculate_cupy_solve(X_prime, Y_np)
        elif self.method == "cupy_pinv":
            self.A_dagger = self._calculate_cupy_pinv(X_prime, Y_np)
        elif self.method == "cupynumeric_solve":
            self.A_dagger = self._calculate_cupynumeric_solve(X_prime, Y_np)
        else:
            raise ValueError(f"未知のメソッドです: {self.method}")
        print(f"手法 '{self.method}' による近似逆作用素の計算が完了しました。")
        print(f"計算時間: {self.computation_time:.4f} 秒")

        return self

    # --- NumPyによる計算 ---
    def _calculate_numpy_solve(self, X, Y):
        X_t = X.T
        Y_t = Y.T
        A = Y_t @ Y_t.T + self.epsilon * np.eye(Y_t.shape[0])
        B = X_t @ Y_t.T
        # A_dagger * A = B を解く
        A_dagger_T = np.linalg.solve(A, B.T)
        return A_dagger_T.T

    def _calculate_numpy_pinv(self, X, Y):
        X_t = X.T
        Y_t = Y.T
        A = Y_t @ Y_t.T + self.epsilon * np.eye(Y_t.shape[0])
        A_pinv = np.linalg.pinv(A)
        B = X_t @ Y_t.T
        return B @ A_pinv

    # --- PyTorchによる計算 ---
    def _calculate_torch_solve(self, X, Y):
        Y_torch = torch.from_numpy(Y.astype(np.float32)).to(self.device)
        X_torch = torch.from_numpy(X.astype(np.float32)).to(self.device)

        Y_t = Y_torch.T
        X_t = X_torch.T

        if self.device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        A = Y_t @ Y_t.T + self.epsilon * torch.eye(
            Y_t.shape[0], device=self.device, dtype=Y_t.dtype
        )
        B = X_t @ Y_t.T

        A_dagger_T = torch.linalg.solve(A, B.T)
        A_dagger = A_dagger_T.T

        if self.device == "cuda":
            torch.cuda.synchronize()
        self.computation_time = time.time() - start_time

        return A_dagger.cpu().numpy()

    def _calculate_torch_pinv(self, X, Y):
        Y_torch = torch.from_numpy(Y.astype(np.float32)).to(self.device)
        X_torch = torch.from_numpy(X.astype(np.float32)).to(self.device)

        Y_t = Y_torch.T
        X_t = X_torch.T

        if self.device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        A = Y_t @ Y_t.T + self.epsilon * torch.eye(
            Y_t.shape[0], device=self.device, dtype=Y_t.dtype
        )
        A_pinv = torch.linalg.pinv(A)
        B = X_t @ Y_t.T
        A_dagger = B @ A_pinv

        if self.device == "cuda":
            torch.cuda.synchronize()
        self.computation_time = time.time() - start_time

        return A_dagger.cpu().numpy()

    # --- PyTorchによる計算 (バッチ) ---
    def _calculate_torch_batch_solve(self, X, Y):
        Y_torch = torch.from_numpy(Y.astype(np.float32)).to(self.device)
        X_torch = torch.from_numpy(X.astype(np.float32)).to(self.device)

        # データをバッチ化 (同じ問題をbatch_size回繰り返す)
        Y_batched = Y_torch.unsqueeze(0).repeat(self.batch_size, 1, 1)
        X_batched = X_torch.unsqueeze(0).repeat(self.batch_size, 1, 1)
        Y_t_batched, X_t_batched = Y_batched.transpose(-2, -1), X_batched.transpose(
            -2, -1
        )

        if self.device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        # バッチ化された YY^T と XY^T を計算（計測対象）
        A_batched = torch.bmm(Y_t_batched, Y_t_batched.transpose(-2, -1))
        I_batched = (
            torch.eye(A_batched.shape[1], device=self.device, dtype=Y_t_batched.dtype)
            .unsqueeze(0)
            .repeat(self.batch_size, 1, 1)
        )
        A_batched = A_batched + self.epsilon * I_batched
        B_batched = torch.bmm(X_t_batched, Y_t_batched.transpose(-2, -1))

        # バッチで連立一次方程式を解く
        A_dagger_batched_T = torch.linalg.solve(A_batched, B_batched.transpose(-2, -1))
        A_dagger_batched = A_dagger_batched_T.transpose(-2, -1)

        if self.device == "cuda":
            torch.cuda.synchronize()
        # バッチ平均時間として保存
        elapsed_time = time.time() - start_time
        self.computation_time = elapsed_time / self.batch_size
        return A_dagger_batched[0].cpu().numpy()

    def _calculate_torch_batch_pinv(self, X, Y):
        Y_torch = torch.from_numpy(Y.astype(np.float32)).to(self.device)
        X_torch = torch.from_numpy(X.astype(np.float32)).to(self.device)

        Y_batched = Y_torch.unsqueeze(0).repeat(self.batch_size, 1, 1)
        X_batched = X_torch.unsqueeze(0).repeat(self.batch_size, 1, 1)
        Y_t_batched, X_t_batched = Y_batched.transpose(-2, -1), X_batched.transpose(
            -2, -1
        )

        # 転送と前処理後に同期し、計算部分のみ計測
        if self.device == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        # バッチ化された YY^T と XY^T を計算（計測対象）
        A_batched = torch.bmm(Y_t_batched, Y_t_batched.transpose(-2, -1))
        I_batched = (
            torch.eye(A_batched.shape[1], device=self.device, dtype=Y_t_batched.dtype)
            .unsqueeze(0)
            .repeat(self.batch_size, 1, 1)
        )
        A_batched = A_batched + self.epsilon * I_batched
        B_batched = torch.bmm(X_t_batched, Y_t_batched.transpose(-2, -1))

        # バッチで疑似逆行列を計算
        A_pinv_batched = torch.linalg.pinv(A_batched)
        A_dagger_batched = torch.bmm(B_batched, A_pinv_batched)

        if self.device == "cuda":
            torch.cuda.synchronize()
        # バッチ平均時間として保存
        elapsed_time = time.time() - start_time
        self.computation_time = elapsed_time / self.batch_size
        return A_dagger_batched[0].cpu().numpy()

    # --- CuPyによる計算 ---
    def _calculate_cupy_solve(self, X, Y):
        Y_cupy = cp.asarray(Y.astype(np.float32))
        X_cupy = cp.asarray(X.astype(np.float32))

        Y_t = Y_cupy.T
        X_t = X_cupy.T

        # 旧: 全体を通して計測（転送含む）
        # 新: 転送後に同期し、計算部分のみ計測
        cp.cuda.Stream.null.synchronize()
        start_time = time.time()

        # 旧: A = Y_t @ Y_t.T + self.epsilon * cp.eye(Y_t.shape[0])
        A = Y_t @ Y_t.T + self.epsilon * cp.eye(Y_t.shape[0], dtype=Y_t.dtype)
        B = X_t @ Y_t.T

        A_dagger_T = cp.linalg.solve(A, B.T)
        A_dagger = A_dagger_T.T

        cp.cuda.Stream.null.synchronize()
        self.computation_time = time.time() - start_time

        return cp.asnumpy(A_dagger)

    def _calculate_cupy_pinv(self, X, Y):
        Y_cupy = cp.asarray(Y.astype(np.float32))
        X_cupy = cp.asarray(X.astype(np.float32))

        Y_t = Y_cupy.T
        X_t = X_cupy.T

        cp.cuda.Stream.null.synchronize()
        start_time = time.time()

        # 旧: A = Y_t @ Y_t.T + self.epsilon * cp.eye(Y_t.shape[0])
        A = Y_t @ Y_t.T + self.epsilon * cp.eye(Y_t.shape[0], dtype=Y_t.dtype)
        A_pinv = cp.linalg.pinv(A)
        B = X_t @ Y_t.T
        A_dagger = B @ A_pinv

        cp.cuda.Stream.null.synchronize()
        self.computation_time = time.time() - start_time

        return cp.asnumpy(A_dagger)

    def _calculate_cupynumeric_solve(self, X, Y):
        Y_cpn = cpn.array(Y.astype(np.float32))
        X_cpn = cpn.array(X.astype(np.float32))

        Y_t, X_t = Y_cpn.T, X_cpn.T

        start_time = legate_time()
        A = Y_t @ Y_t.T + self.epsilon * cpn.eye(Y_t.shape[0], dtype=cpn.float32)
        B = X_t @ Y_t.T
        A_dagger_cpn = cpn.linalg.solve(A, B.T).T

        # Legateのタイマーはミリ秒を返すため1000で割って秒に変換
        self.computation_time = (legate_time() - start_time) / 1000.0

        # 結果をNumPy配列に戻す (CuPy経由)
        if CUPY_AVAILABLE:
            return cp.asnumpy(cp.asarray(A_dagger_cpn))
        else:
            return np.array(A_dagger_cpn)

    def global_feature_importance(
        self,
        feature_names=None,
        class_names=None,
        top_k=None,
        top_k_criterion="average",
    ):
        """
        Visualize and return a DataFrame of global feature importance.
        For each output dimension we set a 'basis', multiply A_dagger, and
        interpret the results as a 'heatmap' vector. Then we plot bar/heatmap.
        """
        if self.A_dagger is None:
            raise ValueError("Please create an explainer first using create_explainer.")

        dim = self.A_dagger.shape[1]
        data = []
        for t in range(dim):
            basis = np.zeros(dim)
            basis[t] = 1
            heatmap = np.dot(self.A_dagger, basis)
            # normalize
            maxval = np.max(np.abs(heatmap))
            if maxval > 0:
                heatmap = heatmap / maxval
            data.append(heatmap)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.A_dagger.shape[0])]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(dim)]

        df = pd.DataFrame(np.array(data), index=class_names, columns=feature_names)

        if top_k is not None:
            if top_k_criterion == "average":
                top_k_features = df.mean(axis=0).nlargest(top_k).index.tolist()
            elif top_k_criterion == "max":
                top_k_features = df.max(axis=0).nlargest(top_k).index.tolist()
            else:
                raise ValueError(f"Unknown top_k_criterion: {top_k_criterion}")
            df = df.loc[:, top_k_features]

        df_melted = df.reset_index().melt(
            id_vars="index", value_name="values", var_name="feature"
        )
        df_melted = df_melted[
            ~pd.to_numeric(df_melted["values"], errors="coerce").isnull()
        ]

        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=df_melted,
            x="values",
            y="feature",
            hue="index",
            palette="pastel",
            dodge=True,
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title("Global Feature Importance (Bar)")
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.heatmap(df, cmap="Blues", annot=True, fmt=".2f")
        plt.title("Global Feature Importance (Heatmap)")
        plt.show()

        return df

    def global_feature_importance_each(
        self,
        feature_names=None,
        class_names=None,
        top_k=None,
        top_k_criterion="average",
        class_num=0,
    ):
        """
        Similar to global_feature_importance but only for a single output dimension (class_num).
        """
        if self.A_dagger is None:
            raise ValueError("Please create an explainer first using create_explainer.")

        dim = self.A_dagger.shape[1]
        data = []
        for t in range(dim):
            basis = np.zeros(dim)
            basis[t] = 1
            heatmap = np.dot(self.A_dagger, basis)
            maxval = np.max(np.abs(heatmap))
            if maxval > 0:
                heatmap /= maxval
            data.append(heatmap)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.A_dagger.shape[0])]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(dim)]

        df = pd.DataFrame(np.array(data), index=class_names, columns=feature_names)

        # select only one row (the chosen class_num)
        df = df.iloc[class_num : class_num + 1, :]

        if top_k is not None:
            if top_k_criterion == "average":
                top_k_features = df.mean(axis=0).nlargest(top_k).index.tolist()
            elif top_k_criterion == "max":
                top_k_features = df.max(axis=0).nlargest(top_k).index.tolist()
            else:
                raise ValueError(f"Unknown top_k_criterion: {top_k_criterion}")
            df = df.loc[:, top_k_features]

        df_melted = df.reset_index().melt(
            id_vars="index", value_name="values", var_name="feature"
        )
        df_melted = df_melted[
            ~pd.to_numeric(df_melted["values"], errors="coerce").isnull()
        ]

        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=df_melted,
            x="values",
            y="feature",
            hue="index",
            palette="pastel",
            dodge=True,
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title(
            f"Global Feature Importance - Single Class ({class_names[class_num]}) (Bar)"
        )
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.heatmap(df, cmap="Blues", annot=True, fmt=".2f")
        plt.title(
            f"Global Feature Importance - Single Class ({class_names[class_num]}) (Heatmap)"
        )
        plt.show()

        return df

    def local_feature_importance(
        self,
        x,
        y,
        feature_names=None,
        scale=True,
        scaler=None,
        top_k=None,
        ignore_zero_features=True,
    ):
        """
        Local feature importance for a single instance x with target y.
        If scale=True and self.scaler is not None, we transform x accordingly.
        Then compute A_dagger * y * x_prime as a naive decomposition.
        """
        if self.A_dagger is None:
            raise ValueError("Please create an explainer first using create_explainer.")
        if x is None or y is None:
            raise ValueError("Please provide x,y for local explanation.")

        if scale and (self.scaler is not None):
            x_prime = self.scaler.transform([x])[0]
        else:
            x_prime = x

        # naive decomposition: A_dagger @ y * x_prime
        heatmap = np.dot(self.A_dagger, y) * x_prime

        if ignore_zero_features:
            heatmap *= x != 0

        maxval = np.max(np.abs(heatmap))
        if maxval > 0:
            heatmap /= maxval

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(x))]

        df = pd.DataFrame([heatmap], columns=feature_names)

        if top_k is not None:
            # sort by absolute value in descending
            sorted_cols = df.iloc[0, :].abs().sort_values(ascending=False).index
            df = df.loc[:, sorted_cols[:top_k]]

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, orient="h", ax=ax, color="lightblue")
        plt.title("Local Feature Importance")
        plt.show()

        return df

    def local_feature_importance_without_viz(
        self,
        x,
        y,
        feature_names=None,
        scale=True,
        scaler=None,
        top_k=None,
        ignore_zero_features=True,
    ):
        """
        Same as local_feature_importance but returns DataFrame without plotting.
        """
        if self.A_dagger is None:
            raise ValueError("Please create an explainer first using create_explainer.")
        if x is None or y is None:
            raise ValueError("Please provide x,y for local explanation.")

        if scale and (self.scaler is not None):
            x_prime = self.scaler.transform([x])[0]
        else:
            x_prime = x

        heatmap = np.dot(self.A_dagger, y) * x_prime
        if ignore_zero_features:
            heatmap *= x != 0

        maxval = np.max(np.abs(heatmap))
        if maxval > 0:
            heatmap /= maxval

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(x))]

        df = pd.DataFrame([heatmap], columns=feature_names)

        if top_k is not None:
            sorted_cols = df.iloc[0, :].abs().sort_values(ascending=False).index
            df = df.loc[:, sorted_cols[:top_k]]

        return df

    def global_feature_importance_without_viz(
        self,
        feature_names=None,
        class_names=None,
        top_k=None,
        top_k_criterion="average",
    ):
        """
        Return a DataFrame of global feature importance without any plotting.
        Similar to global_feature_importance, but no visualization.
        """
        if self.A_dagger is None:
            raise ValueError("Please create an explainer first using create_explainer.")

        dim = self.A_dagger.shape[1]
        data = []
        for t in range(dim):
            basis = np.zeros(dim)
            basis[t] = 1
            heatmap = np.dot(self.A_dagger, basis)
            maxval = np.max(np.abs(heatmap))
            if maxval > 0:
                heatmap /= maxval
            data.append(heatmap)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.A_dagger.shape[0])]
        if class_names is None:
            class_names = [f"class_{i}" for i in range(dim)]

        df = pd.DataFrame(np.array(data), index=class_names, columns=feature_names)

        if top_k is not None:
            if top_k_criterion == "average":
                top_k_features = df.mean(axis=0).nlargest(top_k).index.tolist()
            elif top_k_criterion == "max":
                top_k_features = df.max(axis=0).nlargest(top_k).index.tolist()
            else:
                raise ValueError(f"Unknown top_k_criterion: {top_k_criterion}")
            df = df.loc[:, top_k_features]

        return df

    def rbf_kernel(self, v1, v2, gamma):
        sq_dist = (
            np.sum(v1**2, 1).reshape(-1, 1) + np.sum(v2**2, 1) - 2 * np.dot(v1, v2.T)
        )
        return np.exp(-gamma * sq_dist)

    def plot_rep_instance_similarity(
        self,
        X,
        Y,
        x=None,
        feature_names=None,
        class_names=None,
        gamma=0.1,
        scaler=None,
        class_indices=[0, 1],
        dim_reduce=None,
        n_components=2,
        x_range=None,
        y_range=None,
    ):
        if self.A_dagger is None:
            raise ValueError("Please create an explainer first using create_explainer.")
        repvec = []
        dim = self.A_dagger.shape[1]
        for t in range(dim):
            basis = np.zeros(dim)
            basis[t] = 1
            rep = np.dot(self.A_dagger, basis)
            repvec.append(rep)

        if class_names is None:
            vec_name = ["Class " + str(i) + " repvec" for i in range(dim)]
        else:
            vec_name = class_names

        repvec = np.array(repvec)
        repvec_in = repvec
        if scaler is not None:
            repvec_in = scaler.inverse_transform(repvec)

        if feature_names is None:
            repdf = pd.DataFrame(repvec_in, index=vec_name)
        else:
            repdf = pd.DataFrame(repvec_in, index=vec_name, columns=feature_names)

        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        if x is not None:
            x_scaled = scaler.transform([x]) if scaler else x.reshape(1, -1)
        else:
            x_scaled = None

        if dim_reduce is None:
            if x is not None:
                total_data_transformed = np.concatenate(
                    [X_scaled, repvec, x_scaled], axis=0
                )
            else:
                total_data_transformed = np.concatenate([X_scaled, repvec], axis=0)

        if dim_reduce == "pca":
            pca = PCA(n_components=n_components)
            X_scaled = pca.fit_transform(X_scaled)
            repvec = pca.transform(repvec)
            if x is not None:
                x_scaled = pca.transform(x_scaled)
            total_data_transformed = (
                np.concatenate([X_scaled, repvec, x_scaled], axis=0)
                if x is not None
                else np.concatenate([X_scaled, repvec], axis=0)
            )
        elif dim_reduce == "umap":
            umapp = umap.UMAP(n_components=n_components)
            X_scaled = umapp.fit_transform(X_scaled)
            repvec = umapp.transform(repvec)
            if x is not None:
                x_scaled = umapp.transform(x_scaled)
            total_data_transformed = (
                np.concatenate([X_scaled, repvec, x_scaled], axis=0)
                if x is not None
                else np.concatenate([X_scaled, repvec], axis=0)
            )
        elif dim_reduce == "tsne":
            if x is not None:
                x_scaled = scaler.transform([x]) if scaler else x
                total_data = np.concatenate([X_scaled, repvec, x_scaled], axis=0)
            else:
                total_data = np.concatenate([X_scaled, repvec], axis=0)

            tsne = TSNE(n_components=n_components)
            total_data_transformed = tsne.fit_transform(total_data)

            X_scaled = total_data_transformed[: len(X_scaled)]
            repvec = total_data_transformed[len(X_scaled) : len(X_scaled) + len(repvec)]
            if x is not None:
                x_scaled = total_data_transformed[-1].reshape(1, -1)

        if x is not None:
            X_scaled = total_data_transformed[: len(X_scaled)]
            repvec = total_data_transformed[len(X_scaled) : -1]
            x_scaled = total_data_transformed[-1].reshape(1, -1)
        else:
            X_scaled = total_data_transformed[: len(X_scaled)]
            repvec = total_data_transformed[len(X_scaled) :]

        res = self.rbf_kernel(X_scaled, repvec, gamma)

        if class_names is None:
            resdf = pd.DataFrame(res, columns=["score_" + str(i) for i in range(dim)])
        else:
            resdf = pd.DataFrame(
                res, columns=[class_names[i] + " score" for i in range(dim)]
            )

        max_indices = np.argmax(Y, axis=1)
        resdf["result"] = max_indices

        for idx in class_indices:
            sns.kdeplot(
                data=resdf[resdf["result"] == idx],
                x=resdf.columns[class_indices[0]],
                y=resdf.columns[class_indices[1]],
                fill=True,
                alpha=0.5,
            )

        if x is not None:
            if scaler is not None:
                x_scaled = scaler.transform([x]).reshape(1, -1)
            else:
                x_scaled = x.reshape(1, -1)

            if dim_reduce == "pca":
                x_scaled = pca.transform(x_scaled)
            elif dim_reduce == "tsne":
                x_scaled = tsne.transform(x_scaled)  # 上記で完了
            elif dim_reduce == "umap":
                x_scaled = umapp.transform(x_scaled)

            focus_inst_scores = self.rbf_kernel(x_scaled, repvec, gamma)
            plt.scatter(
                focus_inst_scores[0][class_indices[0]],
                focus_inst_scores[0][class_indices[1]],
                color="gold",
                edgecolors="black",
                linewidths=2,
                marker="o",
                s=100,
            )

        if x_range is not None:
            plt.xlim(x_range)
        if y_range is not None:
            plt.ylim(y_range)

        plt.show()
        return repdf, resdf
