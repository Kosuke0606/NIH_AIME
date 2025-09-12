import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class AIME:
    """
    AIME class with optional Huber-loss-based IRLS for robust approximate inverse operator.
    By default, it uses the classical pseudo-inverse approach (non-robust).
    """

    def __init__(self, use_huber=False, delta=1.0, max_iter=50, tol=1e-5):
        """
        Parameters
        ----------
        use_huber : bool
            If True, use Huber loss + IRLS for robust inverse operator.
            If False, use the standard pseudo-inverse approach.
        delta : float
            Huber-loss threshold parameter (ignored if use_huber=False).
        max_iter : int
            Maximum iteration for IRLS (ignored if use_huber=False).
        tol : float
            Convergence tolerance for IRLS (ignored if use_huber=False).
        """
        self.use_huber = use_huber
        self.delta = delta
        self.max_iter = max_iter
        self.tol = tol

        self.A_dagger = None
        self.scaler = None

    def create_explainer(self, X, Y, normalize=True):
        """
        Create an explainer by deriving the approximate inverse operator
        from input X and output Y. If use_huber=True, applies IRLS with Huber loss.

        Parameters
        ----------
        X : array-like of shape (N, n)
            Input data.
        Y : array-like of shape (N, m)
            Output data.
        normalize : bool
            If True, apply standard scaling to X before computing the operator.

        Returns
        -------
        self : AIME
            Fitted explainer with self.A_dagger as the inverse operator.
        """
        if X is None or Y is None:
            raise ValueError("Both X and Y must be provided.")
        self.A_dagger, self.scaler = self._generate_inverse_operator_from_y(
            X, Y, normalize
        )
        return self

    def _generate_inverse_operator_from_y(self, X, Y, normalize=True):
        """
        Internal function to compute the approximate inverse operator M (A_dagger).
        Depending on 'use_huber', either:
          - standard pseudo-inverse approach (if use_huber=False),
          - IRLS with Huber loss (if use_huber=True).
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        # Optional normalization of X
        if normalize:
            scaler = StandardScaler()
            X_prime = scaler.fit_transform(X)
        else:
            scaler = None
            X_prime = X

        # Transpose to shape (n, N) and (m, N) for consistency with AIME
        X_prime_t = X_prime.T  # shape (n, N)
        Y_t = Y.T  # shape (m, N)

        if not self.use_huber:
            # --- Standard AIME approach: M = X Y^T (Y Y^T)^(-1) in pseudo-inverse form ---
            # But we do Y_pinv = pinv(Y) => shape (m x N) -> (N x m)
            # Then M = (Y^+) * X^T  => shape (m x N) x (N x n) => (m x n), we want (n x m)?
            # Actually we do:  M = X' * Y^+ (since X' is n x N, Y^+ is N x m => result n x m)
            # so let's do pinv(Y_t) => shape (N x m).
            # M = X_prime_t * pinv(Y_t) => shape n x m

            # we can do Y_t_pinv = np.linalg.pinv(Y_t)
            # M = X_prime_t dot Y_t_pinv
            Y_t_pinv = np.linalg.pinv(Y_t)  # shape (N, m)
            A_dagger = X_prime_t @ Y_t_pinv  # shape (n, m)

        else:
            # --- IRLS with Huber loss ---
            A_dagger = self._huber_inverse_operator(X_prime_t, Y_t)

        return A_dagger, scaler

    def _huber_inverse_operator(self, X_t, Y_t):
        """
        Solve min sum_{i=1..N} huber( ||X_i - M Y_i|| ) via IRLS.
        X_t : (n, N)
        Y_t : (m, N)
        Return M shape (n, m).
        """
        n, N = X_t.shape
        m = Y_t.shape[0]
        if Y_t.shape[1] != N:
            raise ValueError("Dimension mismatch in X_t and Y_t.")

        # 1) Initialize M e.g. with pseudo-inverse solution
        try:
            YtY_inv = np.linalg.inv(Y_t @ Y_t.T)  # shape (m, m)
            M_init = X_t @ (Y_t.T @ YtY_inv)  # shape (n, m)
        except np.linalg.LinAlgError:
            # fallback: pseudo-inverse
            YtY_pinv = np.linalg.pinv(Y_t @ Y_t.T)
            M_init = X_t @ (Y_t.T @ YtY_pinv)

        M = M_init.copy()

        delta = self.delta
        max_iter = self.max_iter
        tol = self.tol

        for _ in range(max_iter):
            # 2) Compute residuals r_i = || x_i - M y_i ||
            # shape: x_i is (n,), M y_i is also (n,)
            # but let's do in matrix form:
            # R = X_t - M @ Y_t  => shape (n, N)
            R = X_t - (M @ Y_t)  # shape (n, N)
            residuals = np.linalg.norm(R, axis=0)  # shape (N,)

            # 3) compute weights w_i based on Huber
            w = np.ones_like(residuals)
            mask_large = residuals > delta
            w[mask_large] = delta / residuals[mask_large]

            # 4) Weighted least squares
            #   M^{(k+1)} = argmin sum w_i^2 || x_i - M y_i ||^2
            # define W^(k)
            # multiply each column i of X_t, Y_t by sqrt(w_i)
            W_sqrt = np.sqrt(w)  # shape (N,)
            X_w = X_t * W_sqrt  # broadcasting along columns
            Y_w = Y_t * W_sqrt

            # solve M_new = X_w (Y_w)^T ( Y_w (Y_w)^T )^-1
            # shape: X_w is (n,N), Y_w is (m,N) => Y_w^T is (N,m)
            # Y_w (Y_w)^T is (m,m)
            try:
                tmp_inv = np.linalg.inv(Y_w @ Y_w.T)  # shape (m,m)
            except np.linalg.LinAlgError:
                tmp_inv = np.linalg.pinv(Y_w @ Y_w.T)

            M_new = X_w @ (Y_w.T @ tmp_inv)  # shape (n,m)

            # check convergence
            diff = np.linalg.norm(M_new - M, ord="fro")
            M = M_new
            if diff < tol:
                break

        return M

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
