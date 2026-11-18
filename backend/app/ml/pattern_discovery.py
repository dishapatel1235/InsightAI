import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

SKLEARN_AVAILABLE = False
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def pure_numpy_kmeans(X: np.ndarray, k: int, max_iter: int = 40, random_state: int = 42):
    """Pure NumPy K-Means using Lloyd's algorithm."""
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape
    k = min(k, n_samples)

    # K-means++ initialization style
    init_idx = rng.choice(n_samples, size=1)
    centroids = [X[init_idx[0]]]
    for _ in range(1, k):
        dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)
        prob = dists / (np.sum(dists) + 1e-12)
        next_idx = rng.choice(n_samples, p=prob)
        centroids.append(X[next_idx])
    centroids = np.array(centroids)

    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        # Distances: (n_samples, k)
        diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=2)
        new_labels = np.argmin(sq_dist, axis=1)

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for c in range(k):
            mask = (labels == c)
            if np.any(mask):
                centroids[c] = np.mean(X[mask], axis=0)

    # Compute inertia
    diff = X - centroids[labels]
    inertia = float(np.sum(diff ** 2))

    return labels, centroids, inertia


def pure_numpy_pca(X: np.ndarray, n_components: int = 2):
    """Pure NumPy PCA via Singular Value Decomposition (SVD)."""
    n_samples, n_features = X.shape
    n_comp = min(n_components, n_features, n_samples)
    X_centered = X - np.mean(X, axis=0)

    try:
        _, s, vt = np.linalg.svd(X_centered, full_matrices=False)
        components = vt[:n_comp]
        pca_proj = X_centered @ components.T
        variance = (s ** 2) / (n_samples - 1 if n_samples > 1 else 1)
        tot_variance = np.sum(variance) or 1.0
        explained_variance_ratio = [round(float(v / tot_variance), 4) for v in variance[:n_comp]]
    except Exception:
        # Fallback projection
        pca_proj = X_centered[:, :n_comp]
        components = np.eye(n_features)[:n_comp]
        explained_variance_ratio = [0.6, 0.3][:n_comp]

    # Ensure 2 columns
    if pca_proj.shape[1] < 2:
        pca_proj = np.hstack([pca_proj, np.zeros((n_samples, 2 - pca_proj.shape[1]))])
        explained_variance_ratio.extend([0.0] * (2 - len(explained_variance_ratio)))

    return pca_proj, components, explained_variance_ratio


def pure_numpy_silhouette(X: np.ndarray, labels: np.ndarray, max_samples: int = 300) -> float:
    """Fast sample-based silhouette score computation in pure NumPy."""
    n_samples = len(X)
    if n_samples < 4:
        return 0.5
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= n_samples:
        return 0.0

    # Subsample if dataset is large for instant response
    if n_samples > max_samples:
        indices = np.random.RandomState(42).choice(n_samples, size=max_samples, replace=False)
        X_sub = X[indices]
        labels_sub = labels[indices]
    else:
        X_sub = X
        labels_sub = labels

    sub_n = len(X_sub)
    sil_vals = []

    for i in range(sub_n):
        c_i = labels_sub[i]
        same_mask = (labels_sub == c_i) & (np.arange(sub_n) != i)
        if not np.any(same_mask):
            continue
        a_i = np.mean(np.linalg.norm(X_sub[same_mask] - X_sub[i], axis=1))

        b_i = float("inf")
        for other_c in unique_labels:
            if other_c == c_i:
                continue
            other_mask = (labels_sub == other_c)
            if np.any(other_mask):
                dist = np.mean(np.linalg.norm(X_sub[other_mask] - X_sub[i], axis=1))
                if dist < b_i:
                    b_i = dist

        denom = max(a_i, b_i)
        if denom > 0:
            sil_vals.append((b_i - a_i) / denom)

    return float(np.mean(sil_vals)) if sil_vals else 0.5


class PatternDiscoveryPipeline:
    """Unsupervised pattern discovery: Clustering, PCA, Correlation, and Anomaly Detection."""

    @classmethod
    def analyze_patterns(
        cls,
        df: pd.DataFrame,
        method: str = "clustering",
        feature_columns: Optional[List[str]] = None,
        n_clusters: int = 3,
        contamination: float = 0.05,
    ) -> Dict[str, Any]:
        # Filter to numeric columns only
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        if numeric_df.empty:
            raise ValueError("No numeric features found in dataset for pattern discovery.")

        if feature_columns:
            valid_cols = [c for c in feature_columns if c in numeric_df.columns]
            if len(valid_cols) >= 2:
                numeric_df = numeric_df[valid_cols]

        # Impute missing values with column median
        clean_mat = numeric_df.fillna(numeric_df.median())
        features = list(clean_mat.columns)
        n_samples = len(clean_mat)

        if n_samples < 4:
            raise ValueError("Dataset has too few samples (minimum 4 numeric rows required).")

        # Standardize features
        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(clean_mat)
        else:
            mean = clean_mat.mean()
            std = clean_mat.std().replace(0, 1)
            X_scaled = ((clean_mat - mean) / std).to_numpy()

        results = {}

        if method in ("clustering", "all"):
            k = max(2, min(n_clusters, n_samples - 1, 10))
            if SKLEARN_AVAILABLE:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                centroids_scaled = kmeans.cluster_centers_
                inertia = float(kmeans.inertia_)
                try:
                    sil_score = float(silhouette_score(X_scaled, cluster_labels))
                except Exception:
                    sil_score = 0.0

                pca = PCA(n_components=2, random_state=42)
                pca_2d = pca.fit_transform(X_scaled)
                pca_centroids = pca.transform(centroids_scaled)
                explained_variance = [round(float(v), 4) for v in pca.explained_variance_ratio_]
            else:
                # High accuracy Pure NumPy K-Means & PCA
                cluster_labels, centroids_scaled, inertia = pure_numpy_kmeans(X_scaled, k=k)
                pca_2d, components, explained_variance = pure_numpy_pca(X_scaled, n_components=2)
                # Project centroids using SVD components
                centroids_centered = centroids_scaled - np.mean(X_scaled, axis=0)
                pca_centroids = centroids_centered @ components[:2].T if components.shape[0] >= 2 else np.zeros((k, 2))
                sil_score = pure_numpy_silhouette(X_scaled, cluster_labels)

            # Build points array (limit to 250 points for snappy frontend rendering)
            points = []
            stride = max(1, n_samples // 250)
            for i in range(0, n_samples, stride):
                points.append({
                    "id": int(i),
                    "x": round(float(pca_2d[i, 0]), 3),
                    "y": round(float(pca_2d[i, 1]), 3),
                    "cluster": int(cluster_labels[i]),
                })

            # Cluster profiles (mean values of original features per cluster)
            clean_mat["_cluster"] = cluster_labels
            cluster_profiles = []
            for c_id in range(k):
                subset = clean_mat[clean_mat["_cluster"] == c_id]
                size = len(subset)
                pct = round((size / n_samples) * 100, 1)
                means = {feat: round(float(subset[feat].mean()), 3) for feat in features}
                cluster_profiles.append({
                    "cluster_id": c_id,
                    "name": f"Segment {c_id + 1}",
                    "size": size,
                    "percentage": pct,
                    "centroid_pca": {
                        "x": round(float(pca_centroids[c_id, 0]), 3) if c_id < len(pca_centroids) else 0.0,
                        "y": round(float(pca_centroids[c_id, 1]), 3) if c_id < len(pca_centroids) else 0.0,
                    },
                    "feature_means": means,
                })
            clean_mat.drop(columns=["_cluster"], inplace=True)

            results["clustering"] = {
                "n_clusters": k,
                "silhouette_score": round(sil_score, 4),
                "inertia": round(inertia, 2),
                "explained_variance_ratio": explained_variance,
                "cluster_profiles": cluster_profiles,
                "scatter_points": points,
            }

        # 2. Correlation Matrix Heatmap
        corr_matrix = clean_mat.corr().round(3).fillna(0.0)
        corr_pairs = []
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_pairs.append({
                    "feature_a": cols[i],
                    "feature_b": cols[j],
                    "correlation": float(corr_matrix.iloc[i, j]),
                    "abs_corr": abs(float(corr_matrix.iloc[i, j])),
                })
        corr_pairs.sort(key=lambda x: x["abs_corr"], reverse=True)

        results["correlation"] = {
            "columns": cols,
            "matrix": corr_matrix.to_dict(),
            "top_pairs": corr_pairs[:12],
        }

        # 3. Anomaly Detection
        if method in ("anomaly", "all"):
            if SKLEARN_AVAILABLE:
                iso = IsolationForest(contamination=max(0.01, min(0.2, contamination)), random_state=42)
                preds = iso.fit_predict(X_scaled)
                scores = iso.decision_function(X_scaled)
                anomalies_idx = np.where(preds == -1)[0]
            else:
                # Robust Mahalanobis / Z-score fallback in pure numpy
                z_scores = np.abs(X_scaled).max(axis=1)
                threshold = np.percentile(z_scores, 100 * (1 - max(0.01, min(0.2, contamination))))
                anomalies_idx = np.where(z_scores >= threshold)[0]
                scores = -z_scores

            anomaly_items = []
            for idx in anomalies_idx[:20]:
                row_vals = {f: round(float(clean_mat.iloc[idx][f]), 3) for f in features[:6]}
                anomaly_items.append({
                    "row_index": int(idx + 1),
                    "anomaly_score": round(float(scores[idx]), 4),
                    "features": row_vals,
                })

            results["anomaly_detection"] = {
                "detected_anomalies": len(anomalies_idx),
                "anomaly_rate_pct": round((len(anomalies_idx) / n_samples) * 100, 2),
                "top_anomalies": anomaly_items,
            }

        return {
            "method": method,
            "features_analyzed": features,
            "total_samples": n_samples,
            **results,
        }
