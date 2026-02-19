from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

from .config import ReductionConfig


def reduce_dimensions(features: np.ndarray, cfg: ReductionConfig) -> np.ndarray:
    """Two-stage dimensionality reduction: PCA then UMAP.

    Input:  (N, D)  — raw feature vectors from the model
    Output: (N, umap_components) — compact embeddings for clustering
    """
    n_samples, n_features = features.shape

    # --- PCA ---
    pca_dim = min(cfg.pca_components, n_features, n_samples)
    print(f"  PCA: {n_features} -> {pca_dim}")
    pca = PCA(n_components=pca_dim, random_state=42)
    reduced = pca.fit_transform(features)
    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA пояснює {explained:.1f}% дисперсії")

    # --- UMAP ---
    umap_dim = min(cfg.umap_components, pca_dim)
    n_neighbors = min(cfg.umap_n_neighbors, n_samples - 1)
    print(f"  UMAP: {pca_dim} -> {umap_dim} (n_neighbors={n_neighbors}, metric={cfg.umap_metric})")

    umap = UMAP(
        n_components=umap_dim,
        n_neighbors=n_neighbors,
        min_dist=cfg.umap_min_dist,
        metric=cfg.umap_metric,
        random_state=42,
        verbose=False,
    )
    embedded = umap.fit_transform(reduced)
    print(f"  Фінальна розмірність: {embedded.shape}")

    return embedded
