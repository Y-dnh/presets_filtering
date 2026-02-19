from __future__ import annotations

from collections import Counter

import hdbscan
import numpy as np
from sklearn.neighbors import NearestCentroid

from .config import ClusteringConfig


def cluster(embeddings: np.ndarray, cfg: ClusteringConfig) -> np.ndarray:
    """Run HDBSCAN on embeddings and return cluster labels (0-indexed).

    Noise points (label == -1) are either kept separate or reassigned to the
    nearest cluster depending on ``cfg.noise_handling``.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=cfg.min_cluster_size,
        min_samples=cfg.min_samples,
        cluster_selection_epsilon=cfg.cluster_selection_epsilon,
        cluster_selection_method=cfg.cluster_selection_method,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"  Знайдено кластерів: {n_clusters}")
    print(f"  Шумових точок: {n_noise} ({n_noise / len(labels) * 100:.1f}%)")

    if cfg.noise_handling == "nearest" and n_noise > 0 and n_clusters > 0:
        labels = _assign_noise_to_nearest(embeddings, labels)
        print(f"  Шумові точки призначено до найближчих кластерів")

    labels = _renumber_labels(labels)
    return labels


def _assign_noise_to_nearest(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Assign noise points to their nearest cluster centroid."""
    labels = labels.copy()
    mask_valid = labels != -1
    mask_noise = labels == -1

    clf = NearestCentroid()
    clf.fit(embeddings[mask_valid], labels[mask_valid])
    labels[mask_noise] = clf.predict(embeddings[mask_noise])
    return labels


def _renumber_labels(labels: np.ndarray) -> np.ndarray:
    """Renumber cluster labels to consecutive integers starting from 0.
    Noise (-1) stays -1."""
    unique = sorted(set(labels) - {-1})
    mapping = {old: new for new, old in enumerate(unique)}
    mapping[-1] = -1
    return np.array([mapping[l] for l in labels])


def build_report(labels: np.ndarray) -> str:
    """Build a human-readable report table of cluster sizes."""
    counts = Counter(labels)
    lines = [
        "",
        "=" * 40,
        f"  {'Пресет':<15} {'Зображень':>10}",
        "-" * 40,
    ]

    for label in sorted(counts.keys()):
        if label == -1:
            name = "noise"
        else:
            name = f"preset_{label:03d}"
        lines.append(f"  {name:<15} {counts[label]:>10}")

    lines.append("-" * 40)
    lines.append(f"  {'ВСЬОГО':<15} {sum(counts.values()):>10}")
    lines.append("=" * 40)
    lines.append("")
    return "\n".join(lines)
