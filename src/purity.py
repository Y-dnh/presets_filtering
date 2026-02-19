from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from .config import PurityConfig


def _safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(diff, axis=2)


def evaluate_and_save_purity(
    features: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    viz_dir_name: str,
    cfg: PurityConfig,
) -> Dict:
    """Evaluate cluster isolation and save purity report artifacts."""
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    viz_dir = output_dir / viz_dir_name
    viz_dir.mkdir(parents=True, exist_ok=True)

    valid_mask = labels != -1
    valid_features = features[valid_mask]
    valid_embeddings = embeddings[valid_mask]
    valid_labels = labels[valid_mask]
    cluster_ids = sorted(set(valid_labels.tolist()))

    report: Dict = {
        "run": {
            "timestamp": run_ts,
            "n_images": int(len(labels)),
            "n_clusters": int(len(cluster_ids)),
        },
        "thresholds": {
            "max_centroid_cosine": cfg.max_centroid_cosine,
            "min_silhouette": cfg.min_silhouette,
            "max_nn_cross_ratio": cfg.max_nn_cross_ratio,
            "k_neighbors": cfg.k_neighbors,
        },
    }

    if len(cluster_ids) < 2:
        report["metrics"] = {
            "max_centroid_cosine": None,
            "min_inter_centroid_dist_umap": None,
            "silhouette_mean": None,
            "max_nn_cross_ratio": None,
        }
        report["per_cluster"] = []
        report["leakage_candidates"] = [
            {"reason": "insufficient_clusters", "details": "Потрібно щонайменше 2 кластери"}
        ]
        report["verdict"] = "FAIL"

        latest = viz_dir / "purity_report_latest.json"
        latest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        ts_report = viz_dir / f"purity_report_{run_ts}.json"
        ts_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    # --- Centroids in feature-space ---
    cluster_centroids_f: List[np.ndarray] = []
    cluster_centroids_e: List[np.ndarray] = []
    sizes: Dict[int, int] = {}
    for cid in cluster_ids:
        mask = valid_labels == cid
        sizes[cid] = int(mask.sum())
        c_feat = valid_features[mask].mean(axis=0)
        c_feat = c_feat / max(np.linalg.norm(c_feat), 1e-8)
        cluster_centroids_f.append(c_feat)
        cluster_centroids_e.append(valid_embeddings[mask].mean(axis=0))

    centroids_f = np.stack(cluster_centroids_f, axis=0)
    centroids_e = np.stack(cluster_centroids_e, axis=0)

    cosine_matrix = centroids_f @ centroids_f.T
    dist_matrix = _pairwise_euclidean(centroids_e)

    diag = np.eye(len(cluster_ids), dtype=bool)
    cosine_off = cosine_matrix[~diag]
    dist_off = dist_matrix[~diag]
    max_centroid_cos = float(cosine_off.max()) if cosine_off.size else 0.0
    min_inter_dist = float(dist_off.min()) if dist_off.size else 0.0

    # --- Silhouette in embedding-space ---
    silhouette = None
    if len(valid_embeddings) > len(cluster_ids):
        try:
            silhouette = float(silhouette_score(valid_embeddings, valid_labels))
        except Exception:
            silhouette = None

    # --- kNN cross-cluster leakage in feature-space ---
    n_samples = len(valid_labels)
    k = min(cfg.k_neighbors, max(1, n_samples - 1))
    n_neighbors = min(k + 1, n_samples)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(valid_features)
    indices = nn.kneighbors(return_distance=False)

    per_cluster_ratios: Dict[int, List[float]] = {cid: [] for cid in cluster_ids}
    neighbor_cluster_votes: Dict[int, Dict[int, int]] = {cid: {} for cid in cluster_ids}

    for i in range(n_samples):
        own_cid = int(valid_labels[i])
        neigh = indices[i].tolist()
        neigh = [j for j in neigh if j != i][:k]
        if not neigh:
            continue
        neigh_labels = [int(valid_labels[j]) for j in neigh]
        cross = [c for c in neigh_labels if c != own_cid]
        ratio = len(cross) / len(neigh_labels)
        per_cluster_ratios[own_cid].append(ratio)
        for c in cross:
            neighbor_cluster_votes[own_cid][c] = neighbor_cluster_votes[own_cid].get(c, 0) + 1

    per_cluster_mean_ratio = {
        cid: float(np.mean(vals)) if vals else 0.0
        for cid, vals in per_cluster_ratios.items()
    }
    max_nn_cross_ratio = max(per_cluster_mean_ratio.values()) if per_cluster_mean_ratio else 0.0

    # --- Per-cluster diagnostics ---
    per_cluster = []
    leakage_candidates = []
    id_to_idx = {cid: idx for idx, cid in enumerate(cluster_ids)}
    for cid in cluster_ids:
        idx = id_to_idx[cid]
        cos_row = cosine_matrix[idx].copy()
        dist_row = dist_matrix[idx].copy()
        cos_row[idx] = -1.0
        dist_row[idx] = np.inf
        nearest_cos_idx = int(np.argmax(cos_row))
        nearest_dist_idx = int(np.argmin(dist_row))

        nearest_other_cos = cluster_ids[nearest_cos_idx]
        nearest_other_dist = cluster_ids[nearest_dist_idx]
        dominant_neighbor = None
        if neighbor_cluster_votes[cid]:
            dominant_neighbor = max(
                neighbor_cluster_votes[cid].items(),
                key=lambda x: x[1],
            )[0]

        per_cluster.append(
            {
                "preset": f"preset_{cid:03d}",
                "cluster_id": int(cid),
                "size": sizes[cid],
                "nearest_other_by_cosine": (
                    f"preset_{nearest_other_cos:03d}" if nearest_other_cos is not None else None
                ),
                "cos_to_nearest": _safe_float(cos_row[nearest_cos_idx]),
                "nearest_other_by_distance": (
                    f"preset_{nearest_other_dist:03d}" if nearest_other_dist is not None else None
                ),
                "dist_to_nearest": _safe_float(dist_row[nearest_dist_idx]),
                "nn_cross_ratio": _safe_float(per_cluster_mean_ratio[cid]),
                "dominant_neighbor_cluster": (
                    f"preset_{dominant_neighbor:03d}" if dominant_neighbor is not None else None
                ),
            }
        )

    # Centroid similarity leakage candidates
    for i, cid_a in enumerate(cluster_ids):
        for j in range(i + 1, len(cluster_ids)):
            cid_b = cluster_ids[j]
            sim = float(cosine_matrix[i, j])
            if sim >= cfg.max_centroid_cosine:
                leakage_candidates.append(
                    {
                        "type": "high_centroid_similarity",
                        "cluster_a": f"preset_{cid_a:03d}",
                        "cluster_b": f"preset_{cid_b:03d}",
                        "cosine": sim,
                    }
                )

    # Boundary overlap leakage candidates
    for cid, ratio in per_cluster_mean_ratio.items():
        if ratio >= cfg.max_nn_cross_ratio:
            dominant_neighbor = None
            if neighbor_cluster_votes[cid]:
                dominant_neighbor = max(
                    neighbor_cluster_votes[cid].items(),
                    key=lambda x: x[1],
                )[0]
            leakage_candidates.append(
                {
                    "type": "boundary_overlap",
                    "cluster_a": f"preset_{cid:03d}",
                    "cluster_b": (
                        f"preset_{dominant_neighbor:03d}" if dominant_neighbor is not None else None
                    ),
                    "nn_cross_ratio": float(ratio),
                }
            )

    metrics = {
        "max_centroid_cosine": _safe_float(max_centroid_cos),
        "min_inter_centroid_dist_umap": _safe_float(min_inter_dist),
        "silhouette_mean": _safe_float(silhouette),
        "max_nn_cross_ratio": _safe_float(max_nn_cross_ratio),
    }

    pass_cos = max_centroid_cos < cfg.max_centroid_cosine
    pass_sil = (silhouette is not None) and (silhouette >= cfg.min_silhouette)
    pass_nn = max_nn_cross_ratio < cfg.max_nn_cross_ratio
    verdict = "PASS" if (pass_cos and pass_sil and pass_nn) else "FAIL"

    report["metrics"] = metrics
    report["checks"] = {
        "centroid_cosine": pass_cos,
        "silhouette": pass_sil,
        "nn_cross_ratio": pass_nn,
    }
    report["verdict"] = verdict
    report["leakage_candidates"] = leakage_candidates
    report["per_cluster"] = per_cluster

    # Save artifacts
    np.save(viz_dir / "centroid_cosine_matrix.npy", cosine_matrix)
    np.save(viz_dir / f"centroid_cosine_matrix_{run_ts}.npy", cosine_matrix)
    latest = viz_dir / "purity_report_latest.json"
    latest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    ts_report = viz_dir / f"purity_report_{run_ts}.json"
    ts_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("=== PURITY REPORT ===")
    print(f"  max_centroid_cosine:  {max_centroid_cos:.4f} (поріг < {cfg.max_centroid_cosine})")
    if silhouette is None:
        print(f"  silhouette_mean:      n/a (поріг >= {cfg.min_silhouette})")
    else:
        print(f"  silhouette_mean:      {silhouette:.4f} (поріг >= {cfg.min_silhouette})")
    print(f"  max_nn_cross_ratio:   {max_nn_cross_ratio:.4f} (поріг < {cfg.max_nn_cross_ratio})")
    print(f"  verdict:              {verdict}")
    print(f"  report:               {latest}")
    print("")

    return report
