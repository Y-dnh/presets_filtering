"""Ітеративний тюнер кластеризації для PTZ/тепловізора.

Запуск:
    python iterative_tuning.py

Скрипт:
  1) сканує зображення
  2) один раз витягує ознаки (кеш використовується)
  3) проганяє кілька preset-конфігурацій UMAP/HDBSCAN
  4) для кожної ітерації рахує purity-метрики і зберігає звіт
  5) друкує коротку таблицю результатів і "найкращу" ітерацію
"""

from __future__ import annotations

import copy
import json
import sys
import time
from itertools import product

from src.clusterer import cluster
from src.config import load_config
from src.feature_extractor import extract_features
from src.image_scanner import scan_images
from src.purity import evaluate_and_save_purity
from src.reducer import reduce_dimensions


# ==========================
# БАЗОВА КОНФІГУРАЦІЯ СКРИПТА
# ==========================
BASE_CONFIG_PATH = "config.yaml"


# ==========================
# GRID SEARCH (БАГАТО ІТЕРАЦІЙ)
# ==========================
SEARCH_SPACE = {
    # reduction.*
    "reduction.umap_components": [3, 4, 5],
    "reduction.umap_n_neighbors": [30, 45, 60, 80],
    "reduction.umap_min_dist": [0.0],
    # clustering.*
    "clustering.min_cluster_size": [80, 100, 140, 180, 240, 320],
    "clustering.min_samples": [3, 5, 8, 12, 16],
    "clustering.cluster_selection_epsilon": [0.2, 0.4, 0.6, 0.9, 1.2, 1.6],
    "clustering.cluster_selection_method": ["leaf", "eom"],
    # purity.* (за потреби можна тюнити пороги теж)
    # Наприклад:
    # "purity.max_nn_cross_ratio": [0.4, 0.5, 0.6],
}

# Повний grid за замовчуванням: 3*4*1*6*5*6*2 = 4320 комбінацій.
# За замовчуванням беремо репрезентативну підмножину для "нічного" прогона.
DEFAULT_MAX_TRIALS = 240


def _build_iterations(max_trials: int) -> list[dict]:
    keys = list(SEARCH_SPACE.keys())
    values_grid = [SEARCH_SPACE[k] for k in keys]
    all_combos = []
    for idx, combo in enumerate(product(*values_grid), 1):
        flat = dict(zip(keys, combo))
        overrides: dict = {}
        for dotted_key, value in flat.items():
            section, param = dotted_key.split(".", 1)
            overrides.setdefault(section, {})
            overrides[section][param] = value

        # Формуємо стабільну коротку назву ітерації.
        tokens = []
        for dotted_key in keys:
            section, param = dotted_key.split(".", 1)
            value = flat[dotted_key]
            short = param.replace("cluster_selection_", "csel_").replace("_", "")
            val_s = str(value).replace(".", "_")
            tokens.append(f"{section[0]}{short}{val_s}")
        name = f"iter_{idx:04d}_" + "_".join(tokens[:6])

        all_combos.append(
            {
                "name": name,
                "overrides": overrides,
            }
        )

    if max_trials <= 0 or max_trials >= len(all_combos):
        return all_combos

    # Рівномірно розкладаємо вибірку по всьому простору параметрів.
    selected = []
    used = set()
    for i in range(max_trials):
        pos = round(i * (len(all_combos) - 1) / (max_trials - 1))
        if pos not in used:
            used.add(pos)
            selected.append(all_combos[pos])
    return selected


def _apply_section_overrides(base_cfg, overrides: dict):
    cfg = copy.deepcopy(base_cfg)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Невідомий параметр '{key}' для секції {type(cfg).__name__}")
        setattr(cfg, key, value)
    return cfg


def _score_for_sort(item: dict) -> tuple:
    """Менше значення -> кращий результат."""
    verdict_penalty = 0 if item["verdict"] == "PASS" else 1
    return (
        verdict_penalty,
        item["max_centroid_cosine"] if item["max_centroid_cosine"] is not None else 999.0,
        -(item["silhouette"] if item["silhouette"] is not None else -999.0),
        item["max_nn_cross_ratio"] if item["max_nn_cross_ratio"] is not None else 999.0,
    )


def _parse_max_trials(args: list[str], default: int) -> int:
    for i, arg in enumerate(args):
        if arg == "--max-trials" and i < len(args) - 1:
            return int(args[i + 1])
    return default


def main() -> None:
    config_path = BASE_CONFIG_PATH
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--config" and i < len(args) - 1:
            config_path = args[i + 1]
            break

    max_trials = _parse_max_trials(args, DEFAULT_MAX_TRIALS)
    iterations = _build_iterations(max_trials)

    cfg = load_config(config_path)

    print("=" * 72)
    print("  ITERATIVE TUNING: PTZ PRESET CLUSTERING")
    print("=" * 72)
    print(f"Config:  {config_path}")
    print(f"Input:   {cfg.input_dir}")
    print(f"Output:  {cfg.output_dir}")
    print(f"Trials:  {len(iterations)} (max_trials={max_trials})")

    print("\n[1/4] Сканування зображень...")
    image_paths = scan_images(cfg.input_dir, cfg.image_extensions)
    if not image_paths:
        raise RuntimeError("Зображення не знайдені. Перевір input_dir.")
    print(f"  Знайдено: {len(image_paths)}")

    print("\n[2/4] Витягування ознак (1 раз для всіх ітерацій)...")
    features = extract_features(image_paths, cfg.model, cfg.preprocessing, cfg.cache)
    print(f"  Features: {features.shape}")

    print("\n[3/4] Ітеративний прогін конфігів...")
    results = []
    total = len(iterations)
    start_all = time.time()
    for idx, it in enumerate(iterations, 1):
        t_iter = time.time()
        name = it["name"]
        print("\n" + "-" * 72)
        print(f"[{idx}/{total}] {name}")

        overrides = it.get("overrides", {})
        red_over = overrides.get("reduction", {})
        clu_over = overrides.get("clustering", {})
        pur_over = overrides.get("purity", {})

        rcfg = _apply_section_overrides(cfg.reduction, red_over)
        ccfg = _apply_section_overrides(cfg.clustering, clu_over)
        purity_cfg = _apply_section_overrides(cfg.purity, pur_over)

        print(f"  overrides: {overrides}")

        try:
            embeddings = reduce_dimensions(features, rcfg)
            labels = cluster(embeddings, ccfg)

            trial_output = cfg.output_dir / "_tuning" / name
            purity = evaluate_and_save_purity(
                features=features,
                embeddings=embeddings,
                labels=labels,
                output_dir=trial_output,
                viz_dir_name="viz",
                cfg=purity_cfg,
            )

            m = purity["metrics"]
            row = {
                "name": name,
                "verdict": purity["verdict"],
                "n_clusters": purity["run"]["n_clusters"],
                "max_centroid_cosine": m["max_centroid_cosine"],
                "silhouette": m["silhouette_mean"],
                "max_nn_cross_ratio": m["max_nn_cross_ratio"],
                "leakage_candidates": len(purity.get("leakage_candidates", [])),
                "report_path": str(trial_output / "viz" / "purity_report_latest.json"),
                "params": overrides,
                "status": "ok",
            }
        except Exception as exc:
            row = {
                "name": name,
                "verdict": "FAIL",
                "n_clusters": -1,
                "max_centroid_cosine": None,
                "silhouette": None,
                "max_nn_cross_ratio": None,
                "leakage_candidates": -1,
                "report_path": "",
                "params": overrides,
                "status": f"error: {type(exc).__name__}: {exc}",
            }

        results.append(row)

        elapsed_iter = time.time() - t_iter
        elapsed_all = time.time() - start_all
        avg_iter = elapsed_all / idx
        eta = avg_iter * (total - idx)
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)

        if row["status"] == "ok":
            print(
                "  RESULT:",
                f"verdict={row['verdict']},",
                f"clusters={row['n_clusters']},",
                f"max_cos={row['max_centroid_cosine']:.4f},",
                f"sil={row['silhouette'] if row['silhouette'] is not None else 'n/a'},",
                f"max_nn={row['max_nn_cross_ratio']:.4f},",
                f"leaks={row['leakage_candidates']},",
                f"time={elapsed_iter:.1f}s,",
                f"ETA~{eta_h:02d}:{eta_m:02d}",
            )
        else:
            print(
                "  RESULT:",
                "ERROR,",
                row["status"],
                f"time={elapsed_iter:.1f}s,",
                f"ETA~{eta_h:02d}:{eta_m:02d}",
            )

    print("\n[4/4] Підсумок")
    print("-" * 72)
    ok_results = [r for r in results if r["status"] == "ok"]
    if not ok_results:
        print("Не вдалося завершити жодної ітерації успішно.")
        return

    results_sorted = sorted(ok_results, key=_score_for_sort)
    for i, r in enumerate(results_sorted, 1):
        print(
            f"{i:>2}. {r['name']:<22} "
            f"{r['verdict']:<4} "
            f"clusters={r['n_clusters']:<4} "
            f"max_cos={r['max_centroid_cosine']:.4f} "
            f"sil={r['silhouette'] if r['silhouette'] is not None else 'n/a'} "
            f"nn={r['max_nn_cross_ratio']:.4f} "
            f"leaks={r['leakage_candidates']}"
        )

    best = results_sorted[0]
    print("-" * 72)
    print(f"BEST: {best['name']} | verdict={best['verdict']} | report={best['report_path']}")
    print(f"BEST reduction:  {best['params']['reduction']}")
    print(f"BEST clustering: {best['params']['clustering']}")

    summary_dir = cfg.output_dir / "_tuning"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "tuning_summary_latest.json"
    summary = {
        "config_path": config_path,
        "total_trials_requested": len(iterations),
        "total_trials_ok": len(ok_results),
        "best": best,
        "top20": results_sorted[:20],
        "all_results": results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
