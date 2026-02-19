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
import hashlib
import io
import json
import random
import time
from contextlib import redirect_stdout
from itertools import product
from pathlib import Path

from src.clusterer import cluster
from src.config import load_config
from src.feature_extractor import extract_features
from src.image_scanner import scan_images
from src.purity import evaluate_and_save_purity
from src.reducer import reduce_dimensions
from src.visualizer import visualize


# ==========================
# БАЗОВА КОНФІГУРАЦІЯ СКРИПТА
# ==========================
BASE_CONFIG_PATH = "config.yaml"

# Режим пошуку: "evolution" або "grid"
# evolution  -> генетичний пошук (рекомендовано для довгих запусків)
# grid       -> рівномірна вибірка з повного декартового простору SEARCH_SPACE
SEARCH_STRATEGY = "evolution"

# Бюджет запусків (максимум реальних expensive-eval)
# Це верхня межа кількості ПОВНИХ оцінок (reduce + cluster + purity) за запуск.
MAX_EVALUATIONS = 600

# Параметри еволюційного пошуку (μ + λ GA)
# EVOLUTION_POPULATION_SIZE  -> розмір популяції в кожному поколінні.
# EVOLUTION_ELITE_SIZE       -> скільки найкращих особин копіюємо без змін.
# EVOLUTION_TOURNAMENT_SIZE  -> розмір турніру при selection (більше = сильніший тиск відбору).
# EVOLUTION_MUTATION_RATE    -> ймовірність мутації кожного гена.
# EVOLUTION_RANDOM_SEED      -> фіксує відтворюваність генерації кандидатів.
EVOLUTION_POPULATION_SIZE = 12
EVOLUTION_ELITE_SIZE = 4
EVOLUTION_TOURNAMENT_SIZE = 3
EVOLUTION_MUTATION_RATE = 0.25
EVOLUTION_RANDOM_SEED = 42

# Early stop (плато): зупиняти, якщо немає покращення best score
# EARLY_STOP_PATIENCE_GENERATIONS -> скільки поколінь чекаємо покращення.
# EARLY_STOP_MIN_EVALUATIONS      -> мінімум оцінок, після якого дозволено ранню зупинку.
EARLY_STOP_ENABLED = True
EARLY_STOP_PATIENCE_GENERATIONS = 24
EARLY_STOP_MIN_EVALUATIONS = 64

# Двофазний режим:
#  - фаза 1: швидкий пошук на частині датасету
#  - фаза 2: уточнення топ-кандидатів на 100% даних
# PHASE1_FRACTION      -> частка датасету для фази 1 (0.5 = 50% кадрів).
# PHASE1_BUDGET_RATIO  -> частка бюджету MAX_EVALUATIONS для фази 1.
# PHASE2_TOP_K_SEEDS   -> скільки кращих кандидатів із фази 1 сідують у фазу 2.
TWO_PHASE_ENABLED = True
PHASE1_FRACTION = 0.5
PHASE1_BUDGET_RATIO = 0.6
PHASE2_TOP_K_SEEDS = 16

# Параметри grid fallback
# Використовується лише коли SEARCH_STRATEGY="grid".
GRID_MAX_TRIALS = MAX_EVALUATIONS

# Індекс файлів для прискорення старту.
# Якщо True і файл індексу існує, скрипт не робить повний scan кожен раз.
REUSE_IMAGE_INDEX = True
IMAGE_INDEX_FILENAME = "image_index.json"
FORCE_RESCAN = False
# Якщо True, перевіряє p.exists() для кожного шляху з індексу (може бути повільно на 100k+).
VERIFY_IMAGE_INDEX_PATHS = False

# Режим консолі:
# "full"    -> друк усіх внутрішніх логів reduce/cluster/purity.
# "compact" -> короткі RESULT-рядки; внутрішні логи приглушені.
CONSOLE_VERBOSITY = "compact"
SHOW_OVERRIDES_EACH_ITER = False
SAVE_COMPACT_STDOUT_PER_ITER = False
# Перші N ітерацій друкуються детально навіть у compact-режимі.
DETAILED_FIRST_N_ITERS = 2

# Додатковий post-step після тюнінгу:
# для топ-N кандидатів створити повний пакет візуалізацій як у main.py.
GENERATE_FULL_VISUALS_FOR_TOP_N = 10

# ==========================
# GRID SEARCH (БАГАТО ІТЕРАЦІЙ)
# ==========================
SEARCH_SPACE = {
    # reduction.*
    # Ключі мають формат "section.param" і АВТОМАТИЧНО мапляться в dataclass секції конфіга.
    # Тобто можна додавати будь-який валідний параметр з config-схеми:
    #   "reduction.*", "clustering.*", "purity.*" (і за потреби інші секції, якщо підтримає оцінка).
    "reduction.umap_components": [3, 4, 5],
    "reduction.umap_n_neighbors": [30, 45, 60, 80],
    "reduction.umap_min_dist": [0.0],
    # clustering.*
    "clustering.min_cluster_size": [300, 400, 500],
    "clustering.min_samples": [3, 5, 8, 12, 16],
    "clustering.cluster_selection_epsilon": [0.2, 0.4, 0.6, 0.9, 1.2, 1.6],
    "clustering.cluster_selection_method": ["leaf", "eom"],
    # purity.* (за потреби можна тюнити пороги теж)
    # Наприклад:
    # "purity.max_nn_cross_ratio": [0.4, 0.5, 0.6],
}

# Повний grid за замовчуванням: 3*4*1*6*5*6*2 = 4320 комбінацій.
# За замовчуванням беремо репрезентативну підмножину для "нічного" прогона.
DEFAULT_MAX_TRIALS = GRID_MAX_TRIALS


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


def _space_keys() -> list[str]:
    return list(SEARCH_SPACE.keys())


def _flat_to_overrides(flat: dict) -> dict:
    overrides: dict = {}
    for dotted_key, value in flat.items():
        section, param = dotted_key.split(".", 1)
        overrides.setdefault(section, {})
        overrides[section][param] = value
    return overrides


def _candidate_signature(flat: dict, keys: list[str]) -> tuple:
    return tuple(flat[k] for k in keys)


def _candidate_name(prefix: str, idx: int, flat: dict, keys: list[str]) -> str:
    tokens = []
    for dotted_key in keys:
        section, param = dotted_key.split(".", 1)
        value = flat[dotted_key]
        short = param.replace("cluster_selection_", "csel_").replace("_", "")
        val_s = str(value).replace(".", "_")
        tokens.append(f"{section[0]}{short}{val_s}")
    return f"{prefix}_{idx:04d}_" + "_".join(tokens[:6])


def _random_flat_candidate(rng: random.Random, keys: list[str]) -> dict:
    return {k: rng.choice(SEARCH_SPACE[k]) for k in keys}


def _tournament_pick(rng: random.Random, pool: list[dict], k: int) -> dict:
    subset = rng.sample(pool, min(k, len(pool)))
    return min(subset, key=_score_for_sort)


def _crossover(rng: random.Random, p1_flat: dict, p2_flat: dict, keys: list[str]) -> dict:
    child = {}
    for k in keys:
        child[k] = p1_flat[k] if rng.random() < 0.5 else p2_flat[k]
    return child


def _mutate(rng: random.Random, flat: dict, keys: list[str], mutation_rate: float) -> dict:
    child = dict(flat)
    for k in keys:
        if rng.random() < mutation_rate:
            choices = SEARCH_SPACE[k]
            if len(choices) > 1:
                current = child[k]
                new_val = rng.choice(choices)
                while new_val == current:
                    new_val = rng.choice(choices)
                child[k] = new_val
    return child


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


def _subsample_features(features, fraction: float, seed: int):
    if fraction >= 1.0:
        return features, None
    if fraction <= 0.0:
        raise ValueError("PHASE1_FRACTION має бути > 0")
    n = len(features)
    n_sub = max(1, int(n * fraction))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(n), n_sub))
    return features[indices], indices


def _load_or_build_image_index(cfg, output_root: Path):
    input_key = hashlib.sha256(str(cfg.input_dir).encode("utf-8")).hexdigest()[:12]
    idx_path = output_root / f"{IMAGE_INDEX_FILENAME.replace('.json', '')}_{input_key}.json"
    if REUSE_IMAGE_INDEX and not FORCE_RESCAN and idx_path.exists():
        try:
            data = json.loads(idx_path.read_text(encoding="utf-8"))
            paths = [Path(p) for p in data.get("paths", [])]
            if VERIFY_IMAGE_INDEX_PATHS:
                paths = [p for p in paths if p.exists()]
            if paths:
                print(f"  Індекс завантажено: {idx_path} ({len(paths)} файлів)")
                return paths
            print("  [WARN] Індекс порожній/невалідний, виконуємо повне сканування...")
        except Exception:
            print("  [WARN] Не вдалося прочитати індекс, виконуємо повне сканування...")

    paths = scan_images(cfg.input_dir, cfg.image_extensions)
    if REUSE_IMAGE_INDEX:
        payload = {
            "input_dir": str(cfg.input_dir),
            "n_paths": len(paths),
            "paths": [str(p) for p in paths],
        }
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"  Індекс збережено: {idx_path}")
    return paths


def _evaluate_iteration(
    cfg,
    features,
    flat_candidate: dict,
    name: str,
    force_full_logs: bool = False,
):
    overrides = _flat_to_overrides(flat_candidate)
    red_over = overrides.get("reduction", {})
    clu_over = overrides.get("clustering", {})
    pur_over = overrides.get("purity", {})

    rcfg = _apply_section_overrides(cfg.reduction, red_over)
    ccfg = _apply_section_overrides(cfg.clustering, clu_over)
    purity_cfg = _apply_section_overrides(cfg.purity, pur_over)

    trial_output = cfg.output_dir / "_tuning" / name
    captured_stdout = ""
    use_compact = (CONSOLE_VERBOSITY == "compact") and (not force_full_logs)
    if use_compact:
        buf = io.StringIO()
        with redirect_stdout(buf):
            embeddings = reduce_dimensions(features, rcfg, cfg.acceleration)
            labels = cluster(embeddings, ccfg, cfg.acceleration)
            purity = evaluate_and_save_purity(
                features=features,
                embeddings=embeddings,
                labels=labels,
                output_dir=trial_output,
                viz_dir_name="viz",
                cfg=purity_cfg,
                accel_cfg=cfg.acceleration,
            )
        captured_stdout = buf.getvalue()
        if SAVE_COMPACT_STDOUT_PER_ITER and captured_stdout.strip():
            log_path = trial_output / "iteration_stdout.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(captured_stdout, encoding="utf-8")
    else:
        embeddings = reduce_dimensions(features, rcfg, cfg.acceleration)
        labels = cluster(embeddings, ccfg, cfg.acceleration)
        purity = evaluate_and_save_purity(
            features=features,
            embeddings=embeddings,
            labels=labels,
            output_dir=trial_output,
            viz_dir_name="viz",
            cfg=purity_cfg,
            accel_cfg=cfg.acceleration,
        )

    m = purity["metrics"]
    warn_count = 0
    if captured_stdout:
        warn_count = sum(1 for ln in captured_stdout.splitlines() if "[WARN]" in ln)
    return {
        "name": name,
        "verdict": purity["verdict"],
        "n_clusters": purity["run"]["n_clusters"],
        "max_centroid_cosine": m["max_centroid_cosine"],
        "silhouette": m["silhouette_mean"],
        "max_nn_cross_ratio": m["max_nn_cross_ratio"],
        "leakage_candidates": len(purity.get("leakage_candidates", [])),
        "report_path": str(trial_output / "viz" / "purity_report_latest.json"),
        "params": overrides,
        "flat_params": dict(flat_candidate),
        "warn_count": warn_count,
        "status": "ok",
    }


def _save_tuning_plots(summary_dir: Path, results: list[dict], top_results: list[dict]) -> list[str]:
    """Save tuning progress plots; return created file paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created = []
    ok_results = [r for r in results if r.get("status") == "ok"]
    if not ok_results:
        return created

    # 1) Progress plot over evaluation index.
    eval_x = list(range(1, len(ok_results) + 1))
    max_cos = [r["max_centroid_cosine"] for r in ok_results]
    silhouette = [r["silhouette"] for r in ok_results]
    max_nn = [r["max_nn_cross_ratio"] for r in ok_results]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(eval_x, max_cos, marker="o", markersize=2, linewidth=1)
    axes[0].set_ylabel("max_centroid_cosine")
    axes[0].grid(alpha=0.3)

    axes[1].plot(eval_x, silhouette, marker="o", markersize=2, linewidth=1, color="tab:orange")
    axes[1].set_ylabel("silhouette")
    axes[1].grid(alpha=0.3)

    axes[2].plot(eval_x, max_nn, marker="o", markersize=2, linewidth=1, color="tab:green")
    axes[2].set_ylabel("max_nn_cross_ratio")
    axes[2].set_xlabel("Successful evaluation #")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Tuning Progress")
    fig.tight_layout()
    progress_path = summary_dir / "tuning_progress.png"
    fig.savefig(progress_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    created.append(str(progress_path))

    # 2) Top-20 max_cos + silhouettes.
    top_n = top_results[:20]
    names = [r["name"][-24:] for r in top_n]
    cos_vals = [r["max_centroid_cosine"] for r in top_n]
    sil_vals = [r["silhouette"] if r["silhouette"] is not None else 0.0 for r in top_n]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    bars = ax1.bar(range(len(top_n)), cos_vals, color="tab:red", alpha=0.75)
    ax1.set_ylabel("max_centroid_cosine", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_xticks(range(len(top_n)))
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(range(len(top_n)), sil_vals, color="tab:blue", marker="o", linewidth=1.3)
    ax2.set_ylabel("silhouette", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    for b, v in zip(bars, cos_vals):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            v + 0.002,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.title("Top-20 Candidates: max_centroid_cosine vs silhouette")
    fig.tight_layout()
    top_path = summary_dir / "top20_metrics.png"
    fig.savefig(top_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    created.append(str(top_path))

    return created


def _generate_full_visualizations_for_top_n(
    cfg,
    image_paths,
    features,
    results_sorted: list[dict],
    top_n: int,
) -> list[dict]:
    """Generate main-like visualization artifacts for top-N candidates on full data."""
    if top_n <= 0:
        return []

    top_rows = results_sorted[:top_n]
    generated = []
    total = len(top_rows)

    print("\n[5/5] Повна візуалізація для TOP-кандидатів...")
    for rank, row in enumerate(top_rows, 1):
        name = row["name"]
        print("\n" + "-" * 72)
        print(f"[{rank}/{total}] TOP-{rank}: {name}")
        print("  Повторний full-run (reduce + cluster + purity + visualize)...")

        flat = dict(row.get("flat_params", {}))
        overrides = _flat_to_overrides(flat)
        red_over = overrides.get("reduction", {})
        clu_over = overrides.get("clustering", {})
        pur_over = overrides.get("purity", {})

        rcfg = _apply_section_overrides(cfg.reduction, red_over)
        ccfg = _apply_section_overrides(cfg.clustering, clu_over)
        purity_cfg = _apply_section_overrides(cfg.purity, pur_over)

        run_dir = cfg.output_dir / "_tuning" / "_top_visuals" / f"top_{rank:02d}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        embeddings = reduce_dimensions(features, rcfg, cfg.acceleration)
        labels = cluster(embeddings, ccfg, cfg.acceleration)
        purity = evaluate_and_save_purity(
            features=features,
            embeddings=embeddings,
            labels=labels,
            output_dir=run_dir,
            viz_dir_name=cfg.visualization.viz_dir,
            cfg=purity_cfg,
            accel_cfg=cfg.acceleration,
        )
        if cfg.visualization.enabled:
            visualize(
                image_paths=image_paths,
                embeddings=embeddings,
                labels=labels,
                output_dir=run_dir,
                cfg=cfg.visualization,
                preproc_cfg=cfg.preprocessing,
            )
        elapsed = time.time() - t0

        viz_dir = run_dir / cfg.visualization.viz_dir
        generated.append(
            {
                "rank": rank,
                "name": name,
                "verdict": purity["verdict"],
                "output_dir": str(run_dir),
                "viz_dir": str(viz_dir),
                "purity_report": str(viz_dir / "purity_report_latest.json"),
                "interactive_scatter": str(viz_dir / "interactive_scatter.html"),
                "cluster_sizes": str(viz_dir / "cluster_sizes.png"),
                "cluster_grid": str(viz_dir / "cluster_grid.png"),
                "preprocessing_comparison": str(viz_dir / "preprocessing_comparison.png"),
                "elapsed_sec": round(elapsed, 2),
            }
        )
        print(f"  Done: {run_dir} ({elapsed:.1f}s)")

    return generated


def _run_evolution_search(
    cfg,
    keys: list[str],
    features,
    max_trials: int,
    phase_tag: str,
    rng_seed: int,
    initial_population: list[dict] | None = None,
):
    rng = random.Random(rng_seed)
    pop_size = EVOLUTION_POPULATION_SIZE
    elite_size = min(EVOLUTION_ELITE_SIZE, pop_size)
    tournament_size = EVOLUTION_TOURNAMENT_SIZE
    mutation_rate = EVOLUTION_MUTATION_RATE

    results = []
    evaluated_signatures = set()
    start_all = time.time()

    def log_result(idx: int, total: int, row: dict, elapsed_iter: float):
        elapsed_all = time.time() - start_all
        avg_iter = elapsed_all / max(1, idx)
        eta = avg_iter * max(0, total - idx)
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
                f"warns={row.get('warn_count', 0)},",
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

    population = []
    if initial_population:
        for cand in initial_population:
            sig = _candidate_signature(cand, keys)
            if sig in evaluated_signatures:
                continue
            evaluated_signatures.add(sig)
            population.append(dict(cand))
            if len(population) >= pop_size:
                break

    while len(population) < pop_size:
        cand = _random_flat_candidate(rng, keys)
        sig = _candidate_signature(cand, keys)
        if sig in evaluated_signatures:
            continue
        evaluated_signatures.add(sig)
        population.append(cand)

    eval_count = 0
    generation = 0
    stagnation_generations = 0
    best_score = None

    while eval_count < max_trials:
        generation += 1
        print("\n" + "=" * 72)
        print(f"{phase_tag} | GENERATION {generation}")
        print("=" * 72)

        gen_rows = []
        for cand in population:
            if eval_count >= max_trials:
                break
            eval_count += 1
            t_iter = time.time()
            name = _candidate_name(f"evo_{phase_tag.lower()}", eval_count, cand, keys)
            print("\n" + "-" * 72)
            print(f"[{eval_count}/{max_trials}] {name}")
            detailed_now = eval_count <= DETAILED_FIRST_N_ITERS
            if detailed_now and CONSOLE_VERBOSITY == "compact":
                print("  [detail] expanded logs enabled for warmup iteration")
            if SHOW_OVERRIDES_EACH_ITER or detailed_now:
                print(f"  overrides: {_flat_to_overrides(cand)}")

            try:
                row = _evaluate_iteration(
                    cfg,
                    features,
                    cand,
                    name,
                    force_full_logs=detailed_now,
                )
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
                    "params": _flat_to_overrides(cand),
                    "flat_params": dict(cand),
                    "status": f"error: {type(exc).__name__}: {exc}",
                }

            results.append(row)
            gen_rows.append(row)
            log_result(eval_count, max_trials, row, time.time() - t_iter)

        ok_pool = [r for r in gen_rows if r["status"] == "ok"]
        improved = False
        if ok_pool:
            gen_best = min(ok_pool, key=_score_for_sort)
            gen_best_score = _score_for_sort(gen_best)
            if best_score is None or gen_best_score < best_score:
                best_score = gen_best_score
                improved = True

        if improved:
            stagnation_generations = 0
        else:
            stagnation_generations += 1

        if (
            EARLY_STOP_ENABLED
            and eval_count >= EARLY_STOP_MIN_EVALUATIONS
            and stagnation_generations >= EARLY_STOP_PATIENCE_GENERATIONS
        ):
            print(
                f"\n{phase_tag}: early stop — "
                f"немає покращення {EARLY_STOP_PATIENCE_GENERATIONS} поколінь підряд."
            )
            break

        if eval_count >= max_trials:
            break

        if not ok_pool:
            population = []
            while len(population) < pop_size:
                cand = _random_flat_candidate(rng, keys)
                sig = _candidate_signature(cand, keys)
                if sig in evaluated_signatures:
                    continue
                evaluated_signatures.add(sig)
                population.append(cand)
            continue

        ranked = sorted(ok_pool, key=_score_for_sort)
        elites = ranked[:elite_size]
        elite_flats = [e["flat_params"] for e in elites]

        next_population = [dict(f) for f in elite_flats]
        source_pool = ranked[: max(elite_size * 2, 4)]

        attempts = 0
        while len(next_population) < pop_size and attempts < 2000:
            attempts += 1
            p1 = _tournament_pick(rng, source_pool, tournament_size)["flat_params"]
            p2 = _tournament_pick(rng, source_pool, tournament_size)["flat_params"]
            child = _crossover(rng, p1, p2, keys)
            child = _mutate(rng, child, keys, mutation_rate)
            sig = _candidate_signature(child, keys)
            if sig in evaluated_signatures:
                continue
            evaluated_signatures.add(sig)
            next_population.append(child)

        while len(next_population) < pop_size and len(evaluated_signatures) < 100000:
            child = _random_flat_candidate(rng, keys)
            sig = _candidate_signature(child, keys)
            if sig in evaluated_signatures:
                continue
            evaluated_signatures.add(sig)
            next_population.append(child)

        population = next_population

    return results


def main() -> None:
    config_path = BASE_CONFIG_PATH
    max_trials = MAX_EVALUATIONS
    strategy = SEARCH_STRATEGY.lower().strip()

    cfg = load_config(config_path)
    keys = _space_keys()

    print("=" * 72)
    print("  ITERATIVE TUNING: PTZ PRESET CLUSTERING")
    print("=" * 72)
    print(f"Config:  {config_path}")
    print(f"Input:   {cfg.input_dir}")
    print(f"Output:  {cfg.output_dir}")
    print(f"Strategy: {strategy}")
    print(f"Max evaluations: {max_trials}")
    print(f"Two-phase: {TWO_PHASE_ENABLED} (phase1_fraction={PHASE1_FRACTION})")
    print(f"Early stop: {EARLY_STOP_ENABLED}")
    print(f"Console verbosity: {CONSOLE_VERBOSITY}")
    print(f"Top full visualizations: {GENERATE_FULL_VISUALS_FOR_TOP_N}")

    print("\n[1/4] Сканування зображень...")
    print(f"  Requested acceleration backend: {cfg.acceleration.backend}")
    image_paths = _load_or_build_image_index(cfg, Path(cfg.cache.cache_dir))
    if not image_paths:
        raise RuntimeError("Зображення не знайдені. Перевір input_dir.")
    print(f"  Знайдено: {len(image_paths)}")

    print("\n[2/4] Витягування ознак (1 раз для всіх ітерацій)...")
    features = extract_features(image_paths, cfg.model, cfg.preprocessing, cfg.cache)
    print(f"  Features: {features.shape}")

    print("\n[3/4] Ітеративний прогін конфігів...")
    results = []

    if strategy == "grid":
        evaluated_signatures = set()
        start_all = time.time()

        def log_result(idx: int, total: int, row: dict, elapsed_iter: float):
            elapsed_all = time.time() - start_all
            avg_iter = elapsed_all / max(1, idx)
            eta = avg_iter * max(0, total - idx)
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
                    f"warns={row.get('warn_count', 0)},",
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

        iterations = _build_iterations(DEFAULT_MAX_TRIALS)
        total = min(len(iterations), max_trials)
        for idx, it in enumerate(iterations[:total], 1):
            t_iter = time.time()
            name = it["name"]
            flat = {}
            for section, params in it.get("overrides", {}).items():
                for key, value in params.items():
                    flat[f"{section}.{key}"] = value
            sig = _candidate_signature(flat, keys)
            if sig in evaluated_signatures:
                continue
            evaluated_signatures.add(sig)

            print("\n" + "-" * 72)
            print(f"[{idx}/{total}] {name}")
            detailed_now = idx <= DETAILED_FIRST_N_ITERS
            if detailed_now and CONSOLE_VERBOSITY == "compact":
                print("  [detail] expanded logs enabled for warmup iteration")
            if SHOW_OVERRIDES_EACH_ITER or detailed_now:
                print(f"  overrides: {it.get('overrides', {})}")
            try:
                row = _evaluate_iteration(
                    cfg,
                    features,
                    flat,
                    name,
                    force_full_logs=detailed_now,
                )
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
                    "params": it.get("overrides", {}),
                    "flat_params": flat,
                    "status": f"error: {type(exc).__name__}: {exc}",
                }
            results.append(row)
            log_result(idx, total, row, time.time() - t_iter)
    elif strategy == "evolution":
        if TWO_PHASE_ENABLED:
            phase1_budget = max(1, int(max_trials * PHASE1_BUDGET_RATIO))
            phase2_budget = max_trials - phase1_budget

            print("\n" + "#" * 72)
            print(f"PHASE 1: subset {PHASE1_FRACTION*100:.1f}% | budget={phase1_budget}")
            print("#" * 72)
            features_phase1, _ = _subsample_features(
                features, PHASE1_FRACTION, EVOLUTION_RANDOM_SEED
            )
            phase1_results = _run_evolution_search(
                cfg=cfg,
                keys=keys,
                features=features_phase1,
                max_trials=phase1_budget,
                phase_tag="PHASE1",
                rng_seed=EVOLUTION_RANDOM_SEED,
            )
            results.extend(phase1_results)

            phase1_ok = [r for r in phase1_results if r["status"] == "ok"]
            seeds = []
            if phase1_ok:
                phase1_sorted = sorted(phase1_ok, key=_score_for_sort)
                for row in phase1_sorted[:PHASE2_TOP_K_SEEDS]:
                    seeds.append(dict(row["flat_params"]))

            if phase2_budget > 0:
                print("\n" + "#" * 72)
                print(f"PHASE 2: full dataset 100% | budget={phase2_budget}")
                print("#" * 72)
                phase2_results = _run_evolution_search(
                    cfg=cfg,
                    keys=keys,
                    features=features,
                    max_trials=phase2_budget,
                    phase_tag="PHASE2",
                    rng_seed=EVOLUTION_RANDOM_SEED + 1,
                    initial_population=seeds,
                )
                results.extend(phase2_results)
        else:
            phase_results = _run_evolution_search(
                cfg=cfg,
                keys=keys,
                features=features,
                max_trials=max_trials,
                phase_tag="PHASE1",
                rng_seed=EVOLUTION_RANDOM_SEED,
            )
            results.extend(phase_results)
    else:
        raise ValueError("SEARCH_STRATEGY має бути 'grid' або 'evolution'")

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
        "strategy": strategy,
        "total_trials_requested": max_trials,
        "total_trials_ok": len(ok_results),
        "best": best,
        "top20": results_sorted[:20],
        "all_results": results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")

    try:
        plot_paths = _save_tuning_plots(summary_dir, results, results_sorted)
        if plot_paths:
            print("Plots:")
            for p in plot_paths:
                print(f"  {p}")
            summary["plots"] = plot_paths
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Plot generation warning: {type(exc).__name__}: {exc}")

    full_visual_runs = []
    if GENERATE_FULL_VISUALS_FOR_TOP_N > 0:
        try:
            full_visual_runs = _generate_full_visualizations_for_top_n(
                cfg=cfg,
                image_paths=image_paths,
                features=features,
                results_sorted=results_sorted,
                top_n=GENERATE_FULL_VISUALS_FOR_TOP_N,
            )
            if full_visual_runs:
                summary["top_full_visualizations"] = full_visual_runs
                summary_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception as exc:
            print(f"Top visualization warning: {type(exc).__name__}: {exc}")

    print("=" * 72)


if __name__ == "__main__":
    main()
