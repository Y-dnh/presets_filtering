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
import csv
import hashlib
import io
import json
import shutil
import random
import time
from contextlib import redirect_stdout
from pathlib import Path

from src.clusterer import cluster
from src.config import load_config
from src.feature_extractor import extract_features, get_last_cache_info
from src.image_scanner import scan_images
from src.purity import evaluate_and_save_purity
from src.reducer import reduce_dimensions
from src.visualizer import visualize


# ==========================
# БАЗОВА КОНФІГУРАЦІЯ СКРИПТА
# ==========================
BASE_CONFIG_PATH = "config.yaml"

# Режим пошуку: тільки еволюційний (генетичний алгоритм).
SEARCH_STRATEGY = "evolution"

# Бюджет запусків (максимум реальних expensive-eval)
# Це верхня межа кількості ПОВНИХ оцінок (reduce + cluster + purity) за запуск.
MAX_EVALUATIONS = 24

# Параметри еволюційного пошуку (μ + λ GA)
# EVOLUTION_POPULATION_SIZE  -> розмір популяції в кожному поколінні.
# EVOLUTION_ELITE_SIZE       -> скільки найкращих особин копіюємо без змін.
# EVOLUTION_TOURNAMENT_SIZE  -> розмір турніру при selection (більше = сильніший тиск відбору).
# EVOLUTION_MUTATION_RATE    -> ймовірність мутації кожного гена.
# EVOLUTION_RANDOM_SEED      -> фіксує відтворюваність генерації кандидатів.
EVOLUTION_POPULATION_SIZE = 24
EVOLUTION_ELITE_SIZE = 6
EVOLUTION_TOURNAMENT_SIZE = 4
EVOLUTION_MUTATION_RATE = 0.22
EVOLUTION_RANDOM_SEED = 42

# Early stop (плато): зупиняти, якщо немає покращення best score
# EARLY_STOP_PATIENCE_GENERATIONS -> скільки поколінь чекаємо покращення.
# EARLY_STOP_MIN_EVALUATIONS      -> мінімум оцінок, після якого дозволено ранню зупинку.
EARLY_STOP_ENABLED = True
EARLY_STOP_PATIENCE_GENERATIONS = 60
EARLY_STOP_MIN_EVALUATIONS = 2000

# Двофазний режим:
#  - старт еволюції на частині датасету
#  - автоматичний свіч на 100% при плато/повільному покращенні
# PHASE1_FRACTION      -> частка датасету для фази 1 (0.5 = 50% кадрів).
# SWITCH_PATIENCE_GENERATIONS -> скільки поколінь без суттєвого прогресу до свічу.
# SWITCH_MIN_EVALUATIONS      -> мінімум eval перед дозволом свічу.
# SWITCH_MIN_REL_IMPROVEMENT  -> мін. відносне покращення best score за вікно; менше => "повільно".
# SWITCH_REEVAL_ELITES_TOP_K  -> скільки еліт переоцінити на full одразу після свічу.
TWO_PHASE_ENABLED = True
PHASE1_FRACTION = 0.5
SWITCH_PATIENCE_GENERATIONS = 20
SWITCH_MIN_EVALUATIONS = 240
SWITCH_MIN_REL_IMPROVEMENT = 0.003
SWITCH_REEVAL_ELITES_TOP_K = 12

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

# Короткі назви директорій ітерацій:
# True  -> evo_phase2_0076
# False -> evo_phase2_0076_...довгі_токени...
USE_COMPACT_ITERATION_NAMES = True

# Додатковий post-step після тюнінгу:
# для топ-N кандидатів створити повний пакет візуалізацій як у main.py.
GENERATE_FULL_VISUALS_FOR_TOP_N = 2

# Куди складати purity-артефакти під час ітеративного пошуку.
# Важливо: це окрема папка, щоб `viz/` містив тільки візуалізації.
PURITY_ARTIFACTS_DIR = "metrics"
TUNING_DIR_BASENAME = "_tuning"

# Ваги композитного score (менше = краще):
# score = base + penalty
# де:
#   base = w_cos * max_centroid_cosine
#        + w_nn * max_nn_cross_ratio
#        - w_sil * silhouette
#   penalty = w_cos_excess * max(0, max_centroid_cosine - threshold_max_cos)
#           + w_nn_excess * max(0, max_nn_cross_ratio - threshold_max_nn)
#           + w_sil_deficit * max(0, threshold_min_silhouette - silhouette)
#
# ==============================================================================
# КОМПОЗИТНИЙ SCORE (ФІТНЕС-ФУНКЦІЯ ЕВОЛЮЦІЇ)
# ==============================================================================
# Мета генетичного алгоритму — ЗМІНІМІЗУВАТИ цей score. Чим менше значення, тим краще.
# Формула: Score = (Базова оцінка якості) + (Штрафи за порушення порогів Purity)
# 
# Чому використовуються "м'які штрафи" (Soft Penalties), а не жорстке відсікання?
# Якщо особина має max_cos = 0.966 (ледь порушила поріг 0.965), але при цьому 
# ідеальний silhouette (0.6), ми не хочемо її вбивати. Її "гени" (параметри) дуже
# цінні для популяції. Штраф просто опустить її в рейтингу, але залишить для мутацій.
# ==============================================================================

# --- БАЗОВІ ВАГИ (Вплив на рейтинг у межах норми) ---

# SCORE_WEIGHT_MAX_COS: Найвищий пріоритет (1.15). 
# Косинусна подібність центроїдів напряму відповідає за ризик витоку даних (Data Leakage). 
# Чим нижчий max_cos, тим далі пресети один від одного семантично. 
# Алгоритм має фокусуватися на мінімізації саме цього показника.
SCORE_WEIGHT_MAX_COS = 1.15

# SCORE_WEIGHT_MAX_NN: Середній пріоритет (0.75). 
# Оцінює "шорсткість" кордонів між кластерами. Якщо точки одного пресету 
# фізично знаходяться поруч із точками іншого пресету в просторі UMAP, це погано, 
# але менш критично, ніж глобальна схожість центроїдів.
SCORE_WEIGHT_MAX_NN = 0.75

# SCORE_WEIGHT_SILHOUETTE: Найнижчий пріоритет (0.25). 
# Silhouette оцінює математичну щільність "кульок" кластерів. 
# Ми ставимо низьку вагу, бо для нас ізоляція пресетів набагато важливіша 
# за красиву математичну форму кластера. 
# (Примітка: оскільки більший silhouette — це краще, у формулі він віднімається).
SCORE_WEIGHT_SILHOUETTE = 0.25


# --- ВАГИ ШТРАФІВ (Вмикаються тільки при порушенні порогів з config.yaml) ---

# SCORE_WEIGHT_COS_EXCESS: Дуже жорсткий штраф (2.2).
# Якщо косинусна подібність пробиває поріг (напр. > 0.965), це означає, що 
# алгоритм створив відверті дублікати локацій. Цей множник стрімко запускає 
# score в космос, щоб еволюція "тікала" від таких параметрів.
SCORE_WEIGHT_COS_EXCESS = 2.2

# SCORE_WEIGHT_NN_EXCESS: Помірний штраф (1.5).
# Якщо перетік точок (kNN cross-ratio) перевищує ліміт, ми штрафуємо конфігурацію, 
# бо вона, ймовірно, неправильно розрізала один великий пресет на шматки.
SCORE_WEIGHT_NN_EXCESS = 1.5

# SCORE_WEIGHT_SIL_DEFICIT: М'який штраф (1.0).
# Якщо кластери вийшли занадто пухкими (silhouette < 0.25), ми накладаємо 
# легкий штраф. Це не так критично, як дублікати локацій, але свідчить 
# про високий рівень шуму.
SCORE_WEIGHT_SIL_DEFICIT = 1.0

# ==========================
# SEARCH SPACE (ЕВОЛЮЦІЙНИЙ ПОШУК)
# ==========================
SEARCH_SPACE = {
    # reduction.*
    # Ключі мають формат "section.param" і АВТОМАТИЧНО мапляться в dataclass секції конфіга.
    # Тобто можна додавати будь-який валідний параметр з config-схеми:
    #   "reduction.*", "clustering.*", "purity.*" (і за потреби інші секції, якщо підтримає оцінка).

    # pca.*
    "reduction.pca_components": [50, 100, 150, 250, 400],

    # umap.*
    
    "reduction.umap_components": [3, 4, 5, 6, 8, 10],
    "reduction.umap_n_neighbors": [15, 30, 45, 60, 80, 100, 120, 150, 200],    
    "reduction.umap_min_dist": [0.0],

    # clustering.*
    "clustering.min_cluster_size": [300, 450, 600, 800, 1000],
    "clustering.min_samples": [3, 5, 8, 12, 16, 24],
    "clustering.cluster_selection_epsilon": [0.3, 0.6, 0.9, 1.2, 1.6, 2.0],
    "clustering.cluster_selection_method": ["eom"],

    # purity.* (за потреби можна тюнити пороги теж)
    # Наприклад:
    # "purity.max_nn_cross_ratio": [0.4, 0.5, 0.6],
}

def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: _json_ready(v) for k, v in vars(value).items()}
    return value


def _next_tuning_run_root(output_dir: Path) -> Path:
    base = output_dir / TUNING_DIR_BASENAME
    if not base.exists():
        return base
    idx = 2
    while True:
        candidate = output_dir / f"{TUNING_DIR_BASENAME}{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def _tuner_settings_snapshot() -> dict:
    return {
        "BASE_CONFIG_PATH": BASE_CONFIG_PATH,
        "SEARCH_STRATEGY": SEARCH_STRATEGY,
        "MAX_EVALUATIONS": MAX_EVALUATIONS,
        "EVOLUTION_POPULATION_SIZE": EVOLUTION_POPULATION_SIZE,
        "EVOLUTION_ELITE_SIZE": EVOLUTION_ELITE_SIZE,
        "EVOLUTION_TOURNAMENT_SIZE": EVOLUTION_TOURNAMENT_SIZE,
        "EVOLUTION_MUTATION_RATE": EVOLUTION_MUTATION_RATE,
        "EVOLUTION_RANDOM_SEED": EVOLUTION_RANDOM_SEED,
        "EARLY_STOP_ENABLED": EARLY_STOP_ENABLED,
        "EARLY_STOP_PATIENCE_GENERATIONS": EARLY_STOP_PATIENCE_GENERATIONS,
        "EARLY_STOP_MIN_EVALUATIONS": EARLY_STOP_MIN_EVALUATIONS,
        "TWO_PHASE_ENABLED": TWO_PHASE_ENABLED,
        "PHASE1_FRACTION": PHASE1_FRACTION,
        "SWITCH_PATIENCE_GENERATIONS": SWITCH_PATIENCE_GENERATIONS,
        "SWITCH_MIN_EVALUATIONS": SWITCH_MIN_EVALUATIONS,
        "SWITCH_MIN_REL_IMPROVEMENT": SWITCH_MIN_REL_IMPROVEMENT,
        "SWITCH_REEVAL_ELITES_TOP_K": SWITCH_REEVAL_ELITES_TOP_K,
        "REUSE_IMAGE_INDEX": REUSE_IMAGE_INDEX,
        "IMAGE_INDEX_FILENAME": IMAGE_INDEX_FILENAME,
        "FORCE_RESCAN": FORCE_RESCAN,
        "VERIFY_IMAGE_INDEX_PATHS": VERIFY_IMAGE_INDEX_PATHS,
        "CONSOLE_VERBOSITY": CONSOLE_VERBOSITY,
        "SHOW_OVERRIDES_EACH_ITER": SHOW_OVERRIDES_EACH_ITER,
        "SAVE_COMPACT_STDOUT_PER_ITER": SAVE_COMPACT_STDOUT_PER_ITER,
        "DETAILED_FIRST_N_ITERS": DETAILED_FIRST_N_ITERS,
        "USE_COMPACT_ITERATION_NAMES": USE_COMPACT_ITERATION_NAMES,
        "GENERATE_FULL_VISUALS_FOR_TOP_N": GENERATE_FULL_VISUALS_FOR_TOP_N,
        "PURITY_ARTIFACTS_DIR": PURITY_ARTIFACTS_DIR,
        "SCORE_WEIGHTS": {
            "SCORE_WEIGHT_MAX_COS": SCORE_WEIGHT_MAX_COS,
            "SCORE_WEIGHT_MAX_NN": SCORE_WEIGHT_MAX_NN,
            "SCORE_WEIGHT_SILHOUETTE": SCORE_WEIGHT_SILHOUETTE,
            "SCORE_WEIGHT_COS_EXCESS": SCORE_WEIGHT_COS_EXCESS,
            "SCORE_WEIGHT_NN_EXCESS": SCORE_WEIGHT_NN_EXCESS,
            "SCORE_WEIGHT_SIL_DEFICIT": SCORE_WEIGHT_SIL_DEFICIT,
        },
        "SEARCH_SPACE": SEARCH_SPACE,
    }


def _runtime_config_snapshot(cfg) -> dict:
    return {
        "input_dir": cfg.input_dir,
        "output_dir": cfg.output_dir,
        "model": cfg.model,
        "preprocessing": cfg.preprocessing,
        "cache": cfg.cache,
        "acceleration": cfg.acceleration,
        "reduction": cfg.reduction,
        "clustering": cfg.clustering,
        "purity": cfg.purity,
    }


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
    if USE_COMPACT_ITERATION_NAMES:
        return f"{prefix}_{idx:04d}"
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


def _search_space_size(keys: list[str]) -> int:
    total = 1
    for k in keys:
        total *= max(0, len(SEARCH_SPACE.get(k, [])))
    return total


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


def _compute_dynamic_score(item: dict) -> tuple[float, dict]:
    """Композитний score без жорсткого відсікання по verdict (менше = краще)."""
    cos = item.get("max_centroid_cosine")
    sil = item.get("silhouette")
    nn = item.get("max_nn_cross_ratio")
    th = item.get("thresholds", {})

    # Missing-метрики: великий штраф, але не краш ранжування.
    if cos is None or nn is None:
        return 9999.0, {"reason": "missing_core_metrics"}

    sil_safe = sil if sil is not None else -1.0
    base = (
        SCORE_WEIGHT_MAX_COS * cos
        + SCORE_WEIGHT_MAX_NN * nn
        - SCORE_WEIGHT_SILHOUETTE * sil_safe
    )

    cos_thr = th.get("max_centroid_cosine", 1.0)
    nn_thr = th.get("max_nn_cross_ratio", 1.0)
    sil_thr = th.get("min_silhouette", 0.0)
    cos_excess = max(0.0, cos - cos_thr)
    nn_excess = max(0.0, nn - nn_thr)
    sil_deficit = max(0.0, sil_thr - sil_safe)

    penalty = (
        SCORE_WEIGHT_COS_EXCESS * cos_excess
        + SCORE_WEIGHT_NN_EXCESS * nn_excess
        + SCORE_WEIGHT_SIL_DEFICIT * sil_deficit
    )
    score = base + penalty
    return score, {
        "base": base,
        "penalty": penalty,
        "cos_excess": cos_excess,
        "nn_excess": nn_excess,
        "sil_deficit": sil_deficit,
    }


def _score_for_sort(item: dict) -> tuple:
    """Менше значення -> кращий результат."""
    score = item.get("score")
    if score is None:
        score = 9999.0
    return (
        score,
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
    run_root: Path,
    features,
    flat_candidate: dict,
    name: str,
    force_full_logs: bool = False,
    eval_scope: str = "full",
):
    overrides = _flat_to_overrides(flat_candidate)
    red_over = overrides.get("reduction", {})
    clu_over = overrides.get("clustering", {})
    pur_over = overrides.get("purity", {})

    rcfg = _apply_section_overrides(cfg.reduction, red_over)
    ccfg = _apply_section_overrides(cfg.clustering, clu_over)
    purity_cfg = _apply_section_overrides(cfg.purity, pur_over)

    trial_output = run_root / name
    trial_output.mkdir(parents=True, exist_ok=True)

    # Явний конфіг ітерації: тільки параметри, які змінені відносно базового config.yaml.
    iteration_cfg_path = trial_output / "iteration_config.json"
    iteration_cfg_payload = {
        "name": name,
        "base_config_path": BASE_CONFIG_PATH,
        "changed_params": dict(flat_candidate),
    }
    iteration_cfg_path.write_text(
        json.dumps(iteration_cfg_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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
                viz_dir_name=PURITY_ARTIFACTS_DIR,
                cfg=purity_cfg,
                accel_cfg=cfg.acceleration,
                report_filename=f"iteration_metric_{name}.json",
                matrix_filename=f"cosine_{name}.npy",
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
            viz_dir_name=PURITY_ARTIFACTS_DIR,
            cfg=purity_cfg,
            accel_cfg=cfg.acceleration,
            report_filename=f"iteration_metric_{name}.json",
            matrix_filename=f"cosine_{name}.npy",
        )

    m = purity["metrics"]
    warn_count = 0
    if captured_stdout:
        warn_count = sum(1 for ln in captured_stdout.splitlines() if "[WARN]" in ln)
    thresholds = {
        "max_centroid_cosine": purity_cfg.max_centroid_cosine,
        "min_silhouette": purity_cfg.min_silhouette,
        "max_nn_cross_ratio": purity_cfg.max_nn_cross_ratio,
    }
    row = {
        "name": name,
        "verdict": purity["verdict"],
        "checks_passed": purity["verdict"] == "PASS",
        "n_clusters": purity["run"]["n_clusters"],
        "max_centroid_cosine": m["max_centroid_cosine"],
        "silhouette": m["silhouette_mean"],
        "max_nn_cross_ratio": m["max_nn_cross_ratio"],
        "leakage_candidates": len(purity.get("leakage_candidates", [])),
        "report_path": str(trial_output / PURITY_ARTIFACTS_DIR / f"iteration_metric_{name}.json"),
        "params": overrides,
        "flat_params": dict(flat_candidate),
        "warn_count": warn_count,
        "status": "ok",
        "thresholds": thresholds,
        "iteration_config_path": str(iteration_cfg_path),
        "eval_scope": eval_scope,
    }
    score, score_parts = _compute_dynamic_score(row)
    row["score"] = score
    row["score_parts"] = score_parts
    return row


def _save_tuning_plots(summary_dir: Path, results: list[dict], top_results: list[dict]) -> list[str]:
    """Save tuning progress plots; return created file paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    created = []
    ok_results = [r for r in results if r.get("status") == "ok"]
    if not ok_results:
        return created

    plt.style.use("seaborn-v0_8-whitegrid")

    # 1) Progress plot over evaluation index.
    eval_x = list(range(1, len(ok_results) + 1))
    max_cos = [r["max_centroid_cosine"] for r in ok_results]
    silhouette = [r["silhouette"] for r in ok_results]
    max_nn = [r["max_nn_cross_ratio"] for r in ok_results]
    scopes = [r.get("eval_scope", "full") for r in ok_results]
    switch_eval_idx = None
    for i, scope in enumerate(scopes, 1):
        if scope == "full":
            switch_eval_idx = i
            break

    def _best_so_far(vals, lower_is_better: bool):
        best = []
        cur = None
        for v in vals:
            if v is None:
                best.append(cur)
                continue
            if cur is None:
                cur = v
            elif lower_is_better and v < cur:
                cur = v
            elif (not lower_is_better) and v > cur:
                cur = v
            best.append(cur)
        return best

    best_max_cos = _best_so_far(max_cos, lower_is_better=True)
    best_sil = _best_so_far(silhouette, lower_is_better=False)
    best_max_nn = _best_so_far(max_nn, lower_is_better=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(eval_x, max_cos, marker="o", markersize=3, linewidth=1.2, label="raw")
    axes[0].plot(eval_x, best_max_cos, linestyle="--", linewidth=1.1, label="best-so-far")
    axes[0].set_ylabel("max_centroid_cosine")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(eval_x, silhouette, marker="o", markersize=3, linewidth=1.2, color="tab:orange", label="raw")
    axes[1].plot(eval_x, best_sil, linestyle="--", linewidth=1.1, color="tab:brown", label="best-so-far")
    axes[1].set_ylabel("silhouette")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(eval_x, max_nn, marker="o", markersize=3, linewidth=1.2, color="tab:green", label="raw")
    axes[2].plot(eval_x, best_max_nn, linestyle="--", linewidth=1.1, color="tab:olive", label="best-so-far")
    axes[2].set_ylabel("max_nn_cross_ratio")
    axes[2].set_xlabel("Successful evaluation #")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(alpha=0.3)

    if switch_eval_idx is not None and switch_eval_idx > 1:
        for ax in axes:
            ax.axvline(switch_eval_idx, color="tab:purple", linestyle=":", linewidth=1.3)
            ax.text(
                switch_eval_idx + 0.2,
                ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08,
                "switch->full",
                color="tab:purple",
                fontsize=8,
            )

    fig.suptitle("Tuning Progress")
    fig.tight_layout()
    progress_path = summary_dir / "tuning_progress.png"
    fig.savefig(progress_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    created.append(str(progress_path))

    # 2) Top-20 max_cos + silhouettes.
    top_n = top_results[:20]
    names = [r["name"][-28:] for r in top_n]
    cos_vals = [r["max_centroid_cosine"] for r in top_n]
    sil_vals = [r["silhouette"] if r["silhouette"] is not None else 0.0 for r in top_n]

    fig, ax1 = plt.subplots(figsize=(14, 7))
    bars = ax1.bar(range(len(top_n)), cos_vals, color="tab:red", alpha=0.72)
    ax1.set_ylabel("max_centroid_cosine", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_xticks(range(len(top_n)))
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(range(len(top_n)), sil_vals, color="tab:blue", marker="o", linewidth=1.3)
    ax2.set_ylabel("silhouette", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    for i, (b, v) in enumerate(zip(bars, cos_vals), 1):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            v + 0.002,
            f"#{i} {v:.3f}",
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


def _save_all_results_csv(summary_dir: Path, results: list[dict]) -> Path:
    """Save all tuning runs and metrics to CSV."""
    csv_path = summary_dir / "tuning_results_all.csv"
    fieldnames = [
        "section",
        "rank",
        "name",
        "eval_scope",
        "score",
        "status",
        "checks_passed",
        "verdict",
        "n_clusters",
        "max_centroid_cosine",
        "silhouette",
        "max_nn_cross_ratio",
        "leakage_candidates",
        "warn_count",
        "elapsed_sec",
        "report_path",
        "iteration_config_path",
        "params_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "section": "all",
                    "rank": "",
                    "name": row.get("name", ""),
                    "eval_scope": row.get("eval_scope", ""),
                    "score": row.get("score"),
                    "status": row.get("status", ""),
                    "checks_passed": row.get("checks_passed", False),
                    "verdict": row.get("verdict", ""),
                    "n_clusters": row.get("n_clusters"),
                    "max_centroid_cosine": row.get("max_centroid_cosine"),
                    "silhouette": row.get("silhouette"),
                    "max_nn_cross_ratio": row.get("max_nn_cross_ratio"),
                    "leakage_candidates": row.get("leakage_candidates"),
                    "warn_count": row.get("warn_count", 0),
                    "elapsed_sec": row.get("elapsed_sec"),
                    "report_path": row.get("report_path", ""),
                    "iteration_config_path": row.get("iteration_config_path", ""),
                    "params_json": json.dumps(row.get("params", {}), ensure_ascii=False),
                }
            )
        ok_results = [r for r in results if r.get("status") == "ok"]
        full_ok_results = [r for r in ok_results if r.get("eval_scope") == "full"]
        ranked_pool = full_ok_results if full_ok_results else ok_results
        top_sorted = sorted(ranked_pool, key=_score_for_sort)[:20]
        writer.writerow({})
        for i, row in enumerate(top_sorted, 1):
            writer.writerow(
                {
                    "section": "top",
                    "rank": i,
                    "name": row.get("name", ""),
                    "eval_scope": row.get("eval_scope", ""),
                    "score": row.get("score"),
                    "status": row.get("status", ""),
                    "checks_passed": row.get("checks_passed", False),
                    "verdict": row.get("verdict", ""),
                    "n_clusters": row.get("n_clusters"),
                    "max_centroid_cosine": row.get("max_centroid_cosine"),
                    "silhouette": row.get("silhouette"),
                    "max_nn_cross_ratio": row.get("max_nn_cross_ratio"),
                    "leakage_candidates": row.get("leakage_candidates"),
                    "warn_count": row.get("warn_count", 0),
                    "elapsed_sec": row.get("elapsed_sec"),
                    "report_path": row.get("report_path", ""),
                    "iteration_config_path": row.get("iteration_config_path", ""),
                    "params_json": json.dumps(row.get("params", {}), ensure_ascii=False),
                }
            )
    return csv_path


def _generate_full_visualizations_for_top_n(
    cfg,
    run_root: Path,
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

        run_dir = run_root / "_top_visuals" / f"top_{rank:02d}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        embeddings = reduce_dimensions(features, rcfg, cfg.acceleration)
        labels = cluster(embeddings, ccfg, cfg.acceleration)
        purity = evaluate_and_save_purity(
            features=features,
            embeddings=embeddings,
            labels=labels,
            output_dir=run_dir,
            viz_dir_name=PURITY_ARTIFACTS_DIR,
            cfg=purity_cfg,
            accel_cfg=cfg.acceleration,
            report_filename=f"iteration_metric_top_{rank:02d}_{name}.json",
            matrix_filename=f"cosine_top_{rank:02d}_{name}.npy",
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
        metrics_dir = run_dir / PURITY_ARTIFACTS_DIR
        generated.append(
            {
                "rank": rank,
                "name": name,
                "verdict": purity["verdict"],
                "output_dir": str(run_dir),
                "viz_dir": str(viz_dir),
                "purity_report": str(metrics_dir / f"iteration_metric_top_{rank:02d}_{name}.json"),
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
    run_root: Path,
    keys: list[str],
    features,
    full_features,
    max_trials: int,
    phase_tag: str,
    rng_seed: int,
):
    rng = random.Random(rng_seed)
    pop_size = EVOLUTION_POPULATION_SIZE
    elite_size = min(EVOLUTION_ELITE_SIZE, pop_size)
    tournament_size = EVOLUTION_TOURNAMENT_SIZE
    mutation_rate = EVOLUTION_MUTATION_RATE
    use_subset_first = TWO_PHASE_ENABLED and PHASE1_FRACTION < 1.0
    active_scope = "subset" if use_subset_first else "full"
    switched_to_full = not use_subset_first
    current_features = features if use_subset_first else full_features

    results = []
    evaluated_signatures = set()
    start_all = time.time()
    total_unique = _search_space_size(keys)
    if total_unique <= 0:
        raise ValueError("SEARCH_SPACE порожній: немає кандидатів для еволюції.")
    if pop_size > total_unique:
        print(f"[WARN] population_size={pop_size} > search_space={total_unique}; cap до {total_unique}")
        pop_size = total_unique
        elite_size = min(elite_size, pop_size)

    def log_result(idx: int, total: int, row: dict, elapsed_iter: float):
        elapsed_all = time.time() - start_all
        avg_iter = elapsed_all / max(1, idx)
        eta = avg_iter * max(0, total - idx)
        eta_h = int(eta // 3600)
        eta_m = int((eta % 3600) // 60)
        scope = row.get("eval_scope", "full")
        if scope == "subset":
            scope_label = f"subset({PHASE1_FRACTION*100:.1f}%)"
        elif scope == "full":
            scope_label = "full(100%)"
        else:
            scope_label = str(scope)
        if row["status"] == "ok":
            print(
                "  RESULT:",
                f"data={scope_label},",
                f"score={row.get('score', float('nan')):.4f},",
                f"checks={row.get('checks_passed', False)},",
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
                f"data={scope_label},",
                row["status"],
                f"time={elapsed_iter:.1f}s,",
                f"ETA~{eta_h:02d}:{eta_m:02d}",
            )

    population = []
    attempts = 0
    max_attempts = max(1000, pop_size * 100)
    while len(population) < pop_size and attempts < max_attempts:
        attempts += 1
        cand = _random_flat_candidate(rng, keys)
        sig = _candidate_signature(cand, keys)
        if sig in evaluated_signatures:
            continue
        evaluated_signatures.add(sig)
        population.append(cand)
    if len(population) < pop_size:
        raise RuntimeError(
            f"Не вдалося зібрати початкову популяцію: {len(population)}/{pop_size}. "
            "Перевір SEARCH_SPACE."
        )

    eval_count = 0
    generation = 0
    stagnation_generations = 0
    best_score = None
    best_score_history = []

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
                    run_root,
                    current_features,
                    cand,
                    name,
                    force_full_logs=detailed_now,
                    eval_scope=active_scope,
                )
            except Exception as exc:
                row = {
                    "name": name,
                    "verdict": "N/A",
                    "checks_passed": False,
                    "n_clusters": -1,
                    "max_centroid_cosine": None,
                    "silhouette": None,
                    "max_nn_cross_ratio": None,
                    "leakage_candidates": -1,
                    "report_path": "",
                    "params": _flat_to_overrides(cand),
                    "flat_params": dict(cand),
                    "status": f"error: {type(exc).__name__}: {exc}",
                    "score": None,
                    "iteration_config_path": "",
                    "eval_scope": active_scope,
                }

            elapsed_iter = time.time() - t_iter
            row["elapsed_sec"] = round(elapsed_iter, 4)
            results.append(row)
            gen_rows.append(row)
            log_result(eval_count, max_trials, row, elapsed_iter)

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
        if best_score is not None:
            best_score_history.append(best_score[0])

        can_switch = (
            (not switched_to_full)
            and eval_count >= SWITCH_MIN_EVALUATIONS
            and len(best_score_history) >= SWITCH_PATIENCE_GENERATIONS + 1
        )
        slow_improvement = False
        if can_switch:
            prev_best = best_score_history[-(SWITCH_PATIENCE_GENERATIONS + 1)]
            curr_best = best_score_history[-1]
            rel_gain = (prev_best - curr_best) / max(abs(prev_best), 1e-9)
            slow_improvement = rel_gain < SWITCH_MIN_REL_IMPROVEMENT
            if stagnation_generations >= SWITCH_PATIENCE_GENERATIONS or slow_improvement:
                print("\n" + "#" * 72)
                print(
                    "ADAPTIVE SWITCH: subset -> full "
                    f"(stagnation={stagnation_generations}, rel_gain={rel_gain:.4f})"
                )
                print("#" * 72)

                ranked = sorted(ok_pool, key=_score_for_sort) if ok_pool else []
                elites = ranked[: max(1, min(elite_size, SWITCH_REEVAL_ELITES_TOP_K))]
                reevaluated = []
                current_features = full_features
                active_scope = "full"
                switched_to_full = True

                for elite in elites:
                    if eval_count >= max_trials:
                        break
                    eval_count += 1
                    t_iter = time.time()
                    elite_flat = dict(elite["flat_params"])
                    name = _candidate_name(f"evo_{phase_tag.lower()}_fullseed", eval_count, elite_flat, keys)
                    print("\n" + "-" * 72)
                    print(f"[{eval_count}/{max_trials}] {name} (re-eval elite on full)")
                    try:
                        row = _evaluate_iteration(
                            cfg,
                            run_root,
                            current_features,
                            elite_flat,
                            name,
                            force_full_logs=False,
                            eval_scope="full",
                        )
                    except Exception as exc:
                        row = {
                            "name": name,
                            "verdict": "N/A",
                            "checks_passed": False,
                            "n_clusters": -1,
                            "max_centroid_cosine": None,
                            "silhouette": None,
                            "max_nn_cross_ratio": None,
                            "leakage_candidates": -1,
                            "report_path": "",
                            "params": _flat_to_overrides(elite_flat),
                            "flat_params": elite_flat,
                            "status": f"error: {type(exc).__name__}: {exc}",
                            "score": None,
                            "iteration_config_path": "",
                            "eval_scope": "full",
                        }
                    elapsed_iter = time.time() - t_iter
                    row["elapsed_sec"] = round(elapsed_iter, 4)
                    results.append(row)
                    reevaluated.append(row)
                    log_result(eval_count, max_trials, row, elapsed_iter)

                reevaluated_ok = [r for r in reevaluated if r["status"] == "ok"]
                if reevaluated_ok:
                    best_score = min(reevaluated_ok, key=_score_for_sort)
                    best_score = _score_for_sort(best_score)
                    best_score_history = [best_score[0]]
                    seeds = [dict(r["flat_params"]) for r in sorted(reevaluated_ok, key=_score_for_sort)[:elite_size]]
                    population = []
                    for seed in seeds:
                        if len(population) >= pop_size:
                            break
                        population.append(seed)
                    refill_attempts = 0
                    refill_max_attempts = max(1000, pop_size * 100)
                    while (
                        len(population) < pop_size
                        and len(evaluated_signatures) < total_unique
                        and refill_attempts < refill_max_attempts
                    ):
                        refill_attempts += 1
                        cand = _random_flat_candidate(rng, keys)
                        sig = _candidate_signature(cand, keys)
                        if sig in evaluated_signatures:
                            continue
                        evaluated_signatures.add(sig)
                        population.append(cand)
                stagnation_generations = 0
                if eval_count >= max_trials:
                    break
                continue

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
            refill_attempts = 0
            refill_max_attempts = max(1000, pop_size * 100)
            while (
                len(population) < pop_size
                and len(evaluated_signatures) < total_unique
                and refill_attempts < refill_max_attempts
            ):
                refill_attempts += 1
                cand = _random_flat_candidate(rng, keys)
                sig = _candidate_signature(cand, keys)
                if sig in evaluated_signatures:
                    continue
                evaluated_signatures.add(sig)
                population.append(cand)
            if len(population) < pop_size:
                print("[WARN] Простір кандидатів вичерпано; завершуємо еволюцію.")
                break
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

        refill_attempts = 0
        refill_max_attempts = max(1000, pop_size * 120)
        while (
            len(next_population) < pop_size
            and len(evaluated_signatures) < total_unique
            and refill_attempts < refill_max_attempts
        ):
            refill_attempts += 1
            child = _random_flat_candidate(rng, keys)
            sig = _candidate_signature(child, keys)
            if sig in evaluated_signatures:
                continue
            evaluated_signatures.add(sig)
            next_population.append(child)

        if len(next_population) < pop_size:
            print("[WARN] Простір кандидатів майже вичерпано; зменшуємо популяцію наступного покоління.")
        population = next_population

    return results


def main() -> None:
    config_path = BASE_CONFIG_PATH
    max_trials = MAX_EVALUATIONS

    cfg = load_config(config_path)
    keys = _space_keys()
    run_root = _next_tuning_run_root(cfg.output_dir)
    run_root.mkdir(parents=True, exist_ok=False)

    print("=" * 72)
    print("  ITERATIVE TUNING: CLUSTER SEARCH")
    print("=" * 72)
    print(f"Config:  {config_path}")
    print(f"Input:   {cfg.input_dir}")
    print(f"Output:  {cfg.output_dir}")
    print(f"Run root: {run_root}")
    print(f"Strategy: {SEARCH_STRATEGY}")
    print(f"Max evaluations: {max_trials}")
    print(f"Adaptive subset->full: {TWO_PHASE_ENABLED} (phase1_fraction={PHASE1_FRACTION})")
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
    tuning_manifest = {
        "base_config_path": config_path,
        "run_root": run_root,
        "tuner_settings": _tuner_settings_snapshot(),
        "runtime_config": _runtime_config_snapshot(cfg),
    }
    (run_root / "tuning_config.json").write_text(
        json.dumps(_json_ready(tuning_manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cache_info = get_last_cache_info()
    if cache_info and cache_info.get("cache_manifest_path"):
        src_manifest = Path(cache_info["cache_manifest_path"])
        if src_manifest.exists():
            shutil.copy2(src_manifest, run_root / "_cache_config.json")

    print("\n[3/4] Ітеративний прогін конфігів...")
    results = []
    features_subset = features
    if TWO_PHASE_ENABLED and PHASE1_FRACTION < 1.0:
        features_subset, _ = _subsample_features(features, PHASE1_FRACTION, EVOLUTION_RANDOM_SEED)
        print("\n" + "#" * 72)
        print(
            f"ADAPTIVE START: subset {PHASE1_FRACTION*100:.1f}% -> full 100% "
            "on plateau/slow-improvement"
        )
        print("#" * 72)
    else:
        print("\n" + "#" * 72)
        print("ADAPTIVE START: full 100% from first generation")
        print("#" * 72)

    phase_results = _run_evolution_search(
        cfg=cfg,
        run_root=run_root,
        keys=keys,
        features=features_subset,
        full_features=features,
        max_trials=max_trials,
        phase_tag="EVOLVE",
        rng_seed=EVOLUTION_RANDOM_SEED,
    )
    results.extend(phase_results)

    print("\n[4/4] Підсумок")
    print("-" * 72)
    ok_results = [r for r in results if r["status"] == "ok"]
    if not ok_results:
        print("Не вдалося завершити жодної ітерації успішно.")
        return

    full_ok_results = [r for r in ok_results if r.get("eval_scope") == "full"]
    if full_ok_results:
        results_sorted = sorted(full_ok_results, key=_score_for_sort)
    else:
        print("  [WARN] Немає full-evals, fallback на всі успішні ітерації.")
        results_sorted = sorted(ok_results, key=_score_for_sort)
    for i, r in enumerate(results_sorted, 1):
        print(
            f"{i:>2}. {r['name']:<22} "
            f"score={r.get('score', float('nan')):.4f} "
            f"checks={r.get('checks_passed', False)!s:<5} "
            f"{r['verdict']:<4} "
            f"clusters={r['n_clusters']:<4} "
            f"max_cos={r['max_centroid_cosine']:.4f} "
            f"sil={r['silhouette'] if r['silhouette'] is not None else 'n/a'} "
            f"nn={r['max_nn_cross_ratio']:.4f} "
            f"leaks={r['leakage_candidates']}"
        )

    best = results_sorted[0]
    print("-" * 72)
    print(
        f"BEST: {best['name']} | score={best.get('score', float('nan')):.4f} "
        f"| verdict={best['verdict']} | report={best['report_path']}"
    )
    print(f"BEST reduction:  {best['params']['reduction']}")
    print(f"BEST clustering: {best['params']['clustering']}")

    summary_dir = run_root
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "tuning_summary_latest.json"
    summary = {
        "config_path": config_path,
        "strategy": SEARCH_STRATEGY,
        "total_trials_requested": max_trials,
        "total_trials_ok": len(ok_results),
        "total_trials_ok_full": len(full_ok_results),
        "best": best,
        "top20": results_sorted[:20],
        "all_results": results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")
    csv_path = _save_all_results_csv(summary_dir, results)
    print(f"All runs CSV: {csv_path}")
    summary["all_results_csv"] = str(csv_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

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
                run_root=run_root,
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
