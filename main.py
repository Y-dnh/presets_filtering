"""Фільтрація датасету тепловізійних камер за пресетами (кутами огляду).

Запуск:
    python main.py                        # конфіг з config.yaml
    python main.py --config my_config.yaml # інший конфіг
"""

from __future__ import annotations

import sys
import time

from src.config import load_config
from src.image_scanner import scan_images
from src.feature_extractor import extract_features
from src.reducer import reduce_dimensions
from src.clusterer import cluster, build_report
from src.organizer import organize_files
from src.visualizer import visualize


def main() -> None:
    config_path = "config.yaml"
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--config" and i < len(sys.argv) - 1:
            config_path = sys.argv[i + 1]

    print("=" * 50)
    print("  ФІЛЬТРАЦІЯ ДАТАСЕТУ ЗА ПРЕСЕТАМИ КАМЕР")
    print("=" * 50)

    # 1. Завантаження конфігурації
    print("\n[1/7] Завантаження конфігурації...")
    cfg = load_config(config_path)
    print(f"  Вхідна папка:  {cfg.input_dir}")
    print(f"  Вихідна папка: {cfg.output_dir}")
    print(f"  Режим: {cfg.mode}")

    # 2. Сканування зображень
    print("\n[2/7] Сканування зображень...")
    image_paths = scan_images(cfg.input_dir, cfg.image_extensions)
    if not image_paths:
        print("  Зображень не знайдено. Перевірте input_dir та image_extensions.")
        sys.exit(1)
    print(f"  Знайдено зображень: {len(image_paths)}")

    # 3. Витягування ознак
    print("\n[3/7] Витягування ознак нейронною мережею...")
    t0 = time.time()
    features = extract_features(image_paths, cfg.model, cfg.preprocessing, cfg.cache)
    print(f"  Час: {time.time() - t0:.1f}с")

    # 4. Зменшення розмірності
    print("\n[4/7] Зменшення розмірності (PCA + UMAP)...")
    t0 = time.time()
    embeddings = reduce_dimensions(features, cfg.reduction)
    print(f"  Час: {time.time() - t0:.1f}с")

    # 5. Кластеризація
    print("\n[5/7] Кластеризація (HDBSCAN)...")
    t0 = time.time()
    labels = cluster(embeddings, cfg.clustering)
    print(f"  Час: {time.time() - t0:.1f}с")

    # Звіт
    report = build_report(labels)
    print(report)

    # 6. Візуалізація
    if cfg.visualization.enabled:
        print("[6/7] Візуалізація...")
        t0 = time.time()
        visualize(image_paths, embeddings, labels, cfg.output_dir, cfg.visualization, cfg.preprocessing)
        print(f"  Час: {time.time() - t0:.1f}с")
    else:
        print("[6/7] Візуалізація вимкнена (visualization.enabled: false)")

    # 7. Підтвердження та організація
    answer = input("\nПродовжити розкладання файлів? (y/n): ").strip().lower()
    if answer not in ("y", "yes", "т", "так"):
        print("Скасовано.")
        sys.exit(0)

    print("\n[7/7] Організація файлів...")
    t0 = time.time()
    organize_files(image_paths, labels, cfg.output_dir, cfg.mode, cfg.annotations)
    print(f"  Час: {time.time() - t0:.1f}с")

    print("\nГотово!")
    print(f"Результати: {cfg.output_dir}")
    if cfg.visualization.enabled:
        viz_dir = cfg.output_dir / cfg.visualization.viz_dir
        print(f"Візуалізації: {viz_dir}")
        print(f"  Інтерактивний scatter: {viz_dir / 'interactive_scatter.html'}")
        print(f"  Розміри кластерів:     {viz_dir / 'cluster_sizes.png'}")
        print(f"  Сітка зразків:         {viz_dir / 'cluster_grid.png'}")


if __name__ == "__main__":
    main()
