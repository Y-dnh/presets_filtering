from __future__ import annotations

import base64
import io
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm
from umap import UMAP

import cv2

from .config import PreprocessingConfig, VisualizationConfig


def _thumbnail_to_base64(path: Path, size: int) -> str:
    """Load image, resize to thumbnail, encode as base64 JPEG."""
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


def _project_to_3d(embeddings: np.ndarray) -> np.ndarray:
    """UMAP projection to 3D specifically for visualization."""
    if embeddings.shape[1] == 3:
        return embeddings
    target = min(3, embeddings.shape[1])
    n_neighbors = min(15, embeddings.shape[0] - 1)
    reducer = UMAP(
        n_components=target,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        verbose=False,
    )
    return reducer.fit_transform(embeddings)


def _cluster_color_map(labels: np.ndarray) -> dict[int, str]:
    """Generate distinct colors for each cluster label."""
    unique = sorted(set(labels))
    palette = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990",
        "#e6beff", "#9A6324", "#800000", "#aaffc3", "#808000",
        "#ffd8b1", "#000075", "#a9a9a9", "#fffac8", "#dcbeff",
    ]
    colors = {}
    for i, label in enumerate(unique):
        if label == -1:
            colors[label] = "#888888"
        else:
            colors[label] = palette[i % len(palette)]
    return colors


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="utf-8">
<title>Кластеризація пресетів — 3D</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#111; color:#eee; font-family:'Segoe UI',system-ui,sans-serif; height:100vh; overflow:hidden; }
  .wrap { display:flex; height:100vh; }
  #chart { flex:1; min-width:0; }
  .sidebar { width:360px; border-left:1px solid #333; display:flex; flex-direction:column; align-items:center; padding:16px; gap:12px; overflow-y:auto; }
  .sidebar h3 { font-size:14px; color:#888; font-weight:400; }
  .preview-img { width:300px; height:300px; object-fit:contain; border-radius:6px; border:1px solid #333; background:#1a1a1a; display:none; }
  .meta { text-align:center; font-size:13px; line-height:1.6; }
  .meta b { color:#fff; font-size:14px; }
  .meta .cluster-badge { display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; margin-top:4px; }
  .hint { color:#555; font-size:13px; margin-top:40px; text-align:center; }
  .controls { display:flex; gap:8px; flex-wrap:wrap; justify-content:center; }
  .controls button { background:#222; color:#aaa; border:1px solid #444; border-radius:4px; padding:4px 12px; cursor:pointer; font-size:12px; }
  .controls button:hover { background:#333; color:#fff; }
  .stats { font-size:12px; color:#666; margin-top:auto; padding-top:12px; border-top:1px solid #222; text-align:center; }
</style>
</head>
<body>
<div class="wrap">
  <div id="chart"></div>
  <div class="sidebar">
    <h3>Превью зображення</h3>
    <img id="previewImg" class="preview-img" alt="preview">
    <div id="metaInfo" class="meta"></div>
    <div id="hint" class="hint">Наведіть курсор на точку<br>щоб побачити зображення<br><br><small>Обертання: ЛКМ + тягнути<br>Зум: колесо миші<br>Переміщення: ПКМ + тягнути</small></div>
    <div class="controls">
      <button onclick="resetCamera()">Скинути камеру</button>
      <button onclick="toggleSpin()">Авто-обертання</button>
    </div>
    <div class="stats" id="stats"></div>
  </div>
</div>
<script>
const THUMBS = %%THUMBS%%;
const META = %%META%%;
const TRACES = %%TRACES%%;
const LAYOUT = %%LAYOUT%%;

Plotly.newPlot('chart', TRACES, LAYOUT, {responsive:true});

var imgEl = document.getElementById('previewImg');
var metaEl = document.getElementById('metaInfo');
var hintEl = document.getElementById('hint');
var statsEl = document.getElementById('stats');
var chartEl = document.getElementById('chart');
var spinning = false;
var spinInterval = null;

statsEl.textContent = 'Точок: ' + Object.keys(THUMBS).length;

chartEl.on('plotly_hover', function(data) {
  var pt = data.points[0];
  var gIdx = pt.customdata;
  var thumb = THUMBS[gIdx];
  var m = META[gIdx];
  if (thumb) {
    imgEl.src = 'data:image/jpeg;base64,' + thumb;
    imgEl.style.display = 'block';
    hintEl.style.display = 'none';
    metaEl.innerHTML = '<b>' + m.name + '</b><br>' +
      '<span class="cluster-badge" style="background:' + m.color + '">' + m.cluster_name + '</span>';
  }
});

chartEl.on('plotly_click', function(data) {
  var pt = data.points[0];
  var gIdx = pt.customdata;
  var m = META[gIdx];
  if (m && m.path) {
    metaEl.innerHTML += '<br><span style="color:#888;font-size:11px;">' + m.path + '</span>';
  }
});

function resetCamera() {
  Plotly.relayout('chart', {
    'scene.camera': {eye:{x:1.5, y:1.5, z:1.5}, up:{x:0,y:0,z:1}}
  });
}

function toggleSpin() {
  spinning = !spinning;
  if (spinning) {
    var angle = 0;
    spinInterval = setInterval(function() {
      angle += 0.02;
      var r = 2.0;
      Plotly.relayout('chart', {
        'scene.camera.eye': {x: r*Math.cos(angle), y: r*Math.sin(angle), z: 0.8}
      });
    }, 50);
  } else {
    clearInterval(spinInterval);
  }
}
</script>
</body>
</html>"""


def create_interactive_scatter(
    image_paths: List[Path],
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    cfg: VisualizationConfig,
) -> Path:
    """Create interactive HTML scatter with image preview panel on hover."""
    n_total = len(image_paths)
    max_pts = cfg.max_points

    if n_total > max_pts:
        indices = sorted(random.sample(range(n_total), max_pts))
    else:
        indices = list(range(n_total))

    print(f"    3D проекція для {len(indices)} точок...")
    sub_embeddings = embeddings[indices]
    sub_labels = labels[indices]
    sub_paths = [image_paths[i] for i in indices]

    coords_3d = _project_to_3d(sub_embeddings)

    print(f"    Генерація мініатюр ({cfg.thumbnail_size}px)...")
    thumbs_dict: dict[int, str] = {}
    meta_dict: dict[int, dict] = {}
    color_map = _cluster_color_map(sub_labels)

    for i, (path, label) in enumerate(
        tqdm(zip(sub_paths, sub_labels), total=len(sub_paths), desc="    Мініатюри", unit="img")
    ):
        thumbs_dict[i] = _thumbnail_to_base64(path, cfg.thumbnail_size)
        cname = "noise" if label == -1 else f"preset_{label:03d}"
        meta_dict[i] = {
            "name": path.name,
            "path": str(path),
            "cluster": int(label),
            "cluster_name": cname,
            "color": color_map[int(label)],
        }

    cluster_ids = sorted(set(sub_labels))
    traces = []
    for cid in cluster_ids:
        mask = sub_labels == cid
        global_indices = np.where(mask)[0]
        cname = "noise" if cid == -1 else f"preset_{cid:03d}"

        traces.append({
            "x": coords_3d[mask, 0].tolist(),
            "y": coords_3d[mask, 1].tolist(),
            "z": coords_3d[mask, 2].tolist(),
            "mode": "markers",
            "type": "scatter3d",
            "name": cname,
            "marker": {"color": color_map[cid], "size": 3, "opacity": 0.8},
            "customdata": global_indices.tolist(),
            "hovertemplate": "<b>%{customdata}</b><extra>" + cname + "</extra>",
        })

    layout = {
        "title": f"Кластеризація пресетів 3D ({len(cluster_ids)} кластерів, {len(indices)} зображень)",
        "scene": {
            "xaxis": {"title": "UMAP-1", "color": "#888", "gridcolor": "#222", "backgroundcolor": "#111"},
            "yaxis": {"title": "UMAP-2", "color": "#888", "gridcolor": "#222", "backgroundcolor": "#111"},
            "zaxis": {"title": "UMAP-3", "color": "#888", "gridcolor": "#222", "backgroundcolor": "#111"},
            "bgcolor": "#111",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}},
        },
        "paper_bgcolor": "#111",
        "font": {"color": "#ccc"},
        "legend": {"title": {"text": "Пресети"}, "itemsizing": "constant"},
        "hoverlabel": {"bgcolor": "#222", "font_size": 12},
        "margin": {"l": 0, "r": 0, "t": 50, "b": 0},
    }

    html = _HTML_TEMPLATE
    html = html.replace("%%THUMBS%%", json.dumps(thumbs_dict))
    html = html.replace("%%META%%", json.dumps(meta_dict))
    html = html.replace("%%TRACES%%", json.dumps(traces))
    html = html.replace("%%LAYOUT%%", json.dumps(layout))

    output_path = output_dir / "interactive_scatter.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def create_cluster_grid(
    image_paths: List[Path],
    labels: np.ndarray,
    output_dir: Path,
    cfg: VisualizationConfig,
) -> Path:
    """Create a static grid showing sample images from each cluster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cluster_to_paths: dict[int, List[Path]] = defaultdict(list)
    for path, label in zip(image_paths, labels):
        cluster_to_paths[label].append(path)

    cluster_ids = sorted(cluster_to_paths.keys())
    n_clusters = len(cluster_ids)
    samples = cfg.samples_per_cluster
    cols = min(samples, 8)
    rows_per = (samples + cols - 1) // cols

    fig_height = n_clusters * rows_per * 1.8 + 1
    fig_width = cols * 2
    fig, axes_grid = plt.subplots(
        n_clusters * rows_per, cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    for ax_row in axes_grid:
        for ax in ax_row:
            ax.axis("off")

    for cluster_idx, cid in enumerate(cluster_ids):
        paths = cluster_to_paths[cid]
        sampled = random.sample(paths, min(samples, len(paths)))
        name = "noise" if cid == -1 else f"preset_{cid:03d}"

        for i, path in enumerate(sampled):
            row = cluster_idx * rows_per + i // cols
            col = i % cols
            try:
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.thumbnail((256, 256))
                axes_grid[row][col].imshow(img)
            except Exception:
                pass
            if i == 0:
                axes_grid[row][0].set_title(
                    f"{name} ({len(paths)} imgs)", fontsize=10, loc="left"
                )

    plt.tight_layout()
    output_path = output_dir / "cluster_grid.png"
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_size_chart(
    labels: np.ndarray,
    output_dir: Path,
) -> Path:
    """Create a bar chart of cluster sizes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter

    counts = Counter(labels)
    cluster_ids = sorted(counts.keys())
    names = ["noise" if c == -1 else f"preset_{c:03d}" for c in cluster_ids]
    sizes = [counts[c] for c in cluster_ids]

    color_map = _cluster_color_map(labels)
    colors = [color_map[c] for c in cluster_ids]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6), 5))
    bars = ax.bar(names, sizes, color=colors, edgecolor="white", linewidth=0.5)

    for bar, size in zip(bars, sizes):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sizes) * 0.01,
            str(size), ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("Кластер (пресет)")
    ax.set_ylabel("Кількість зображень")
    ax.set_title("Розподіл зображень по кластерах")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = output_dir / "cluster_sizes.png"
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_preprocessing_comparison(
    image_paths: List[Path],
    labels: np.ndarray,
    output_dir: Path,
    preproc_cfg: PreprocessingConfig,
    n_samples: int = 8,
) -> Path:
    """Create a comparison grid: Original / CLAHE(normal) / CLAHE(inverted).

    Shows what the dual-pass sees — both versions that get averaged by the model.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cluster_to_paths: dict[int, List[Path]] = defaultdict(list)
    for path, label in zip(image_paths, labels):
        cluster_to_paths[label].append(path)

    cluster_ids = sorted(k for k in cluster_to_paths.keys() if k != -1)

    sampled: List[Path] = []
    per_cluster = max(1, n_samples // len(cluster_ids)) if cluster_ids else n_samples
    for cid in cluster_ids:
        paths = cluster_to_paths[cid]
        sampled.extend(random.sample(paths, min(per_cluster, len(paths))))
        if len(sampled) >= n_samples:
            break
    sampled = sampled[:n_samples]

    n_cols = 3
    fig, axes = plt.subplots(len(sampled), n_cols, figsize=(n_cols * 5, len(sampled) * 2.5))
    if len(sampled) == 1:
        axes = [axes]

    method = preproc_cfg.equalization
    clahe_obj = None
    if method == "clahe":
        clahe_obj = cv2.createCLAHE(
            clipLimit=preproc_cfg.clahe_clip_limit,
            tileGridSize=(preproc_cfg.clahe_grid_size, preproc_cfg.clahe_grid_size),
        )

    def _equalize(gray_img: np.ndarray) -> np.ndarray:
        if method == "clahe" and clahe_obj is not None:
            return clahe_obj.apply(gray_img)
        if method == "he":
            return cv2.equalizeHist(gray_img)
        return gray_img

    for i, path in enumerate(sampled):
        try:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

            has_trim = (preproc_cfg.trim_top > 0 or preproc_cfg.trim_bottom > 0
                        or preproc_cfg.trim_left > 0 or preproc_cfg.trim_right > 0)
            if has_trim:
                h, w = gray.shape[:2]
                y1 = int(h * preproc_cfg.trim_top)
                y2 = h - int(h * preproc_cfg.trim_bottom)
                x1 = int(w * preproc_cfg.trim_left)
                x2 = w - int(w * preproc_cfg.trim_right)
                gray = gray[y1:y2, x1:x2]

            eq_normal = _equalize(gray)
            eq_inverted = _equalize(255 - gray)

            axes[i][0].imshow(gray, cmap="gray")
            axes[i][0].axis("off")
            axes[i][1].imshow(eq_normal, cmap="gray")
            axes[i][1].axis("off")
            axes[i][2].imshow(eq_inverted, cmap="gray")
            axes[i][2].axis("off")

            trim_label = ""
            if has_trim:
                trim_label = " (trimmed)"
            method_label = method.upper() if method != "none" else "raw"

            if i == 0:
                axes[i][0].set_title(f"Оригінал{trim_label}", fontsize=12, fontweight="bold")
                axes[i][1].set_title(f"{method_label} (normal)", fontsize=12, fontweight="bold")
                axes[i][2].set_title(f"{method_label} (inverted)", fontsize=12, fontweight="bold")

            axes[i][0].set_ylabel(
                path.name, fontsize=6, rotation=0, labelpad=80, va="center"
            )
        except Exception:
            for c in range(n_cols):
                axes[i][c].axis("off")

    plt.suptitle(
        "Dual-pass: модель бачить ОБА варіанти та усереднює ознаки",
        fontsize=14, fontweight="bold", y=1.0,
    )
    plt.tight_layout()

    output_path = output_dir / "preprocessing_comparison.png"
    fig.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def visualize(
    image_paths: List[Path],
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    cfg: VisualizationConfig,
    preproc_cfg: PreprocessingConfig | None = None,
) -> None:
    """Run all visualizations and save to viz_dir inside output_dir."""
    viz_dir = output_dir / cfg.viz_dir
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("  Інтерактивний scatter-графік...")
    scatter_path = create_interactive_scatter(
        image_paths, embeddings, labels, viz_dir, cfg
    )
    print(f"    -> {scatter_path}")

    print("  Графік розмірів кластерів...")
    chart_path = create_size_chart(labels, viz_dir)
    print(f"    -> {chart_path}")

    print("  Сітка зразків по кластерах...")
    grid_path = create_cluster_grid(image_paths, labels, viz_dir, cfg)
    print(f"    -> {grid_path}")

    if preproc_cfg is not None and preproc_cfg.equalization != "none":
        print("  Порівняння препроцесингу (до/після)...")
        comparison_path = create_preprocessing_comparison(
            image_paths, labels, viz_dir, preproc_cfg,
            n_samples=cfg.clahe_comparison_samples,
        )
        print(f"    -> {comparison_path}")
