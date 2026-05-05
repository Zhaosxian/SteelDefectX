import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PERCENTILES = [5, 10, 20, 50, 80, 90, 95]


def list_files_by_stem(directory: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            files[path.stem] = path
    return files


def read_gray_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def read_binary_mask(mask_path: Path) -> np.ndarray:
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    return (mask_gray > 127).astype(np.uint8)


def compute_scale(mask: np.ndarray) -> float:
    return int(mask.sum()) / float(mask.size)


def compute_polarity_saliency(gray: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, float]:
    defect = gray[mask == 1]
    background = gray[mask == 0]

    mean_defect = float(np.mean(defect)) if defect.size > 0 else np.nan
    mean_background = float(np.mean(background)) if background.size > 0 else np.nan

    if np.isnan(mean_defect) or np.isnan(mean_background):
        return mean_defect, mean_background, np.nan, np.nan

    diff = mean_defect - mean_background
    return mean_defect, mean_background, diff, abs(diff)


def compute_component_statistics(mask: np.ndarray, min_component_area: int = 1) -> Tuple[int, float]:
    total_mask_area = int(mask.sum())
    if total_mask_area == 0:
        return 0, 0.0

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    component_areas = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            component_areas.append(area)

    if not component_areas:
        return 0, 0.0

    component_count = len(component_areas)
    largest_ratio = max(component_areas) / float(sum(component_areas))
    return component_count, largest_ratio


def finite_values(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def summarize_metric(values: Sequence[float], percentiles: Sequence[int] = PERCENTILES, bins: int = 50) -> Dict[str, object]:
    arr = finite_values(values)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "percentiles": {str(p): None for p in percentiles},
            "histogram": {"bin_edges": [], "counts": []},
        }

    counts, edges = np.histogram(arr, bins=bins)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "percentiles": {str(p): float(np.percentile(arr, p)) for p in percentiles},
        "histogram": {"bin_edges": edges.astype(float).tolist(), "counts": counts.astype(int).tolist()},
    }


def save_histogram(values: Sequence[float], title: str, xlabel: str, out_path: Path, bins: int = 50) -> None:
    arr = finite_values(values)
    plt.figure(figsize=(8, 5))
    if arr.size > 0:
        plt.hist(arr, bins=bins, color="#4C72B0", edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    fieldnames = [
        "image_name",
        "mask_name",
        "scale_ratio",
        "mean_defect",
        "mean_background",
        "polarity_diff",
        "saliency",
        "component_count",
        "largest_component_ratio",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_dataset(
    image_dir: Path,
    mask_dir: Path,
    bins: int = 50,
    min_component_area: int = 1,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    image_map = list_files_by_stem(image_dir)
    mask_map = list_files_by_stem(mask_dir)

    common_stems = sorted(set(image_map.keys()) & set(mask_map.keys()))
    missing_masks = sorted(set(image_map.keys()) - set(mask_map.keys()))
    missing_images = sorted(set(mask_map.keys()) - set(image_map.keys()))

    rows: List[Dict[str, object]] = []
    scale_values: List[float] = []
    polarity_values: List[float] = []
    saliency_values: List[float] = []
    comp_count_values: List[int] = []
    largest_ratio_values: List[float] = []

    for index, stem in enumerate(common_stems, start=1):
        image_path = image_map[stem]
        mask_path = mask_map[stem]

        try:
            gray = read_gray_image(image_path)
            mask = read_binary_mask(mask_path)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            continue

        if gray.shape != mask.shape:
            print(f"[WARN] Shape mismatch, skip: {image_path.name} vs {mask_path.name}")
            continue

        scale_ratio = compute_scale(mask)
        mean_defect, mean_background, polarity_diff, saliency = compute_polarity_saliency(gray, mask)
        component_count, largest_ratio = compute_component_statistics(mask, min_component_area=min_component_area)

        row = {
            "image_name": image_path.name,
            "mask_name": mask_path.name,
            "scale_ratio": float(scale_ratio),
            "mean_defect": float(mean_defect) if np.isfinite(mean_defect) else np.nan,
            "mean_background": float(mean_background) if np.isfinite(mean_background) else np.nan,
            "polarity_diff": float(polarity_diff) if np.isfinite(polarity_diff) else np.nan,
            "saliency": float(saliency) if np.isfinite(saliency) else np.nan,
            "component_count": int(component_count),
            "largest_component_ratio": float(largest_ratio),
        }
        rows.append(row)

        scale_values.append(scale_ratio)
        polarity_values.append(polarity_diff)
        saliency_values.append(saliency)
        comp_count_values.append(component_count)
        largest_ratio_values.append(largest_ratio)

        if index % 500 == 0:
            print(f"Processed {index}/{len(common_stems)}")

    summary = {
        "input": {
            "image_dir": str(image_dir),
            "mask_dir": str(mask_dir),
            "num_images": len(image_map),
            "num_masks": len(mask_map),
            "num_pairs": len(common_stems),
            "num_processed": len(rows),
            "missing_masks_count": len(missing_masks),
            "missing_images_count": len(missing_images),
            "missing_masks_examples": missing_masks[:20],
            "missing_images_examples": missing_images[:20],
        },
        "scale": summarize_metric(scale_values, bins=bins),
        "polarity_diff": summarize_metric(polarity_values, bins=bins),
        "saliency": summarize_metric(saliency_values, bins=bins),
        "component_count": summarize_metric(comp_count_values, bins=bins),
        "largest_component_ratio": summarize_metric(largest_ratio_values, bins=bins),
    }
    return rows, summary


def save_statistics_outputs(rows: List[Dict[str, object]], summary: Dict[str, object], output_dir: Path, bins: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / "per_image_statistics.csv"
    out_json = output_dir / "global_summary.json"

    write_csv(rows, out_csv)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    save_histogram([float(row["scale_ratio"]) for row in rows], "Scale Distribution", "Mask Area Ratio", output_dir / "hist_scale_ratio.png", bins=bins)
    save_histogram([float(row["polarity_diff"]) for row in rows], "Polarity Distribution", "Mean Defect - Mean Background", output_dir / "hist_polarity_diff.png", bins=bins)
    save_histogram([float(row["saliency"]) for row in rows], "Saliency Distribution", "Absolute Mean Difference", output_dir / "hist_saliency.png", bins=bins)
    save_histogram([int(row["component_count"]) for row in rows], "Component Count Distribution", "Number of Connected Components", output_dir / "hist_component_count.png", bins=bins)
    save_histogram([float(row["largest_component_ratio"]) for row in rows], "Largest Component Ratio Distribution", "Largest Component Area / Total Mask Area", output_dir / "hist_largest_component_ratio.png", bins=bins)


def load_statistics_map(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(row["image_name"]): row for row in rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute deterministic image-mask statistics for SteelDefectX.")
    parser.add_argument("--image-dir", required=True, help="Directory containing source images.")
    parser.add_argument("--mask-dir", required=True, help="Directory containing binary masks.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV, summary JSON, and histograms.")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for histogram outputs.")
    parser.add_argument("--min-component-area", type=int, default=1, help="Ignore connected components smaller than this area.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, summary = analyze_dataset(
        image_dir=Path(args.image_dir),
        mask_dir=Path(args.mask_dir),
        bins=args.bins,
        min_component_area=args.min_component_area,
    )
    save_statistics_outputs(rows, summary, Path(args.output_dir), bins=args.bins)


if __name__ == "__main__":
    main()
