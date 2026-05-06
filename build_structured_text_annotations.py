import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import openai
from PIL import Image
from tqdm import tqdm

from data_analysis import analyze_dataset, load_statistics_map, save_statistics_outputs


SHAPE_OPTIONS = {
    "linear",
    "curved",
    "elongated",
    "circular",
    "irregular",
    "fragmented",
    "network-like",
    "spot-like",
    "patch-like",
    "diffuse",
    "unclear",
}
DIRECTION_OPTIONS = {"horizontal", "vertical", "diagonal", "none"}
SPATIAL_DISTRIBUTION_OPTIONS = {"isolated", "clustered", "scattered", "unclear"}
GRID_NAMES = [
    ["top-left", "top-center", "top-right"],
    ["middle-left", "center", "middle-right"],
    ["bottom-left", "bottom-center", "bottom-right"],
]


def configure_openai_client() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    openai.api_key = api_key

    api_base = os.getenv("OPENAI_API_BASE")
    if api_base:
        openai.api_base = api_base


def scale_category(scale_ratio: float) -> str:
    if scale_ratio < 0.01:
        return "tiny"
    if scale_ratio < 0.03:
        return "small"
    if scale_ratio < 0.10:
        return "medium"
    if scale_ratio <= 0.25:
        return "large"
    return "extensive"


def polarity_category(polarity_diff: float) -> str:
    if polarity_diff < -5:
        return "dark"
    if polarity_diff > 5:
        return "bright"
    return "neutral"


def saliency_category(saliency: float) -> str:
    if saliency < 10:
        return "low"
    if saliency <= 40:
        return "medium"
    return "high"


def number_category(component_count: Optional[int]) -> str:
    mapping = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    return mapping.get(component_count, "multiple")


def grid_name_to_bbox(image_width: int, image_height: int, row: int, col: int) -> Tuple[int, int, int, int]:
    cell_width = image_width / 3.0
    cell_height = image_height / 3.0
    left = int(round(col * cell_width))
    top = int(round(row * cell_height))
    right = int(round((col + 1) * cell_width)) if col < 2 else image_width
    bottom = int(round((row + 1) * cell_height)) if row < 2 else image_height
    return left, top, right, bottom


def compute_position(mask_path: str) -> List[List[object]]:
    if not os.path.exists(mask_path):
        return []

    with Image.open(mask_path) as image:
        grayscale = image.convert("L")
        width, height = grayscale.size
        pixels = grayscale.load()

        total_defect_pixels = 0
        cell_counts = []

        for row in range(3):
            for col in range(3):
                left, top, right, bottom = grid_name_to_bbox(width, height, row, col)
                cell_defect_pixels = 0
                for y in range(top, bottom):
                    for x in range(left, right):
                        if pixels[x, y] > 0:
                            cell_defect_pixels += 1
                cell_counts.append((GRID_NAMES[row][col], cell_defect_pixels))
                total_defect_pixels += cell_defect_pixels

    if total_defect_pixels == 0:
        return []

    return [
        [name, round(count / total_defect_pixels, 6)]
        for name, count in sorted(cell_counts, key=lambda item: item[1], reverse=True)
        if count > 0
    ]


def clean_model_response(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_model_attributes(text: str) -> Dict[str, str]:
    cleaned = clean_model_response(text)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return {str(key): str(value) for key, value in payload.items()}
    except json.JSONDecodeError:
        pass

    attributes: Dict[str, str] = {}
    for line in cleaned.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            attributes[key.strip()] = value.strip().strip(",")
    return attributes


def canonicalize_choice(value: str, allowed: Sequence[str], fallback: str) -> str:
    normalized = value.lower().strip().replace("_", "-")
    for candidate in allowed:
        if normalized == candidate:
            return candidate
    for candidate in allowed:
        if candidate in normalized:
            return candidate
    return fallback


def infer_shape_direction_spatial(description: str, component_count: int) -> Dict[str, str]:
    prompt = (
        "You are an expert in industrial steel surface defect analysis.\n\n"
        "Given a natural language description, infer three visual attributes.\n\n"
        "Allowed values:\n"
        f"Shape: {', '.join(sorted(SHAPE_OPTIONS))}\n"
        f"Direction: {', '.join(sorted(DIRECTION_OPTIONS))}\n"
        f"Spatial Distribution: {', '.join(sorted(SPATIAL_DISTRIBUTION_OPTIONS))}\n\n"
        "Rules:\n"
        "- Use only the allowed values.\n"
        "- Output JSON only.\n"
        "- If component_count is 1, Spatial Distribution must be isolated.\n"
        "- If the description does not reveal a clear direction, use none.\n\n"
        f"component_count: {component_count}\n"
        f"description: {description}\n\n"
        'Return JSON with keys "Shape", "Direction", and "Spatial Distribution".'
    )

    response = openai.ChatCompletion.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=120,
    )
    raw_text = response["choices"][0]["message"]["content"].strip()
    parsed = parse_model_attributes(raw_text)

    shape = canonicalize_choice(parsed.get("Shape", ""), sorted(SHAPE_OPTIONS), "unclear")
    direction = canonicalize_choice(parsed.get("Direction", ""), sorted(DIRECTION_OPTIONS), "none")
    spatial = canonicalize_choice(parsed.get("Spatial Distribution", ""), sorted(SPATIAL_DISTRIBUTION_OPTIONS), "unclear")
    if component_count == 1:
        spatial = "isolated"

    return {"Shape": shape, "Direction": direction, "Spatial Distribution": spatial}


def position_to_text(position_list: Sequence[Sequence[object]], top_k: int = 2) -> str:
    if not position_list:
        return "an unclear location"

    sorted_positions = sorted([item for item in position_list if len(item) >= 2], key=lambda item: float(item[1]), reverse=True)
    top_positions = [str(item[0]) for item in sorted_positions[:top_k]]
    if not top_positions:
        return "an unclear location"
    if len(top_positions) == 1:
        return top_positions[0]
    return f"{top_positions[0]} and {top_positions[1]}"


def build_template_sentence(attributes: Dict[str, object]) -> str:
    sentence = (
        f"A {attributes.get('Scale', 'unknown')} {attributes.get('Polarity', 'unknown')} "
        f"{attributes.get('Defect type', 'defect')} is observed on the steel surface. "
        f"It has a {attributes.get('Shape', 'unclear')} shape. "
    )
    direction = str(attributes.get("Direction", "none"))
    if direction != "none":
        sentence += f"It extends in a {direction} direction. "

    sentence += (
        f"The defects are {attributes.get('Spatial Distribution', 'unclear')}, "
        f"consisting of {attributes.get('Number of Defects', 'multiple')} region(s), "
        f"mainly located at {position_to_text(attributes.get('Position', []))}. "
        f"The defect exhibits {attributes.get('Saliency', 'unknown')} saliency."
    )
    return sentence.strip()


def build_structured_attributes(class_name: str, stats_row: Dict[str, object], description: str, mask_path: str) -> Dict[str, object]:
    scale_ratio = float(stats_row.get("scale_ratio", 0.0))
    polarity_diff = float(stats_row.get("polarity_diff", 0.0))
    saliency = float(stats_row.get("saliency", 0.0))
    component_count = int(float(stats_row.get("component_count", 0)))

    visual_attributes = infer_shape_direction_spatial(description, component_count)
    return {
        "Defect type": class_name,
        "Shape": visual_attributes["Shape"],
        "Direction": visual_attributes["Direction"],
        "Spatial Distribution": visual_attributes["Spatial Distribution"],
        "Number of Defects": number_category(component_count),
        "Position": compute_position(mask_path),
        "Scale": scale_category(scale_ratio),
        "Polarity": polarity_category(polarity_diff),
        "Saliency": saliency_category(saliency),
    }


def augment_annotations(input_json: str, mask_dir: str, output_json: str, statistics_map: Dict[str, Dict[str, object]]) -> None:
    with open(input_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    results: List[Dict[str, object]] = []
    for item in tqdm(data, desc="Building structured annotations"):
        image_name = item.get("image_name")
        if not image_name:
            continue

        class_name = item.get("class_name") or "defect"
        description = item.get("natural_language_description") or item.get("final_description") or ""
        stats_row = statistics_map.get(image_name, {"mask_name": f"{Path(image_name).stem}.png"})
        mask_name = str(stats_row.get("mask_name", f"{Path(image_name).stem}.png"))
        mask_path = os.path.join(mask_dir, mask_name)

        structured_attributes = build_structured_attributes(class_name, stats_row, description, mask_path)
        results.append(
            {
                "image_name": image_name,
                "class_name": class_name,
                "natural_language_description": description,
                "structured_attributes": structured_attributes,
                "template_sentence": build_template_sentence(structured_attributes),
            }
        )

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build structured attributes and template sentences from natural-language descriptions and segmentation masks."
    )
    parser.add_argument("--input-json", required=True, help="Input JSON containing natural-language descriptions.")
    parser.add_argument("--image-dir", required=True, help="Directory containing source images.")
    parser.add_argument("--mask-dir", required=True, help="Directory containing binary segmentation masks.")
    parser.add_argument("--output-json", required=True, help="Output JSON path.")
    parser.add_argument(
        "--statistics-output-dir",
        default="",
        help="Optional directory for intermediate CSV, summary JSON, and histogram plots.",
    )
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for histogram outputs.")
    parser.add_argument("--min-component-area", type=int, default=1, help="Ignore connected components smaller than this area.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_openai_client()

    rows, summary = analyze_dataset(
        image_dir=Path(args.image_dir),
        mask_dir=Path(args.mask_dir),
        bins=args.bins,
        min_component_area=args.min_component_area,
    )
    if args.statistics_output_dir:
        save_statistics_outputs(rows, summary, Path(args.statistics_output_dir), bins=args.bins)

    augment_annotations(
        input_json=args.input_json,
        mask_dir=args.mask_dir,
        output_json=args.output_json,
        statistics_map=load_statistics_map(rows),
    )


if __name__ == "__main__":
    main()
