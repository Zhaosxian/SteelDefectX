import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import openai
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


DEFAULT_MODEL_NAME = "gpt-4o"
DEFAULT_DIMENSIONS = ("shape", "size", "depth", "position", "contrast")
DIMENSION_KEYWORDS = {
    "shape": ["shape", "round", "irregular", "linear", "circular"],
    "size": ["small", "large", "span", "covers", "extends"],
    "depth": ["shallow", "deep", "surface-level", "indented", "raised"],
    "position": ["top", "bottom", "left", "right", "center", "corner"],
    "contrast": ["contrast", "noticeable", "faint", "sharp", "blend"],
}
PROMPT_A = (
    "Describe the steel surface defect using short, clear sentences. "
    "Focus on visual features such as appearance, shape, size, depth, position, and contrast. "
    "Avoid speculation or vague language."
)
PROMPT_B_QUESTIONS = [
    "What does the defect on the steel surface look like in the image?",
    "What is the shape of the defect?",
    "What is its approximate size relative to the image or surface?",
    "Does the defect appear shallow or deep?",
    "Where is it located within the image?",
    "How does the defect contrast with the background surface?",
]

embedding_model = None


def configure_openai_client(api_base: str = "") -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    openai.api_key = api_key

    resolved_api_base = api_base or os.getenv("OPENAI_API_BASE", "")
    if resolved_api_base:
        openai.base_url = resolved_api_base


def load_embedding_model(model_path_or_name: str) -> SentenceTransformer:
    if not hasattr(torch.backends, "mps"):
        torch.backends.mps = type("FakeMPS", (), {"is_available": staticmethod(lambda: False)})()
    return SentenceTransformer(model_path_or_name)


def is_incomplete_sentence(text: str) -> bool:
    return not bool(re.search(r"[.!?]$", text.strip()))


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def call_openai_image(prompt: str, image_b64: str, model_name: str) -> str:
    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
        temperature=0.9,
        top_p=0.9,
        max_tokens=80,
    )
    return response.choices[0].message.content.strip()


def select_top_k_diverse(descriptions: List[str], top_k: int = 3, similarity_threshold: float = 0.9) -> List[str]:
    if not descriptions:
        return []

    embeddings = embedding_model.encode(descriptions)
    selected = [descriptions[0]]
    selected_embeddings = [embeddings[0]]

    for index in range(1, len(descriptions)):
        max_similarity = max(util.cos_sim(embeddings[index], emb)[0] for emb in selected_embeddings)
        if max_similarity < similarity_threshold:
            selected.append(descriptions[index])
            selected_embeddings.append(embeddings[index])
        if len(selected) >= top_k:
            break

    return selected


def generate_prompt_b_candidates(image_b64: str, model_name: str, n_samples: int) -> List[str]:
    return [call_openai_image(PROMPT_A, image_b64, model_name) for _ in range(n_samples)]


def generate_prompt_a_candidate(image_b64: str, model_name: str) -> str:
    answers = [call_openai_image(question, image_b64, model_name) for question in PROMPT_B_QUESTIONS]
    return " ".join(answers)


def get_dim_bitcode(text: str) -> str:
    lowered = text.lower()
    return "".join(
        "1" if any(keyword in lowered for keyword in DIMENSION_KEYWORDS[dimension]) else "0"
        for dimension in DEFAULT_DIMENSIONS
    )


def compute_score(dim_bitcode: str, sim_score: float, w1: float = 0.6, w2: float = 0.4) -> float:
    dim_score = dim_bitcode.count("1")
    return w1 * (dim_score / len(DEFAULT_DIMENSIONS)) + w2 * (1 - sim_score)


def should_trigger_prompt_a(candidate_list: List[Dict[str, object]], dim_thresh: int = 4) -> bool:
    return bool(candidate_list) and all(candidate["dim_score"] < dim_thresh for candidate in candidate_list)


def rank_candidates(candidate_texts: List[str]) -> List[Dict[str, object]]:
    if not candidate_texts:
        return []

    embeddings = embedding_model.encode(candidate_texts, convert_to_tensor=True)
    outputs: List[Dict[str, object]] = []

    for index, text in enumerate(candidate_texts):
        bitcode = get_dim_bitcode(text)
        sim_sum = sum(util.cos_sim(embeddings[index], emb).item() for j, emb in enumerate(embeddings) if j != index)
        sim_score = sim_sum / max(1, len(embeddings) - 1)
        outputs.append(
            {
                "text": text,
                "dim_coverage": bitcode,
                "dim_score": bitcode.count("1"),
                "sim_score": round(sim_score, 4),
                "final_score": round(compute_score(bitcode, sim_score), 4),
                "incomplete": is_incomplete_sentence(text),
            }
        )

    outputs.sort(key=lambda item: (item["incomplete"], -item["final_score"]))
    return outputs


def process_image(image_path: str, model_name: str, n_samples: int) -> Dict[str, object]:
    image_b64 = image_to_base64(image_path)
    raw_candidates = generate_prompt_b_candidates(image_b64, model_name, n_samples)
    diverse_candidates = select_top_k_diverse(raw_candidates, top_k=3)
    ranked_candidates = rank_candidates(diverse_candidates)

    if should_trigger_prompt_a(ranked_candidates):
        augmented = generate_prompt_a_candidate(image_b64, model_name)
        augmented_bitcode = get_dim_bitcode(augmented)
        ranked_candidates.append(
            {
                "text": augmented,
                "dim_coverage": augmented_bitcode,
                "dim_score": augmented_bitcode.count("1"),
                "sim_score": 0.5,
                "final_score": round(compute_score(augmented_bitcode, 0.5), 4),
                "incomplete": is_incomplete_sentence(augmented),
            }
        )
        ranked_candidates.sort(key=lambda item: (item["incomplete"], -item["final_score"]))

    if not ranked_candidates:
        raise ValueError(f"No valid description candidates were generated for {image_path}")

    return {
        "image_name": os.path.basename(image_path),
        "class_name": os.path.basename(os.path.dirname(image_path)),
        "natural_language_description": ranked_candidates[0]["text"],
    }


def collect_image_paths(input_dir: str) -> List[str]:
    image_paths: List[str] = []
    for root, _, file_names in os.walk(input_dir):
        for file_name in file_names:
            if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(os.path.join(root, file_name))
    return sorted(image_paths)


def generate_descriptions(
    input_dir: str,
    output_json: str,
    model_name: str,
    embedding_model_path: str,
    n_samples: int,
) -> None:
    global embedding_model

    embedding_model = load_embedding_model(embedding_model_path)
    image_paths = collect_image_paths(input_dir)
    results: List[Dict[str, object]] = []

    for image_path in tqdm(image_paths, desc="Generating descriptions"):
        try:
            results.append(process_image(image_path, model_name=model_name, n_samples=n_samples))
            with open(output_json, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"Error processing {image_path}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample-level natural-language descriptions for defect images.")
    parser.add_argument("--input-dir", required=True, help="Directory containing defect images grouped by class.")
    parser.add_argument("--output-json", required=True, help="Output JSON path.")
    parser.add_argument(
        "--embedding-model-path",
        default=str(Path(__file__).resolve().parent / "saved_model"),
        help="SentenceTransformer checkpoint path or model name.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Vision-language model used for captioning.")
    parser.add_argument("--n-samples", type=int, default=4, help="Number of prompt-B samples per image.")
    parser.add_argument(
        "--api-base",
        default="",
        help="Optional custom OpenAI-compatible API base URL. Can also be set with OPENAI_API_BASE.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_openai_client(api_base=args.api_base)
    generate_descriptions(
        input_dir=args.input_dir,
        output_json=args.output_json,
        model_name=args.model_name,
        embedding_model_path=args.embedding_model_path,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()
