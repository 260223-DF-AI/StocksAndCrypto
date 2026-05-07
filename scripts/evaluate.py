"""
ResearchFlow — RAGAS Evaluation Pipeline

Loads a golden dataset and runs a formal RAGAS evaluation measuring
faithfulness, answer relevancy, and context precision.

Usage:
    python -m scripts.evaluate --golden-dataset ./data/golden_dataset.json
"""

import argparse
import json

from datasets import Dataset  # pip install datasets
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrock
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from agents.supervisor import build_supervisor_graph


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments."""
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation.")
    parser.add_argument(
        "--golden-dataset",
        type=str,
        required=True,
        help="Path to the golden dataset JSON file.",
    )
    return parser.parse_args()


def load_golden_dataset(filepath: str) -> list[dict]:
    """
    Load the golden dataset from a JSON file.

    Expected format: see data/golden_dataset.json for the schema.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_predictions(dataset: list[dict]) -> list[dict]:
    """
    Run each question through the ResearchFlow pipeline and collect predictions.

    TODO:
    - For each entry in the dataset, invoke the Supervisor graph.
    - Capture the generated answer and the retrieved contexts.
    - Return a list of dicts with keys: question, answer, contexts.
    """
    graph = build_supervisor_graph()
    out = []
    for i, entry in enumerate(dataset):
        config = {"configurable": {"thread_id": f"eval-{i}"}}
        try:
            result = graph.invoke(
                {"question": entry["question"], "user_id": "evaluator"},
                config=config,
            )
        except Exception as e:
            print(f"  [warn] entry {i} failed: {e}")
            out.append({"question": entry["question"], "answer": "", "contexts": []})
            continue
        analysis = result.get("analysis", {}) or {}
        contexts = [c["content"] for c in result.get("retrieved_chunks", [])]
        out.append({
            "question": entry["question"],
            "answer": analysis.get("answer", ""),
            "contexts": contexts,
            "ground_truth": entry["ground_truth_answer"],
        })
        print(f"  [{i+1}/{len(dataset)}] done")
    return out


def run_ragas_evaluation(predictions: list[dict], golden: list[dict]) -> dict:
    """
    Evaluate predictions against the golden dataset using RAGAS.

    TODO:
    - Construct a RAGAS Dataset from predictions and ground truth.
    - Evaluate with metrics: faithfulness, answer_relevancy, context_precision.
    - Return a dict of metric_name → score.
    """
    bedrock_llm = LangchainLLMWrapper(
        ChatBedrock(model_id="amazon.nova-pro-v1:0", region_name="us-east-1")
    )
    bedrock_embeddings = LangchainEmbeddingsWrapper(
        BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0", region_name="us-east-1"
        )
    )

    metrics = [faithfulness, answer_relevancy, context_precision]
    for m in metrics:
        m.llm = bedrock_llm
        if hasattr(m, "embeddings"):
            m.embeddings = bedrock_embeddings

    if len(predictions) != len(golden):
        raise ValueError(
            f"Prediction count ({len(predictions)}) does not match golden count ({len(golden)})."
        )

    rows = []
    for pred, gold in zip(predictions, golden):
        rows.append(
            {
                "question": pred["question"],
                "answer": pred["answer"],
                "contexts": pred["contexts"],
                "ground_truth": gold["ground_truth_answer"],
                "reference_contexts": gold.get("ground_truth_contexts", []),
            }
        )
    ds = Dataset.from_list(rows)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    # ragas<0.5 can expose scores as a list; newer versions may expose a dict.
    metric_names = ["faithfulness", "answer_relevancy", "context_precision"]
    scores_obj = getattr(result, "scores", None)
    if isinstance(scores_obj, dict):
        return {k: float(v) for k, v in scores_obj.items()}
    if isinstance(scores_obj, list):
        result_dict = {}
        for name, value in zip(metric_names, scores_obj):
            if isinstance(value, dict):
                # Extract numeric value from dict (try 'score' key first, then any numeric value)
                if "score" in value:
                    result_dict[name] = float(value["score"])
                else:
                    for v in value.values():
                        if isinstance(v, (int, float)):
                            result_dict[name] = float(v)
                            break
            else:
                result_dict[name] = float(value)
        return result_dict

    # Fallback for older/newer ragas result objects.
    if hasattr(result, "to_dict"):
        raw = result.to_dict()
        if isinstance(raw, dict):
            return {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}

    raise TypeError(f"Unsupported ragas result format: {type(result)!r}")


def main() -> None:
    """Orchestrate the evaluation pipeline."""
    load_dotenv()
    args = parse_args()

    golden = load_golden_dataset(args.golden_dataset)
    predictions = generate_predictions(golden)
    results = run_ragas_evaluation(predictions, golden)

    print("\n📊 RAGAS Evaluation Results:")
    print("-" * 40)
    for metric, score in results.items():
        print(f"  {metric:<25} {score:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    main()
