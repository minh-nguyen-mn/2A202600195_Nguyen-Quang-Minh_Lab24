import json
import pandas as pd

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_openai import ChatOpenAI


def dummy_rag_pipeline(question: str):
    answer = f"Generated answer for: {question}"
    contexts = ["Retrieved context chunk"]
    return answer, contexts


def run_eval():
    testset = pd.read_csv("phase-a/testset_v1.csv")

    rows = []

    for _, row in testset.iterrows():
        answer, contexts = dummy_rag_pipeline(row["question"])

        rows.append(
            {
                "question": row["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": row["ground_truth"],
            }
        )

    dataset = Dataset.from_list(rows)

    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ChatOpenAI(model="gpt-4o-mini"),
    )

    scores_df = scores.to_pandas()

    scores_df.to_csv("phase-a/ragas_results.csv", index=False)

    summary = {
        "faithfulness": float(scores["faithfulness"]),
        "answer_relevancy": float(scores["answer_relevancy"]),
        "context_precision": float(scores["context_precision"]),
        "context_recall": float(scores["context_recall"]),
    }

    with open("phase-a/ragas_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(summary)


if __name__ == "__main__":
    run_eval()