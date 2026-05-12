import json
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


JUDGE_PROMPT = PromptTemplate.from_template(
    """
You are an impartial evaluator.

Question: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Return JSON only:

{{
    "winner": "A",
    "reason": "short explanation"
}}
"""
)


ABSOLUTE_PROMPT = PromptTemplate.from_template(
    """
Question: {question}

Answer:
{answer}

Score:

1. accuracy
2. relevance
3. conciseness
4. helpfulness

Return JSON only.
"""
)


def parse_judge_output(text):
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return {"winner": "tie", "reason": "parse_error"}


def pairwise_judge_with_swap(question, ans1, ans2, judge_llm):
    results = []

    p1 = JUDGE_PROMPT.format(
        question=question,
        answer_a=ans1,
        answer_b=ans2,
    )

    r1 = parse_judge_output(judge_llm.invoke(p1).content)

    results.append(r1)

    p2 = JUDGE_PROMPT.format(
        question=question,
        answer_a=ans2,
        answer_b=ans1,
    )

    r2 = parse_judge_output(judge_llm.invoke(p2).content)

    if r2["winner"] == "A":
        r2["winner"] = "B"
    elif r2["winner"] == "B":
        r2["winner"] = "A"

    results.append(r2)

    if results[0]["winner"] == results[1]["winner"]:
        return {
            "winner_after_swap": results[0]["winner"],
            "run1_winner": results[0]["winner"],
            "run2_winner": results[1]["winner"],
        }

    return {
        "winner_after_swap": "tie",
        "run1_winner": results[0]["winner"],
        "run2_winner": results[1]["winner"],
    }


def absolute_score(question, answer, judge_llm):
    prompt = ABSOLUTE_PROMPT.format(
        question=question,
        answer=answer,
    )

    out = judge_llm.invoke(prompt)

    parsed = parse_judge_output(out.content)

    dims = [
        "accuracy",
        "relevance",
        "conciseness",
        "helpfulness",
    ]

    parsed["overall"] = sum(parsed[d] for d in dims) / 4

    return parsed


def main():
    llm = ChatOpenAI(model="gpt-4o-mini")

    rows = []

    for i in range(30):
        result = pairwise_judge_with_swap(
            question=f"Question {i}",
            ans1="Answer A",
            ans2="Answer B",
            judge_llm=llm,
        )

        rows.append(
            {
                "question": f"Question {i}",
                **result,
            }
        )

    pd.DataFrame(rows).to_csv(
        "phase-b/pairwise_results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()