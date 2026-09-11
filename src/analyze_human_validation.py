import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path("data/evaluation/human_evaluation_filled.csv")

def correlation(x, y):
    return np.corrcoef(x, y)[0, 1]


def analyze_human_validation():
    print("Loading human evaluation data...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total evaluated responses: {len(df)}")

    dimensions = [
        ("relevance", "llm_relevance", "human_relevance"),
        ("factual_correctness", "llm_factual_correctness", "human_factual_correctness"),
        ("completeness", "llm_completeness", "human_completeness"),
        ("groundedness", "llm_groundedness", "human_groundedness"),
        ("professional_tone", "llm_professional_tone", "human_professional_tone"),
        ("overall_score", "llm_overall_score", "human_overall_score"),
    ]

    print("\n" + "=" * 60)
    print("LLM vs HUMAN EVALUATION")
    print("=" * 60)

    results = []

    for name, llm_col, human_col in dimensions:

        llm = pd.to_numeric(df[llm_col], errors="coerce")
        human = pd.to_numeric(df[human_col], errors="coerce")

        valid = llm.notna() & human.notna()

        llm_values = llm[valid].values
        human_values = human[valid].values

        if len(llm_values) == 0:
            continue

        corr = correlation(llm_values, human_values)

        within_one = np.mean(
            np.abs(llm_values - human_values) <= 1
        ) * 100

        mean_difference = np.mean(
            np.abs(llm_values - human_values)
        )

        results.append({
            "dimension": name,
            "correlation": corr,
            "within_1_point_percent": within_one,
            "mean_absolute_difference": mean_difference
        })

        print(f"\n{name}")
        print(f"Correlation: {corr:.3f}")
        print(f"Within ±1 point: {within_one:.1f}%")
        print(f"Mean absolute difference: {mean_difference:.3f}")

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not results_df.empty:
        print(
            f"\nAverage correlation: "
            f"{results_df['correlation'].mean():.3f}"
        )

        print(
            f"Average within ±1 point agreement: "
            f"{results_df['within_1_point_percent'].mean():.1f}%"
        )

        print(
            f"Average absolute difference: "
            f"{results_df['mean_absolute_difference'].mean():.3f}"
        )


if __name__ == "__main__":
    analyze_human_validation()