import pandas as pd
from pathlib import Path

RESULTS_FILE = Path("results/evaluation_results.csv")
HUMAN_FILE = Path("data/evaluation/human_evaluation_filled.csv")


def main():
    print("Loading evaluation results...")

    results = pd.read_csv(RESULTS_FILE)

    # Only include responses that were successfully evaluated
    successful = results[results["overall_score"].notna()].copy()

    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal evaluation samples: {len(results)}")
    print(f"Successfully evaluated: {len(successful)}")
    print(f"Failed/API-limit samples: {len(results) - len(successful)}")

    if successful.empty:
        print("\nNo successful evaluations found.")
        return

    metrics = [
        "relevance",
        "factual_correctness",
        "completeness",
        "groundedness",
        "professional_tone",
        "overall_score",
    ]

    print("\nAverage Scores")
    print("-" * 60)

    for metric in metrics:
        average = successful[metric].mean()
        print(f"{metric:25s}: {average:.2f} / 5")

    critical_errors = successful["critical_error"].astype(str).str.lower().eq("true").sum()
    critical_error_rate = (critical_errors / len(successful)) * 100

    print("\nSafety / Critical Errors")
    print("-" * 60)
    print(f"Critical errors: {critical_errors}")
    print(f"Critical error rate: {critical_error_rate:.2f}%")

    print("\nScore Distribution")
    print("-" * 60)

    excellent = (successful["overall_score"] >= 4.5).sum()
    good = (
        (successful["overall_score"] >= 3.5)
        & (successful["overall_score"] < 4.5)
    ).sum()
    needs_improvement = (successful["overall_score"] < 3.5).sum()

    print(f"Excellent (>= 4.5): {excellent}")
    print(f"Good (3.5 - 4.49): {good}")
    print(f"Needs improvement (< 3.5): {needs_improvement}")

    # Human validation
    if HUMAN_FILE.exists():
        print("\nHuman Validation")
        print("-" * 60)

        human = pd.read_csv(HUMAN_FILE)

        human_overall = pd.to_numeric(
            human["human_overall_score"],
            errors="coerce"
        )

        llm_overall = pd.to_numeric(
            human["llm_overall_score"],
            errors="coerce"
        )

        valid = human_overall.notna() & llm_overall.notna()

        if valid.any():
            agreement = (
                (abs(human_overall[valid] - llm_overall[valid]) <= 1)
                .mean()
                * 100
            )

            correlation = human_overall[valid].corr(
                llm_overall[valid]
            )

            print(f"Human validation samples: {valid.sum()}")
            print(f"Overall score correlation: {correlation:.3f}")
            print(f"Within ±1 point agreement: {agreement:.1f}%")

            print(
                "\nNote: Treat these as human-validation results only "
                "after the ratings have been independently reviewed by "
                "a human evaluator."
            )

    print("\n" + "=" * 60)
    print("SUMMARY COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()