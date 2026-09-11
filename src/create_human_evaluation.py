import pandas as pd
from pathlib import Path

INPUT_FILE = Path("results/evaluation_results.csv")
OUTPUT_FILE = Path("data/evaluation/human_evaluation.csv")


def create_human_evaluation():

    print("Loading automated evaluation results...")

    df = pd.read_csv(INPUT_FILE)

    # Keep only successfully evaluated responses
    df = df[df["relevance"].notna()].copy()

    print(f"Successful responses available: {len(df)}")

    # Create columns for human evaluation
    human_df = df[
        [
            "email_id",
            "subject",
            "generated_reply",
            "reference_reply",
            "relevance",
            "factual_correctness",
            "completeness",
            "groundedness",
            "professional_tone",
            "overall_score"
        ]
    ].copy()

    # Rename automated scores so they are clearly identified
    human_df = human_df.rename(
        columns={
            "relevance": "llm_relevance",
            "factual_correctness": "llm_factual_correctness",
            "completeness": "llm_completeness",
            "groundedness": "llm_groundedness",
            "professional_tone": "llm_professional_tone",
            "overall_score": "llm_overall_score"
        }
    )

    # Empty columns for human scores
    human_df["human_relevance"] = ""
    human_df["human_factual_correctness"] = ""
    human_df["human_completeness"] = ""
    human_df["human_groundedness"] = ""
    human_df["human_professional_tone"] = ""
    human_df["human_critical_error"] = ""
    human_df["human_overall_score"] = ""
    human_df["human_reason"] = ""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    human_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nHuman evaluation sheet created!")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nColumns to fill manually:")
    print("- human_relevance")
    print("- human_factual_correctness")
    print("- human_completeness")
    print("- human_groundedness")
    print("- human_professional_tone")
    print("- human_critical_error")
    print("- human_overall_score")
    print("- human_reason")


if __name__ == "__main__":
    create_human_evaluation()