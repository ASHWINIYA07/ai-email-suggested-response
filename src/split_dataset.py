import pandas as pd
from pathlib import Path


INPUT_FILE = Path("data/processed/emails.csv")

TRAIN_FILE = Path("data/processed/knowledge_base.csv")
EVAL_FILE = Path("data/evaluation/evaluation_set.csv")


def split_dataset():

    print("Loading cleaned dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total records: {len(df)}")

    # Shuffle the dataset in a reproducible way
    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Use 80% for the retrieval knowledge base
    split_index = int(len(df) * 0.8)

    knowledge_base = df.iloc[:split_index].copy()
    evaluation_set = df.iloc[split_index:].copy()

    # Create output directory
    EVAL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the two datasets
    knowledge_base.to_csv(
        TRAIN_FILE,
        index=False
    )

    evaluation_set.to_csv(
        EVAL_FILE,
        index=False
    )

    print("\nDataset split complete!")

    print(f"Knowledge base: {len(knowledge_base)}")
    print(f"Evaluation set: {len(evaluation_set)}")

    print(f"\nKnowledge base saved to:")
    print(TRAIN_FILE)

    print(f"\nEvaluation set saved to:")
    print(EVAL_FILE)


if __name__ == "__main__":
    split_dataset()