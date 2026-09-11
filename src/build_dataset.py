import pandas as pd
from pathlib import Path


RAW_FILE = Path("data/raw/dataset-tickets-multi-lang-4-20k.csv")
OUTPUT_FILE = Path("data/processed/emails.csv")


def build_dataset():
    print("Loading dataset...")

    df = pd.read_csv(RAW_FILE)

    print(f"Original rows: {len(df)}")

    # Keep only records that contain both
    # an incoming email and an agent response.
    df = df.dropna(subset=["body", "answer"])

    # Convert text columns to strings.
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()

    # Remove empty emails or responses.
    df = df[
        (df["body"] != "") &
        (df["answer"] != "")
    ]

    # Remove duplicate email/response pairs.
    df = df.drop_duplicates(
        subset=["body", "answer"]
    )

    # Create a simple unique ID.
    df.insert(
        0,
        "email_id",
        [f"email_{i:05d}" for i in range(1, len(df) + 1)]
    )

    # Keep only the fields needed for our project.
    columns = [
        "email_id",
        "subject",
        "body",
        "answer",
        "type",
        "queue",
        "priority",
        "language",
        "tag_1",
        "tag_2",
        "tag_3",
        "tag_4",
        "tag_5",
        "tag_6",
        "tag_7",
        "tag_8"
    ]

    df = df[columns]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Clean rows: {len(df)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_dataset()