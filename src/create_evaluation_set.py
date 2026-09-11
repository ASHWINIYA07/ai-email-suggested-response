import pandas as pd
from pathlib import Path


INPUT_FILE = Path(
    "data/evaluation/evaluation_set.csv"
)

OUTPUT_FILE = Path(
    "data/evaluation/golden_set.csv"
)


def create_evaluation_set():

    print("Loading evaluation data...")

    df = pd.read_csv(INPUT_FILE)

    # Use English emails only
    df = df[
        df["language"] == "en"
    ].copy()

    print(
        f"Available English evaluation emails: "
        f"{len(df)}"
    )

    # We want 200 examples.
    # Stratify using queue + priority + type.
    df["stratum"] = (
        df["queue"].fillna("unknown")
        + "_"
        + df["priority"].fillna("unknown")
        + "_"
        + df["type"].fillna("unknown")
    )

    # Sample proportionally from each stratum.
    golden_set = (
        df.groupby(
            "stratum",
            group_keys=False
        )
        .apply(
            lambda group: group.sample(
                n=max(
                    1,
                    round(
                        len(group)
                        / len(df)
                        * 200
                    )
                ),
                random_state=42
            )
        )
    )

    # If proportional sampling produced
    # slightly more than 200 rows, trim it.
    if len(golden_set) > 200:

        golden_set = golden_set.sample(
            n=200,
            random_state=42
        )

    # If fewer than 200 rows were selected,
    # fill from remaining records.
    if len(golden_set) < 200:

        remaining = df[
            ~df["email_id"].isin(
                golden_set["email_id"]
            )
        ]

        additional = remaining.sample(
            n=200 - len(golden_set),
            random_state=42
        )

        golden_set = pd.concat(
            [golden_set, additional]
        )

    # Remove helper column
        if "stratum" in golden_set.columns:
            golden_set = golden_set.drop(columns=["stratum"])

    golden_set = golden_set.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    golden_set.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nGolden evaluation set created:"
        f" {len(golden_set)} emails"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("\nDistribution:")
    print(
        golden_set[
            ["queue", "priority", "type"]
        ].value_counts()
    )


if __name__ == "__main__":
    create_evaluation_set()