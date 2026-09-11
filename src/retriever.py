import pandas as pd
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer


KNOWLEDGE_BASE = Path(
    "data/processed/knowledge_base.csv"
)

INDEX_FILE = Path(
    "data/indexes/email_index.faiss"
)

EMBEDDINGS_FILE = Path(
    "data/indexes/embeddings.npy"
)


class EmailRetriever:

    def __init__(self):

        print("Loading knowledge base...")

        self.df = pd.read_csv(KNOWLEDGE_BASE)

        # Initial experiment: English only
        self.df = self.df[
            self.df["language"] == "en"
        ].reset_index(drop=True)

        print(
            f"Using {len(self.df)} English "
            "historical emails."
        )

        # Load embedding model
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # Create directory for saved indexes
        INDEX_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if INDEX_FILE.exists():

            print("Loading existing FAISS index...")

            self.index = faiss.read_index(
                str(INDEX_FILE)
            )

            print(
                f"Loaded index with "
                f"{self.index.ntotal} emails."
            )

        else:

            print(
                "No existing index found."
            )

            print(
                "Creating embeddings..."
            )

            self.texts = (
                self.df["subject"].fillna("")
                + "\n"
                + self.df["body"].fillna("")
            ).tolist()

            embeddings = self.model.encode(
                self.texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )

            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)

            dimension = embeddings.shape[1]

            self.index = faiss.IndexFlatIP(
                dimension
            )

            self.index.add(embeddings)

            # Save index
            faiss.write_index(
                self.index,
                str(INDEX_FILE)
            )

            # Also save embeddings for reproducibility
            np.save(
                EMBEDDINGS_FILE,
                embeddings
            )

            print(
                f"Saved FAISS index to "
                f"{INDEX_FILE}"
            )


    def search(
        self,
        email,
        top_k=3
    ):

        query_embedding = self.model.encode(
            [email],
            convert_to_numpy=True
        )

        faiss.normalize_L2(
            query_embedding
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            row = self.df.iloc[index]

            results.append({
                "email_id": row["email_id"],
                "subject": row["subject"],
                "body": row["body"],
                "answer": row["answer"],
                "similarity": float(score)
            })

        return results


if __name__ == "__main__":

    retriever = EmailRetriever()

    test_email = """
    My computer keeps shutting down unexpectedly.
    This has happened several times today.
    """

    results = retriever.search(
        test_email,
        top_k=3
    )

    print(
        "\nSimilar historical emails:\n"
    )

    for result in results:

        print("=" * 60)

        print(
            f"Email ID: {result['email_id']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.3f}"
        )

        print(
            f"Subject: "
            f"{result['subject']}"
        )

        print(
            f"Customer: "
            f"{result['body']}"
        )

        print(
            f"Historical response: "
            f"{result['answer']}"
        )