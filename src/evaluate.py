import os
import json
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from generator import generate_reply
from retriever import EmailRetriever
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EVALUATION_FILE = "data/evaluation/golden_set.csv"
OUTPUT_FILE = "results/evaluation_results.csv"


def evaluate_reply(email, generated_reply, reference_reply):
    print("Evaluating response...")

    prompt = f"""
You are an expert evaluator of customer-support email responses.

Evaluate the AI-generated response against the incoming customer
email and the historical/reference response.

IMPORTANT:
- Do not require the AI response to match the reference word-for-word.
- Judge whether the AI response is actually useful and appropriate.
- Do not reward the response simply because it is similar to the reference.
- Penalize unsupported claims, invented facts, and incorrect information.

Give a score from 1 to 5 for each category:

1. Relevance
Does the response directly address the customer's problem?

2. Factual Correctness
Does the response avoid incorrect or unsupported claims?

3. Completeness
Does it cover the important information needed to help the customer?

4. Groundedness
Is the response consistent with the information available in the
customer email and reference response?

5. Professional Tone
Is the response clear, polite, concise, and professional?

Also determine whether there is a critical error.

A critical error includes things such as:
- dangerous or seriously incorrect advice
- claiming an action was completed when it was not
- inventing important facts
- completely misunderstanding the customer's problem

Return ONLY valid JSON in this format:

{{
    "relevance": 1,
    "factual_correctness": 1,
    "completeness": 1,
    "groundedness": 1,
    "professional_tone": 1,
    "critical_error": false,
    "reason": "Short explanation"
}}

INCOMING CUSTOMER EMAIL:
{email}

REFERENCE RESPONSE:
{reference_reply}

AI-GENERATED RESPONSE:
{generated_reply}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    result_text = response.output_text.strip()

    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        print("Warning: evaluator returned invalid JSON.")
        print(result_text)
        result = {
            "relevance": None,
            "factual_correctness": None,
            "completeness": None,
            "groundedness": None,
            "professional_tone": None,
            "critical_error": None,
            "reason": result_text
        }

    return result


if __name__ == "__main__":
    
    print("Loading evaluation data...")

    df = pd.read_csv(EVALUATION_FILE)

    # Use only English emails
    df = df[df["language"] == "en"].copy()

    print(f"Evaluation emails available: {len(df)}")

    # ---------------------------------------------------------
    # Load previous results if they already exist
    # ---------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):
        previous_results_df = pd.read_csv(OUTPUT_FILE)

        # Keep only successfully evaluated emails
        completed_ids = set(
            previous_results_df.loc[
                previous_results_df["relevance"].notna(),
                "email_id"
            ]
        )

        results = previous_results_df.to_dict("records")

        print(
            f"Found {len(completed_ids)} previously completed emails."
        )

    else:
        completed_ids = set()
        results = []

        print("No previous results found.")

    # ---------------------------------------------------------
    # Load retriever
    # ---------------------------------------------------------

    print("\nLoading retriever...")

    retriever = EmailRetriever()

    # ---------------------------------------------------------
    # Process remaining emails
    # ---------------------------------------------------------

    for index, row in df.iterrows():

        email_id = row["email_id"]

        # Skip emails already evaluated successfully
        if email_id in completed_ids:
            continue

        print("\n" + "=" * 70)
        print(f"Processing email {index + 1} of {len(df)}")
        print(f"Email ID: {email_id}")
        print("=" * 70)

        email = f"""
Subject: {row["subject"]}

{row["body"]}
"""

        reference_reply = row["answer"]

        try:

            # -------------------------------------------------
            # Generate AI response
            # -------------------------------------------------

            print("Generating AI reply...")

            generated_reply, retrieved_cases = generate_reply(
                email,
                retriever
            )

            # -------------------------------------------------
            # Evaluate AI response
            # -------------------------------------------------

            print("Evaluating AI reply...")

            evaluation = evaluate_reply(
                email,
                generated_reply,
                reference_reply
            )

            # -------------------------------------------------
            # Store result
            # -------------------------------------------------

            result = {
                "email_id": email_id,
                "subject": row["subject"],
                "generated_reply": generated_reply,
                "reference_reply": reference_reply,
                "relevance": evaluation.get("relevance"),
                "factual_correctness": evaluation.get(
                    "factual_correctness"
                ),
                "completeness": evaluation.get("completeness"),
                "groundedness": evaluation.get("groundedness"),
                "professional_tone": evaluation.get(
                    "professional_tone"
                ),
                "critical_error": evaluation.get(
                    "critical_error"
                ),
                "reason": evaluation.get("reason")
            }

            results.append(result)

            # Mark as completed
            completed_ids.add(email_id)

            # -------------------------------------------------
            # Save immediately
            # -------------------------------------------------

            pd.DataFrame(results).to_csv(
                OUTPUT_FILE,
                index=False
            )

            print("\nScores:")
            print(
                f"Relevance: "
                f"{evaluation.get('relevance')}/5"
            )
            print(
                f"Factual Correctness: "
                f"{evaluation.get('factual_correctness')}/5"
            )
            print(
                f"Completeness: "
                f"{evaluation.get('completeness')}/5"
            )
            print(
                f"Groundedness: "
                f"{evaluation.get('groundedness')}/5"
            )
            print(
                f"Professional Tone: "
                f"{evaluation.get('professional_tone')}/5"
            )
            print(
                f"Critical Error: "
                f"{evaluation.get('critical_error')}"
            )

            print(
                f"Progress saved: "
                f"{len(completed_ids)} emails completed."
            )

        except Exception as e:

            error_message = str(e)

            print(
                f"\nERROR processing {email_id}: "
                f"{error_message}"
            )

            # -------------------------------------------------
            # Stop immediately when API rate limit is reached
            # -------------------------------------------------

            if "429" in error_message or "rate_limit" in error_message:

                print("\nAPI rate limit reached.")
                print(
                    "Stopping evaluation so we don't waste "
                    "additional requests."
                )

                break

            # -------------------------------------------------
            # Save other errors and continue
            # -------------------------------------------------

            results.append({
                "email_id": email_id,
                "subject": row["subject"],
                "generated_reply": "",
                "reference_reply": reference_reply,
                "relevance": None,
                "factual_correctness": None,
                "completeness": None,
                "groundedness": None,
                "professional_tone": None,
                "critical_error": None,
                "reason": f"Evaluation failed: {error_message}"
            })

            pd.DataFrame(results).to_csv(
                OUTPUT_FILE,
                index=False
            )

    # ---------------------------------------------------------
    # Calculate metrics for successful evaluations
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("CURRENT EVALUATION SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)

    score_columns = [
        "relevance",
        "factual_correctness",
        "completeness",
        "groundedness",
        "professional_tone"
    ]

    for column in score_columns:
        results_df[column] = pd.to_numeric(
            results_df[column],
            errors="coerce"
        )

    successful = results_df[
        results_df["relevance"].notna()
    ].copy()

    print(
        f"\nSuccessful evaluations: "
        f"{len(successful)}"
    )

    print("\nAverage Scores:")

    for column in score_columns:

        average = successful[column].mean()

        print(
            f"{column}: "
            f"{average:.2f}/5"
        )

    # ---------------------------------------------------------
    # Weighted overall score
    # ---------------------------------------------------------

    successful["overall_score"] = (
        successful["relevance"] * 0.25
        + successful["factual_correctness"] * 0.30
        + successful["completeness"] * 0.20
        + successful["groundedness"] * 0.15
        + successful["professional_tone"] * 0.10
    )

    print(
        f"\nOverall Score: "
        f"{successful['overall_score'].mean():.2f}/5"
    )

    # ---------------------------------------------------------
    # Critical error rate
    # ---------------------------------------------------------

    critical_errors = (
        successful["critical_error"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )

    if len(successful) > 0:

        critical_error_rate = (
            critical_errors / len(successful)
        ) * 100

        print(
            f"Critical Error Rate: "
            f"{critical_error_rate:.2f}%"
        )

    # ---------------------------------------------------------
    # Save final/current results
    # ---------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nResults saved to: "
        f"{OUTPUT_FILE}"
    )