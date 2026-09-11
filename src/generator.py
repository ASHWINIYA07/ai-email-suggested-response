import os

from dotenv import load_dotenv
from openai import OpenAI

from retriever import EmailRetriever


# Load variables from .env
load_dotenv()

# Create OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def generate_reply(email, retriever):

    # 1. Find similar historical emails
    results = retriever.search(
        email,
        top_k=3
    )

    # 2. Prepare historical examples
    historical_cases = []

    for i, result in enumerate(results, start=1):

        historical_cases.append(
            f"""
Historical Case {i}

Customer Email:
{result["body"]}

Previous Agent Response:
{result["answer"]}
"""
        )

    historical_context = "\n".join(
        historical_cases
    )

    # 3. Create the LLM instruction
    prompt = f"""
You are a customer-support email assistant.

Your job is to draft a suggested response to
the incoming customer email.

Use the historical support cases as guidance.

IMPORTANT RULES:

- Address the customer's actual problem.
- Use the historical responses as guidance.
- Do not invent facts.
- Do not claim that an action has been completed
  unless the evidence supports it.
- If important information is missing, ask for it.
- Be professional and concise.
- Do not mention the historical cases.
- Return only the suggested email reply.

INCOMING EMAIL:

{email}

HISTORICAL SUPPORT CASES:

{historical_context}

SUGGESTED REPLY:
"""

    # 4. Ask the LLM to generate the response
    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return response.output_text, results


if __name__ == "__main__":

    print("Loading retriever...")

    retriever = EmailRetriever()

    test_email = """
    My computer keeps shutting down unexpectedly.
    This has happened several times today.
    I have already restarted it, but the problem
    keeps happening. Can you help me?
    """

    print("\nGenerating response...\n")

    reply, retrieved_cases = generate_reply(
        test_email,
        retriever
    )

    print("=" * 60)
    print("INCOMING EMAIL")
    print("=" * 60)

    print(test_email)

    print("=" * 60)
    print("SUGGESTED RESPONSE")
    print("=" * 60)

    print(reply)

    print("=" * 60)
    print("RETRIEVED CASES")
    print("=" * 60)

    for case in retrieved_cases:

        print(
            f"{case['email_id']} "
            f"(similarity: "
            f"{case['similarity']:.3f})"
        )