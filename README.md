# AI Email Suggested-Response System

An AI-powered email assistant that generates suggested replies for incoming customer-support emails.

The system uses historical email-response pairs as a knowledge base. For a new incoming email, it retrieves semantically similar historical cases using embeddings and FAISS, then provides those examples to an LLM to generate a relevant, grounded, and professional suggested response.

## Features

- Historical email-response dataset
- Dataset cleaning and preprocessing
- Train/evaluation split
- Semantic search using sentence embeddings
- FAISS-based retrieval
- LLM-powered response generation
- Automated response quality evaluation
- Per-response quality scores
- Overall evaluation metrics
- Human-validation framework
- Reproducible evaluation pipeline

## Project Architecture

```text
Incoming Email
      |
      v
Text Embedding
      |
      v
FAISS Similarity Search
      |
      v
Top-K Historical Email/Response Pairs
      |
      v
LLM Prompt + Retrieved Context
      |
      v
Suggested Email Reply
      |
      v
Quality Evaluation

## Technology Stack

- Python
- Pandas
- Sentence Transformers
- FAISS
- OpenAI API
- NumPy
- python-dotenv

## Dataset

The project uses the `dataset-tickets-multi-lang-4-20k.csv` dataset containing customer-support email conversations.

The dataset includes fields such as:

- Subject
- Customer email body
- Historical support response
- Support type
- Queue
- Priority
- Language
- Tags

The dataset is synthetic and is used as a source of historical email-response examples for the retrieval and generation pipeline.

## Data Preparation

The raw dataset was cleaned using the following steps:

1. Removed records without an email body or historical response.
2. Filled missing subject values with an empty string.
3. Removed leading and trailing whitespace.
4. Removed empty email or response records.
5. Removed duplicate email-response pairs.
6. Added a unique `email_id` to each record.

After preprocessing, the dataset contained 28,580 usable email-response pairs.

The cleaned dataset was randomly shuffled using a fixed random seed and divided into:

- 80% knowledge base: 22,864 records
- 20% evaluation set: 5,716 records

Only the knowledge-base portion is used for retrieval during response generation. The evaluation data is kept separate to reduce evaluation leakage.

## Retrieval-Augmented Generation

The system uses Retrieval-Augmented Generation (RAG) to ground generated replies in historical support conversations.

### Retrieval

Historical emails from the knowledge base are converted into embeddings using the `all-MiniLM-L6-v2` sentence-transformer model.

The email subject and body are combined and converted into vector representations.

These embeddings are stored in a FAISS index.

When a new email arrives:

1. The incoming email is converted into an embedding.
2. FAISS performs similarity search against the historical knowledge base.
3. The top 3 most similar historical emails are retrieved.
4. The corresponding historical customer emails and agent responses are provided to the LLM as context.

### Generation

The retrieved historical cases are included in the LLM prompt along with the new incoming email.

The LLM uses these examples as guidance to generate a new suggested response.

The prompt instructs the model to:

- Address the customer's actual problem.
- Use historical responses as guidance.
- Avoid inventing facts.
- Avoid claiming unsupported actions.
- Ask for missing information when necessary.
- Maintain a professional and concise tone.
- Return only the suggested email reply.

This approach combines semantic retrieval with LLM-based generation instead of using a traditional classification model.

## Retrieval-Augmented Generation

The system uses Retrieval-Augmented Generation (RAG) to ground generated replies in historical support conversations.

### Retrieval

Historical emails from the knowledge base are converted into embeddings using the `all-MiniLM-L6-v2` sentence-transformer model.

The email subject and body are combined and converted into vector representations.

These embeddings are stored in a FAISS index.

When a new email arrives:

1. The incoming email is converted into an embedding.
2. FAISS performs similarity search against the historical knowledge base.
3. The top 3 most similar historical emails are retrieved.
4. The corresponding historical customer emails and agent responses are provided to the LLM as context.

### Generation

The retrieved historical cases are included in the LLM prompt along with the new incoming email.

The LLM uses these examples as guidance to generate a new suggested response.

The prompt instructs the model to:

- Address the customer's actual problem.
- Use historical responses as guidance.
- Avoid inventing facts.
- Avoid claiming unsupported actions.
- Ask for missing information when necessary.
- Maintain a professional and concise tone.
- Return only the suggested email reply.

This approach combines semantic retrieval with LLM-based generation instead of using a traditional classification model.

## Evaluation Methodology

The system evaluates generated replies using an LLM-based judge rather than exact string matching.

Each generated response is compared with the reference response from the held-out evaluation dataset.

The evaluator scores each response from 1 to 5 on:

- Relevance
- Factual correctness
- Completeness
- Groundedness
- Professional tone

The evaluator also checks for critical errors such as unsupported claims or serious factual mistakes.

### Overall Score

The overall score is calculated using weighted scoring:

- Relevance: 25%
- Factual correctness: 30%
- Completeness: 20%
- Groundedness: 15%
- Professional tone: 10%

This allows the evaluation to measure response quality even when the generated response is worded differently from the reference response.

## Human Validation

A subset of generated responses is used to validate whether the automated LLM-based evaluation agrees with human quality judgments.

Human evaluators use the same 1–5 scoring dimensions as the automated evaluator.

Agreement is measured using:

- Correlation between automated and human scores
- Percentage of scores within ±1 point
- Mean absolute difference

This validation helps determine whether the automated evaluation metric is a reasonable proxy for human judgment.

## Evaluation Results

The evaluation set contains 200 selected emails from the held-out evaluation data.

During the current API quota window, 19 responses were successfully generated and evaluated. The remaining 181 samples could not be evaluated because of the API request-per-day limit.

Results from the 19 successfully evaluated responses:

| Metric | Average Score |
|---|---:|
| Relevance | 4.89 / 5 |
| Factual correctness | 4.79 / 5 |
| Completeness | 4.37 / 5 |
| Groundedness | 4.74 / 5 |
| Professional tone | 5.00 / 5 |
| Overall score | 4.74 / 5 |

### Critical Error Analysis

- Critical errors: 0
- Critical error rate: 0.00%
- Responses scoring 4.5 or higher: 17 / 19
- Responses scoring between 3.5 and 4.49: 1 / 19
- Responses scoring below 3.5: 1 / 19

### Automated Evaluation Validation

For the current 19-response validation sample:

- Overall score correlation: 0.979
- Scores within ±1 point: 100%

These validation results should be considered provisional until the ratings are independently reviewed by a human evaluator.

## How to Run

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Email

## Design Decisions and Trade-offs

### Why RAG?

A retrieval-based approach was chosen instead of relying only on the LLM's general knowledge.

Historical support responses provide examples of how similar customer issues were handled. Retrieving these examples helps the model produce responses that are more relevant to the support domain.

### Why Sentence Transformers?

The `all-MiniLM-L6-v2` model was selected because it provides lightweight semantic embeddings suitable for similarity search while keeping local computation relatively efficient.

### Why FAISS?

FAISS provides efficient vector similarity search and is simple to integrate with the embedding pipeline.

For this project, an exact inner-product index is sufficient because the knowledge base is relatively small and embeddings are normalized before indexing.

### Why Top-3 Retrieval?

The system retrieves the three most similar historical cases.

Using a small number of examples keeps the LLM prompt focused and reduces unnecessary context while still providing multiple examples for guidance.

### Why LLM-based Evaluation?

Exact string matching is not suitable for email-response evaluation because two responses can have different wording while conveying the same information.

An LLM-based evaluator can assess semantic qualities such as relevance, correctness, completeness, groundedness, and professional tone.

### Limitations

- The current evaluation run was limited by the API request quota.
- Only 19 of the 200 selected evaluation samples were successfully evaluated in the current run.
- Human validation currently uses a small validation sample and requires independent human review before being reported as final human validation.
- The dataset is synthetic and may not fully represent real-world customer-support conversations.
- Retrieval quality depends on the similarity of historical examples available in the knowledge base.
- LLM-generated responses may still require human review before being sent to customers.

## AI Tools Used

AI tools were used during development as development assistants.

They were used for:

- Understanding the assignment requirements.
- Designing the project architecture.
- Generating and refining Python code.
- Debugging errors during development.
- Designing the evaluation methodology.
- Improving prompts for response generation and evaluation.
- Explaining technical concepts and implementation decisions.

The final implementation was tested and executed locally, and the generated code was reviewed and modified during development.