"""Prompt templates. Placeholders in braces are filled with str.format."""

PLAN_PROMPT = """You are the planning step of a research assistant that answers questions from a user's documents.

Break the question into 1 to 3 short, focused search queries for a search engine over the documents.
- Use a single query for simple questions.
- Use one query per item when the question compares or combines several things.
- Keep the key terms from the question (names, acronyms, numbers).

Set needs_web to true only if the question clearly needs recent or external information that is unlikely to be in the documents. Web search available: {web_available}.

Question: {question}"""

GRADE_PROMPT = """You check whether retrieved context is enough to answer a question.

Question: {question}

Context:
{context}

Decide whether the context contains the information needed to answer the question fully and accurately.
- If it does, set sufficient to true and return an empty list of queries.
- If it does not, say briefly what is missing and write up to 2 new search queries (different wording, more specific terms) that could find it.
- If the question is clearly about something the documents don't cover at all, set sufficient to false and return an empty list of queries: searching again won't help."""

ANSWER_PROMPT = """You are a careful research assistant. Answer the question using ONLY the numbered sources below.

Rules:
- Cite every factual claim inline with its source number in square brackets, like [1] or [2][3].
- Only cite numbers that appear in the source list.
- If the sources do not contain the answer, say so plainly and set insufficient_information to true. Do not use outside knowledge, even if you know the answer.
- If the sources answer only part of the question, answer that part and say what is missing.
- Be concise and specific: include the numbers, names and definitions the sources give.
- Write any mathematical notation in LaTeX between $...$ (inline) or $$...$$ (on its own line), for example $p(y_i \\mid x, y_{{1:i-1}})$.
- List in cited_sources only the source numbers you actually used.

Question: {question}

Sources:
{context}"""
