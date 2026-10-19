"""Prompt templates. Placeholders in braces are filled with str.format."""

PLAN_PROMPT = """You are the planning step of a research assistant that answers questions from a user's documents.

Break the question into 1 to 3 short, focused search queries for a search engine over the documents.
- Use a single query for simple questions.
- Use one query per item when the question compares or combines several things.
- Keep the key terms from the question (names, acronyms, numbers).

Set needs_web to true only if the question clearly needs recent or external information that is unlikely to be in the documents. Web search available: {web_available}.

Question: {question}"""

CONDENSE_PROMPT = """You prepare the latest message of a conversation for a research assistant that searches the user's documents.

Conversation so far (oldest first):
{history}

{attachments}

Latest message: {question}

Rewrite the latest message as one standalone question that can be understood without the conversation.
- Replace pronouns and references ("it", "this paper", "the second one", "that method") with what they refer to, using the names and titles from the conversation or the attached files.
- If the latest message is already standalone, or changes the topic, return it unchanged.
- Don't answer it, and don't add details or conditions the user didn't ask for.
- Keep the user's key terms, names and acronyms exactly as written.

Set about_attached_files to true only if the latest message asks about the newly attached file(s) listed above, for example "What is this about?" or "Does this person know React?". Set it to false if no files are listed, or if the message also asks about other documents."""

GRADE_PROMPT = """You check whether retrieved context is enough to answer a question.

Question: {question}

Context:
{context}

Choose one verdict:
- "sufficient": the context contains the information needed to answer the question fully and accurately. Return an empty list of queries.
- "retry": the context is about the question's subject, but a specific detail is missing. Say briefly what is missing and write up to 2 new search queries (different wording, more specific terms) that could find it.
- "out_of_scope": none of the passages is about the question's subject at all (for example, the question asks about a different model, event, person or topic than every passage), so searching these documents again can't help. Say briefly what the question asks about and return an empty list of queries.
If you are unsure whether to choose "retry" or "out_of_scope", choose "retry"."""

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
