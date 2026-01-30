# judge.py
from vllm import SamplingParams

JUDGE_SYSTEM_PROMPT = (
    "You are a strict routing judge for a search-augmented reasoning agent.\n"
    "Your job is to decide whether the agent should ANSWER now or SEARCH again.\n\n"
    "You will receive:\n"
    "- the question\n"
    "- the agent's current reasoning text\n"
    "- the retrieved web documents\n"
    "- how many searches have already been used and the maximum search limit\n\n"
    "You must output exactly one of these labels:\n"
    "- FINALIZE   : Evidence clearly contains enough information to answer correctly.\n"
    "- SEARCH_MORE: Evidence is missing key facts, or the reasoning is incomplete/uncertain.\n"
    "- GIVE_UP    : Search budget is exhausted OR documents are clearly irrelevant and more search is unlikely to help.\n\n"
    "Strict criteria:\n"
    "1. Do NOT choose FINALIZE unless the reasoning is explicitly supported by the documents.\n"
    "   - The key facts used in the answer must appear in the retrieved documents.\n"
    "   - If the answer relies on unsupported guesses, choose SEARCH_MORE (if budget remains) or GIVE_UP.\n"
    "2. If the reasoning contradicts the documents, choose SEARCH_MORE (if budget remains) or GIVE_UP.\n"
    "3. If search_count is 0 and the documents are short or generic, strongly prefer SEARCH_MORE over FINALIZE.\n"
    "4. For math or code questions, only choose FINALIZE if the chain of reasoning is logically complete and reaches a clear final answer.\n\n"
    "Output ONLY one label: FINALIZE, SEARCH_MORE, or GIVE_UP."
)

def build_judge_prompt(question, reasoning, documents, search_count, max_search_limit):
    return (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        f"Current reasoning:\n{reasoning}\n\n"
        f"Retrieved documents (may be long, skim for key facts):\n{documents}\n\n"
        f"Search usage: {search_count} / {max_search_limit}\n\n"
        "Decision (one of: FINALIZE, SEARCH_MORE, GIVE_UP):"
    )



def build_judge_prompt(question, reasoning, documents, search_count, max_search_limit):
    return (
        f"{JUDGE_SYSTEM_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        f"Current reasoning:\n{reasoning}\n\n"
        f"Retrieved documents:\n{documents}\n\n"
        f"Search usage: {search_count} / {max_search_limit}\n\n"
        "Decision (one of: FINALIZE, SEARCH_MORE, GIVE_UP):"
    )


def run_judge(llm, tokenizer, question, reasoning, documents, search_count, max_search_limit):
    # If we've truly exhausted budget, we must stop
    if search_count >= max_search_limit:
        return "GIVE_UP"

    prompt = build_judge_prompt(
        question=question,
        reasoning=reasoning,
        documents=documents,
        search_count=search_count,
        max_search_limit=max_search_limit,
    )

    chat_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = llm.generate(
        [chat_prompt],
        SamplingParams(
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
        ),
    )

    text = outputs[0].outputs[0].text.strip().upper()

    if "FINALIZE" in text:
        label = "FINALIZE"
    elif "SEARCH_MORE" in text:
        label = "SEARCH_MORE"
    elif "GIVE_UP" in text:
        label = "GIVE_UP"
    else:
        label = None

    # Fallback: if unclear and we still have budget, search more
    if label is None:
        if search_count >= max_search_limit:
            return "GIVE_UP"
        return "SEARCH_MORE"

    # Bias away from FINALIZE when budget is still large
    # but allow FINALIZE when we are close to the limit
    if label == "FINALIZE" and search_count < max_search_limit - 2:
        return "SEARCH_MORE"

    return label


