# judge.py
from vllm import SamplingParams

JUDGE_SYSTEM_PROMPT = (
    "You are a strict QA sufficiency judge for a tool-using agent.\n"
    "Your job is to decide whether the agent has enough evidence to "
    "answer the question.\n\n"
    "You will receive:\n"
    "- the question\n"
    "- the agent's current reasoning text\n"
    "- the retrieved web documents\n"
    "- how many searches have already been used and the maximum search limit\n\n"
    "You must output exactly one of these labels:\n"
    "- FINALIZE   : Evidence clearly contains enough information to answer.\n"
    "- SEARCH_MORE: Evidence is missing key facts or is clearly insufficient.\n"
    "- GIVE_UP    : Search budget is exhausted or evidence is hopelessly irrelevant.\n\n"
    "Output ONLY one label: FINALIZE, SEARCH_MORE, or GIVE_UP."
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
    # Hard fail-safe on search budget
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

    # Conservative fallback: if not clearly FINALIZE, prefer SEARCH_MORE while budget remains
    if label is None:
        if search_count >= max_search_limit:
            return "GIVE_UP"
        return "SEARCH_MORE"

    # Bias away from FINALIZE when we still have plenty of budget
    if label == "FINALIZE" and search_count < max_search_limit - 1:
        return "SEARCH_MORE"

    return label

