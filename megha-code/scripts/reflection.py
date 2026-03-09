from vllm import SamplingParams

# ===================== PROMPTS =====================

JUDGE_SNIPPET_PROMPT = """
You are a relevance evaluator.
User Question: {question}
Search Query Used: {query}
Search Results (Snippets):
{snippets}

Task: Do these snippets appear RELEVANT and helpful for the search query?
- If YES, output exactly: "JUDGEMENT: YES"
- If NO, output exactly: "JUDGEMENT: NO | Reason: [Short explanation of why it failed]"

Example of NO: "JUDGEMENT: NO | Reason: The results are about fruit apple, but the user asked about Apple Inc. tech."
"""

# ---- FOR REFLECTION 1 ----
REFLECTION_QUERY_PROMPT = """
The previous search query failed.
User Question: {question}
Failed Query: {query}
Judge's Complaint: {reason}

Task: Write a NEW, better search query that specifically addresses the Judge's complaint.
Output Format:
Reasoning: [How I will fix the error]
New_Query: [The new query string]
"""

# ---- FOR EXTRACTION REFINEMENT ----
PRESENCE_CHECK_PROMPT = """
You are a Fact Validator.
User Query: "{search_query}"
Search Results: "{document_text}"

Task: Determine if the answer to the query is present ANYWHERE in these search results.
- If the text contains specific facts, numbers, or names that answer the query, output "STATUS: PRESENT".
- If the text is irrelevant or does not contain the answer, output "STATUS: ABSENT".
"""

REFINE_EXTRACTION_PROMPT = """
You previously missed the information in these documents.
User Query: "{search_query}"
Search Results: "{document_text}"

Validator Note: The answer IS present in these results.

Task: Read the results again carefully and extract the EXACT answer to the query.
- Start your response with "**Final Information**" followed by the extracted facts.
- If you still cannot find it (despite the note), output "No helpful information found."
"""

# ---- FOR FINAL REFLECTION ----
JUDGE_CONTENT_PROMPT = """
You are evaluating the utility of a search result.

Search Query Used: {search_query}
Extracted Information: {info}

Task: Did this search provide ANY useful information, context, or new entities to investigate?

Criteria for "YES":
1. Direct Answer found.
2. **Partial Clues found** (e.g., found a book title that *might* be it, even if unverified).
3. **Correction found** (e.g., "The author is NOT X, but Y").

Criteria for "NO":
1. "No helpful information found."
2. Completely irrelevant spam.

Output format:
"JUDGEMENT: YES" 
or 
"JUDGEMENT: NO | Reason: [Explain exactly why it failed]"
"""

REFLECTION_CONTENT_PROMPT = """
The previous search for "{search_query}" failed or yielded insufficient results.
User Question: {question}
Current Status: {info}
Failure Reason: {reason}

Before generating a new query, determine:
- What intermediate entity must be resolved?
- Has that entity been fully explored?
- Is the current search answering the original question directly?

Also determine what TYPE of answer the original question requires (person, date range, location, event outcome, concept, etc.).
Ensure the next query is aimed at retrieving that answer type.

If the current search only answers an intermediate fact, generate a query that moves one step closer to answering the original question.

Task: Diagnose the failure and generate a precise follow-up search query.

1. **Analyze:** Why did this search fail? (e.g., Too specific? Too broad? Wrong entity? Concept needs splitting?)
2. **Align:** How does fixing this help answer the original User Question?
3. **Act:** Generate the EXACT query string to try next.

Strategies:
- **If too specific/no results:** Broaden to the category (e.g., "Science fiction book series with alien slavery").
- **If results are generic:** Add a specific attribute (e.g., "Author of Animorphs birth year").
- **If complex comparison:** Split into single-fact queries (e.g., Search for Person A first).

Output Format:
Analysis: [Why it failed and what we need]
Next_Query: [The single, optimized query string]
"""

# ---- FOR HALLUCINATION CHECK ----
HALLUCINATION_CHECK_PROMPT = """
You are a strict Fact Checker.

User Question: {question}
Proposed Final Answer: {final_answer}

Search History:
{history}

Task:
1. Does the final answer contain names, dates, or facts that NEVER appeared in the search history?
2. Is the answer directly supported by retrieved evidence?

If the answer includes entities or facts NOT present in the search history, output:
"JUDGEMENT: YES"

If the answer is fully grounded in retrieved evidence, output:
"JUDGEMENT: NO"
"""

# ===================== LOGIC FUNCTIONS =====================

def run_judge_snippet(llm, tokenizer, question, history, query, results):
    snippets = "\n".join([f"- {r.get('snippet', '')[:150]}" for r in results[:5]])
    history_short = history[-500:] if history else ""
    prompt_content = JUDGE_SNIPPET_PROMPT.format(
        question=question, history=history_short, query=query, snippets=snippets
    )
    messages = [{"role": "user", "content": prompt_content}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = llm.generate([text_input], sampling_params=SamplingParams(max_tokens=500, temperature=0.1))
    response = output[0].outputs[0].text
    if "JUDGEMENT: YES" in response:
        return True, None
    elif "JUDGEMENT: NO" in response:
        try: reason = response.split("| Reason:")[1].strip()
        except: reason = "Results appeared irrelevant."
        return False, reason
    return True, None 

# ---- REFLECTION 1 ----
def run_reflection_query(llm, tokenizer, question, history, failed_query, failure_reason):
    history_short = history[-500:] if history else ""
    prompt_content = REFLECTION_QUERY_PROMPT.format(
        question=question, history=history_short, query=failed_query, reason=failure_reason
    )
    messages = [{"role": "user", "content": prompt_content}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = llm.generate([text_input], sampling_params=SamplingParams(max_tokens=100, temperature=0.0))
    response = output[0].outputs[0].text
    if "New_Query:" in response:
        try: return response.split("New_Query:")[1].strip().split('\n')[0]
        except: return None
    return None

# ---- EXTRACTION REFINEMENT ----
def run_presence_check(llm, tokenizer, search_query, document_text):
    short_doc = document_text[:30000]
    prompt_content = PRESENCE_CHECK_PROMPT.format(search_query=search_query, document_text=short_doc)
    messages = [{"role": "user", "content": prompt_content}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = llm.generate([text_input], sampling_params=SamplingParams(max_tokens=10, temperature=0.1))
    response = output[0].outputs[0].text
    return "STATUS: PRESENT" in response

def run_refine_extraction(llm, tokenizer, search_query, document_text):
    prompt_content = REFINE_EXTRACTION_PROMPT.format(search_query=search_query, document_text=document_text[:35000])
    messages = [{"role": "user", "content": prompt_content}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = llm.generate([text_input], sampling_params=SamplingParams(max_tokens=1500, temperature=0.5))
    return output[0].outputs[0].text

# ---- FINAL REFLECTION ----
def run_judge_content(llm, tokenizer, question, search_query, history, extracted_info):
    history_short = history[-500:] if history else ""
    prompt_content = JUDGE_CONTENT_PROMPT.format(search_query=search_query, history=history_short, info=extracted_info[:2000])
    messages = [{"role": "user", "content": prompt_content}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = llm.generate([text_input], sampling_params=SamplingParams(max_tokens=100, temperature=0.1))
    response = output[0].outputs[0].text
    if "JUDGEMENT: YES" in response:
        return True, None
    else:
        try: reason = response.split("| Reason:")[1].strip()
        except: reason = "Information was too vague or incomplete."
        return False, reason

def run_reflection_content(llm, tokenizer, question, search_query, extracted_info, failure_reason):
    prompt_content = REFLECTION_CONTENT_PROMPT.format(
        question=question, search_query=search_query, info=extracted_info[:1000], reason=failure_reason
    )
    messages = [{"role": "user", "content": prompt_content}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = llm.generate([text_input], sampling_params=SamplingParams(max_tokens=300, temperature=0.0))
    return output[0].outputs[0].text.strip()

# ---- HALLUCINATION CHECK ----
def run_hallucination_check(llm, tokenizer, question, history, final_answer):
    prompt_content = HALLUCINATION_CHECK_PROMPT.format(
        question=question, final_answer=final_answer[:2000], history=history[:4000]
    )
    messages = [{"role": "user", "content": prompt_content}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = llm.generate([text_input], sampling_params=SamplingParams(max_tokens=50, temperature=0.1))
    response = output[0].outputs[0].text
    return "JUDGEMENT: YES" in response