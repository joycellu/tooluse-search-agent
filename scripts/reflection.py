from vllm import SamplingParams

# ===========================
#        PROMPTS
# ===========================

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

#### FOR REFLECTION
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
#### FOR REFLECTION


PRESENCE_CHECK_PROMPT = """
You are a Fact Validator.
User Query: "{search_query}"
Search Results:
"{document_text}"

Task: Determine if the answer to the query is present ANYWHERE in these search results.
- If the text contains specific facts, numbers, or names that answer the query, output "STATUS: PRESENT".
- If the text is irrelevant or does not contain the answer, output "STATUS: ABSENT".
"""

REFINE_EXTRACTION_PROMPT = """
You previously missed the information in these documents.
User Query: "{search_query}"
Search Results:
"{document_text}"

Validator Note: The answer IS present in these results.

Task: Read the results again carefully and extract the EXACT answer to the query.
- Start your response with "**Final Information**" followed by the extracted facts.
- If you still cannot find it (despite the note), output "No helpful information found."
"""

JUDGE_CONTENT_PROMPT = """
You are evaluating if a search result was successful.

Search Query Used: {search_query}
Reasoning Context: ...{history}...
Extracted Information: {info}

Task: Did this search successfully find the information requested in the **Search Query**?
(Do NOT judge based on whether it fully answers the ultimate user question, only if it satisfied the immediate search intent.)

- If the search query asked for "CEO name" and the text contains "Tim Cook", say YES.
- If the text is irrelevant to the query, say NO.

Output format:
"JUDGEMENT: YES" 
or 
"JUDGEMENT: NO | Reason: [Explain exactly what is missing]"
"""

REFLECTION_CONTENT_PROMPT = """
The extracted information was judged as INSUFFICIENT to answer the search query.
User Question: {question}
Search Query: {search_query}
Judge's Verdict: {reason}
Current Info: {info}

Task:
1. Analyze what specific information is still missing.
2. Propose a high-level SEARCH DIRECTION or STRATEGY to find this missing information.

Output Format:
Analysis: [What is missing]
Search_Direction: [Strategic advice for the next step]
"""

HALLUCINATION_CHECK_PROMPT = """
You are a Fact Checker.
User Question: {question}
Proposed Final Answer: {final_answer}

Task: Does this answer contain specific factual claims (dates, numbers, names) that appear UNVERIFIED or possibly hallucinated given the context?
- If the answer is safe or generic, say NO.
- If the answer makes specific claims without clear evidence in reasoning history, say YES.

Output exactly: "JUDGEMENT: YES" or "JUDGEMENT: NO"
"""

# ===========================
#    LOGIC FUNCTIONS
# ===========================

def run_judge_snippet(llm, tokenizer, question, history, query, results):
    """Gate 1: Checks search snippet relevance."""
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

def run_reflection_query(llm, tokenizer, question, history, failed_query, failure_reason):
    """Gate 1 Reflector: Generates new query."""
    history_short = history[-500:] if history else ""
    prompt_content = REFLECTION_QUERY_PROMPT.format(
        question=question, history=history_short, query=failed_query, reason=failure_reason
    )
    
    messages = [{"role": "user", "content": prompt_content}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    output = llm.generate([text_input], sampling_params=SamplingParams(max_tokens=100, temperature=0.7))
    response = output[0].outputs[0].text
    
    if "New_Query:" in response:
        try: return response.split("New_Query:")[1].strip().split('\n')[0]
        except: return None
    return None

def run_presence_check(llm, tokenizer, search_query, document_text):
    """Checks if info exists in the batch of docs (Boolean check)."""
    # Truncate to ~30k chars (approx 7-8k tokens) to fit in context while covering most docs
    short_doc = document_text[:30000] 
    prompt_content = PRESENCE_CHECK_PROMPT.format(
        search_query=search_query, 
        document_text=short_doc
    )
    
    messages = [{"role": "user", "content": prompt_content}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    output = llm.generate([text_input], sampling_params=SamplingParams(max_tokens=10, temperature=0.1))
    response = output[0].outputs[0].text
    
    return "STATUS: PRESENT" in response

def run_refine_extraction(llm, tokenizer, search_query, document_text):
    """Forces a re-read of the documents."""
    # Use a larger window (35k chars) to ensure we see all 10 snippets
    prompt_content = REFINE_EXTRACTION_PROMPT.format(
        search_query=search_query, 
        document_text=document_text[:35000] 
    )
    
    messages = [{"role": "user", "content": prompt_content}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Increase max_tokens slightly to allow for a full explanation
    output = llm.generate([text_input], sampling_params=SamplingParams(max_tokens=1500, temperature=0.5))
    return output[0].outputs[0].text

def run_judge_content(llm, tokenizer, question, search_query, history, extracted_info):
    """Gate 2: Checks content sufficiency."""
    history_short = history[-500:] if history else ""
    prompt_content = JUDGE_CONTENT_PROMPT.format(
        search_query=search_query, 
        history=history_short, 
        info=extracted_info[:2000]
    )
    
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
    """Gate 2 Reflector: Generates search direction."""
    prompt_content = REFLECTION_CONTENT_PROMPT.format(
        question=question, 
        search_query=search_query,
        info=extracted_info[:1000], 
        reason=failure_reason
    )
    
    messages = [{"role": "user", "content": prompt_content}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    output = llm.generate([text_input], sampling_params=SamplingParams(max_tokens=300, temperature=0.7))
    return output[0].outputs[0].text.strip()

def run_hallucination_check(llm, tokenizer, question, final_answer):
    """Post-Hoc Check: Checks for uncited claims."""
    prompt_content = HALLUCINATION_CHECK_PROMPT.format(
        question=question, final_answer=final_answer[:2000]
    )
    
    messages = [{"role": "user", "content": prompt_content}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    output = llm.generate([text_input], sampling_params=SamplingParams(max_tokens=50, temperature=0.1))
    response = output[0].outputs[0].text
    
    return "JUDGEMENT: YES" in response