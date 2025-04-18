from config import ALLOWED_TOPICS, PRIVACY_KEYWORDS, GOOGLE_API_KEY, GEMINI_MODEL_NAME # Added GOOGLE_API_KEY, GEMINI_MODEL_NAME
from utils import get_logger
# Uncommented LLM import
from langchain_google_genai import ChatGoogleGenerativeAI

logger = get_logger(__name__)

# Uncommented and initialized LLM for guardrails
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.0) # Use low temp for classification

def check_input_guardrails(query: str) -> tuple[bool, str]:
    """
    Checks if the input query is appropriate using keyword and LLM checks.
    Returns (is_safe, message).
    """
    query_lower = query.lower()

    # 1. Privacy Check
    for keyword in PRIVACY_KEYWORDS:
        if keyword in query_lower:
            message = "Input contains potentially sensitive information. Please rephrase your question."
            logger.warning(f"Input guardrail triggered (Privacy): {query}")
            return False, message

    # 2. Topic Check (LLM-based)
    is_topic_allowed = False
    try:
        # Use LLM for more robust topic classification
        prompt = f"""Is the following query primarily related to mathematics, logic puzzles, or math education? Answer only with 'yes' or 'no'.

Query: '{query}'

Answer:"""
        response = llm.invoke(prompt)
        llm_decision = response.content.strip().lower()
        logger.info(f"LLM topic check for '{query}': Decision='{llm_decision}'")
        is_topic_allowed = "yes" in llm_decision

    except Exception as e:
        logger.error(f"LLM topic check failed: {e}. Falling back to keyword check.")
        # Fallback to simple keyword matching if LLM fails
        is_topic_allowed = any(topic in query_lower for topic in ALLOWED_TOPICS)
        # Add common math problem indicators as fallback keywords
        fallback_keywords = ["calculate", "solve", "how many", "what is the value", "find the", "equation", "problem", "sum", "difference", "product", "ratio", "average", "percent"]
        is_topic_allowed = is_topic_allowed or any(keyword in query_lower for keyword in fallback_keywords)


    if not is_topic_allowed:
        # Check for common greetings or non-math questions
        if any(greeting in query_lower for greeting in ["hello", "hi", "how are you", "what is your name"]):
             message = "Hello! I am a Math Professor Agent. Please ask me a math-related question."
             logger.info(f"Input guardrail handled (Greeting): {query}")
             # Return False to prevent processing greetings as math problems
             return False, message
        else:
            message = "My expertise is in mathematics and education. Please ask a relevant question."
            logger.warning(f"Input guardrail triggered (Topic): {query}")
            return False, message

    logger.info(f"Input guardrail passed: {query}")
    return True, "Input is valid."

def check_output_guardrails(response: str) -> tuple[bool, str]:
    """
    Checks if the generated response is appropriate.
    Returns (is_safe, response).
    """
    response_lower = response.lower()

    # 1. Privacy Check (Less likely but good practice)
    for keyword in PRIVACY_KEYWORDS:
        if keyword in response_lower:
            message = "Sorry, I cannot provide a response containing potentially sensitive information."
            logger.error(f"Output guardrail triggered (Privacy): {response[:100]}...") # Log snippet
            return False, message

    # 2. Refusal Check (Check if the LLM refused inappropriately or generated harmful content)
    refusal_phrases = ["i cannot", "i'm unable to", "i apologize, but", "as an ai"]
    if any(phrase in response_lower for phrase in refusal_phrases) and len(response) < 150: # Simple check
         # Potentially log for review, but might be a valid refusal (e.g., for harmful content)
         logger.warning(f"Potential refusal detected in output: {response[:100]}...")

    # 3. Placeholder for Harmful Content Check (Requires more sophisticated tools/models)
    # e.g., using Perspective API or a dedicated content moderation model.
    # For now, we assume the base LLM has some safety built-in.

    logger.info("Output guardrail passed.")
    return True, response
