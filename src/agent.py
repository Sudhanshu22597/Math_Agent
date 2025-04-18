from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
import requests
from bs4 import BeautifulSoup

from config import GOOGLE_API_KEY, GEMINI_MODEL_NAME, SIMILARITY_THRESHOLD, MAX_WEB_RESULTS, TAVILY_API_KEY
from vector_store import create_or_load_vector_store
from guardrails import check_input_guardrails, check_output_guardrails
from utils import get_logger

logger = get_logger(__name__)

class MathAgent:
    def __init__(self):
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found.")
        if not TAVILY_API_KEY:
            logger.warning("TAVILY_API_KEY not found. Web search functionality will be limited.")

        self.llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.5)
        self.vector_store = create_or_load_vector_store()
        if self.vector_store:
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 docs
        else:
            self.retriever = None
            logger.error("Vector store not loaded. Knowledge base retrieval disabled.")

        self.web_search_tool = TavilySearchResults(max_results=MAX_WEB_RESULTS, api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

        # --- Prompt Templates ---
        self.rag_prompt_template = PromptTemplate.from_template(
            """You are a helpful Math Professor AI assistant. Your goal is to provide a clear, step-by-step solution to the user's math question, based *only* on the provided context.
            If the context does not contain the answer, state that the information is not available in the knowledge base. Do not make up answers.

            Context from Knowledge Base:
            {context}

            User Question: {question}

            Step-by-step Solution:"""
        )

        self.web_search_prompt_template = PromptTemplate.from_template(
            """You are an expert Math Professor AI assistant. Your goal is to provide a clear, step-by-step solution to the user's math question.
            Use the provided web search results as a primary source of information, formulas, or methods. If the results provide a direct solution, explain it clearly.
            If the results provide relevant concepts or formulas but not a full solution, use your own mathematical reasoning abilities to solve the problem step-by-step, referencing the search results where appropriate.
            If the search results are irrelevant or insufficient even for guiding the solution, state that you could not find enough information online to solve the problem confidently. Do not make up answers if you lack the necessary information or steps.

            Web Search Results:
            {context}

            User Question: {question}

            Step-by-step Solution:"""
        )

        self.no_answer_prompt_template = PromptTemplate.from_template(
            """You are a helpful Math Professor AI assistant. You were unable to find a relevant answer to the user's question in your knowledge base or through web search.
            Politely inform the user that you cannot provide an answer at this time.

            User Question: {question}

            Response:"""
        )

        # --- Chains ---
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.rag_prompt_template
            | self.llm
            | StrOutputParser()
        ) if self.retriever else None

        self.web_chain = (
             # Note: Context here will be formatted web results
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.web_search_prompt_template
            | self.llm
            | StrOutputParser()
        )

        self.no_answer_chain = (
            {"question": RunnablePassthrough()}
            | self.no_answer_prompt_template
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _fetch_and_extract_web_content(self, query: str) -> str:
        """Performs web search and extracts content from top results."""
        if not self.web_search_tool:
            logger.warning("Web search tool not available.")
            return "Web search is not configured."

        try:
            search_results = self.web_search_tool.invoke(query)
            logger.info(f"Web search results for '{query}': {len(search_results)} found.")

            extracted_content = []
            for result in search_results:
                url = result.get("url")
                content_snippet = result.get("content", "") # Use snippet provided by Tavily first
                if url:
                    try:
                        # Basic extraction - enhance with better parsing/filtering
                        headers = {'User-Agent': 'Mozilla/5.0'} # Be polite
                        response = requests.get(url, timeout=5, headers=headers)
                        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        # Try to find main content, fallback to body text
                        main_content = soup.find('main') or soup.find('article') or soup.body
                        text = main_content.get_text(separator='\n', strip=True) if main_content else soup.get_text(separator='\n', strip=True)
                        # Limit length per source
                        extracted_content.append(f"Source: {url}\nContent:\n{text[:1500]}...") # Limit content length
                        logger.debug(f"Extracted content from {url}")
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Failed to fetch URL {url}: {e}. Using snippet: {content_snippet}")
                        extracted_content.append(f"Source: {url}\nContent Snippet:\n{content_snippet}")
                    except Exception as e:
                         logger.warning(f"Failed to parse URL {url}: {e}. Using snippet: {content_snippet}")
                         extracted_content.append(f"Source: {url}\nContent Snippet:\n{content_snippet}")
                elif content_snippet:
                     extracted_content.append(f"Source: Search Result Snippet\nContent:\n{content_snippet}")


            if not extracted_content:
                logger.info("No content extracted from web search results.")
                return "No relevant information found in web search results."

            return "\n\n---\n\n".join(extracted_content)

        except Exception as e:
            logger.error(f"Error during web search or extraction: {e}")
            return "An error occurred during web search."


    def process_query(self, query: str) -> str:
        """Processes the user query through the agent workflow."""
        logger.info(f"Processing query: {query}")

        # 1. Input Guardrails
        is_safe, message = check_input_guardrails(query)
        if not is_safe:
            return message
        # Handle cases where guardrail allows but modifies (like greetings)
        if message != "Input is valid.":
            # If it was just a greeting, we might still want to proceed or just return the greeting response
             if any(greeting in query.lower() for greeting in ["hello", "hi", "how are you", "what is your name"]):
                 return message # Return the specific greeting response
             # Otherwise, proceed with the potentially modified query if applicable, or just use original

        final_response = "Sorry, I encountered an issue and couldn't process your request."

        # 2. Knowledge Base Retrieval
        retrieved_docs = []
        if self.retriever:
            try:
                # Use similarity search with score to apply threshold
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=1)
                if docs_with_scores:
                    best_doc, score = docs_with_scores[0]
                    # Log the score regardless of threshold
                    logger.info(f"Best KB match score: {score} (Threshold: {SIMILARITY_THRESHOLD})")
                    if score < SIMILARITY_THRESHOLD: # FAISS uses L2 distance, lower is better
                        retrieved_docs.append(best_doc)
                        logger.info("Found relevant document in Knowledge Base (below threshold).")
                        # Re-retrieve top k for context if needed, or just use the best one
                        full_context_docs = self.retriever.invoke(query)
                        formatted_context = self._format_docs(full_context_docs)
                        # Use RAG chain
                        try: # Add try-except around RAG chain invocation
                            response = self.rag_chain.invoke({"context": formatted_context, "question": query})
                            final_response = response
                        except Exception as e_rag:
                            logger.error(f"Error invoking RAG chain: {e_rag}", exc_info=True)
                            # Fallback if RAG chain fails
                            final_response = "Sorry, I found relevant information but encountered an error processing it."
                            # Optionally proceed to web search here as another fallback
                            retrieved_docs = [] # Clear retrieved docs to trigger web search if desired

                    else:
                        logger.info("KB documents found but score was above similarity threshold.")
                else:
                    logger.info("No relevant documents found in Knowledge Base.")
            except Exception as e:
                logger.error(f"Error during KB retrieval: {e}")
                # Decide whether to proceed to web search or return error

        # 3. Web Search (if KB retrieval failed or wasn't confident)
        if not retrieved_docs: # If no docs found or below threshold
            logger.info("Proceeding to Web Search.")
            web_context = self._fetch_and_extract_web_content(query)

            if "No relevant information found" in web_context or "Web search is not configured" in web_context or "An error occurred" in web_context:
                 # If web search fails or finds nothing, use the no_answer chain
                 logger.info("Web search failed or found no relevant info. Using no_answer chain.")
                 final_response = self.no_answer_chain.invoke({"question": query})
            else:
                # Use Web chain
                logger.info("Found relevant info via web search. Generating response.")
                try:
                    response = self.web_chain.invoke({"context": web_context, "question": query})
                    final_response = response
                except Exception as e:
                    logger.error(f"Error invoking web chain: {e}")
                    final_response = self.no_answer_chain.invoke({"question": query}) # Fallback


        # 4. Output Guardrails
        is_safe, final_response = check_output_guardrails(final_response)
        if not is_safe:
             logger.error("Output guardrail failed.")
             # Return the safe message from the guardrail itself

        logger.info(f"Final response generated.")
        return final_response

