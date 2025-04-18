# Math Professor Agent

## Project Goal

This project implements an AI agent designed to replicate a mathematical professor. It leverages an Agentic-RAG (Retrieval-Augmented Generation) architecture to understand mathematical questions and generate step-by-step solutions, simplifying complex concepts for students.

## Architecture Overview

The Math Agent employs a routing pipeline to determine the best way to answer a user's query:

1.  **AI Gateway (Guardrails):** All input queries and generated outputs pass through guardrails. These checks ensure:
    *   **Topic Relevance:** The query is related to mathematics or education. An LLM (Gemini) assists in classifying the topic.
    *   **Privacy:** The query and response do not contain sensitive keywords (e.g., passwords, PII).
2.  **Knowledge Base Retrieval:** The agent first searches a local vector store (built from a provided CSV dataset) for similar, previously answered questions.
    *   If a relevant answer with sufficient confidence (similarity score) is found, it's used as context for the LLM to generate a step-by-step solution.
3.  **Web Search:** If the knowledge base doesn't contain a suitable answer, the agent performs a web search using Tavily.
    *   Content is extracted from the top search results using BeautifulSoup.
    *   The LLM uses this extracted web context, combined with its own reasoning capabilities, to generate a step-by-step solution.
4.  **LLM Generation:** Google's Gemini model (`gemini-2.0-flash`) is used for both guardrail checks and final answer generation, guided by specific prompts based on whether the context came from the knowledge base or web search.
5.  **Human-in-the-Loop (HITL):** A simple feedback mechanism is integrated into the Streamlit UI. Users can rate the generated response (Correct, Incorrect, Needs Improvement), and this feedback is logged to `feedback_log.jsonl` for potential future fine-tuning or analysis.

## Features

*   **Agentic-RAG Workflow:** Dynamically routes between knowledge base retrieval and web search.
*   **Input/Output Guardrails:** Basic topic and privacy filtering.
*   **Knowledge Base:** Utilizes a FAISS vector store created from the `jee_math.csv` dataset.
*   **Web Search & Extraction:** Leverages Tavily for search and BeautifulSoup for basic content extraction.
*   **LLM Integration:** Uses Google Gemini for reasoning and generation.
*   **Streamlit UI:** Provides an interactive chat interface.
*   **Feedback Mechanism:** Allows users to provide feedback on responses, logged locally.

## Setup and Installation

**Prerequisites:**

*   Python 3.9+
*   Git (optional, for cloning)

**Steps:**

1.  **Clone the Repository (Optional):**
    ```bash
    git clone <your-repository-url>
    cd math_agent
    ```
    If you didn't clone, ensure you are in the `/home/-/Desktop/sudhanshu/math_agent` directory.

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    Create a file named `.env` in the project root directory (`/home/-/Desktop/sudhanshu/math_agent/`) and add your API keys:
    ```env
    # /home/-/Desktop/sudhanshu/math_agent/.env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    TAVILY_API_KEY="YOUR_TAVILY_API_KEY"
    ```
    Replace `"YOUR_..._KEY"` with your actual keys.

5.  **Create the Vector Store:**
    Run the following command from the project root directory (`/home/-/Desktop/sudhanshu/math_agent/`) to process the `jee_math.csv` file and create the FAISS index:
    ```bash
    python -m src.vector_store
    ```
    This will create a `faiss_index_jee_math` folder (or the name specified in `config.py`).

## Running the Application

Once the setup is complete, run the Streamlit application from the project root directory:

```bash
streamlit run app.py
```

Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Knowledge Base Details

*   **Dataset:** `data/jee_math.csv` (provided)
*   **Content:** Contains pairs of mathematical questions and their corresponding solutions, likely targeted towards JEE preparation level.
*   **Format:** CSV file with two columns (implicitly 'question' and 'answer').
*   **Example Questions (from KB):**
    *   `"the lcm and hcf of two numbers are 8 and 48 respectively . if one of them is 24 , find the other ?"`
    *   `"a man is 24 years older than his son . in three years , his age will be twice the age of his son . the present age of the son is"`
    *   `"find the volume and surface area of a cuboid 16 m long , 14 m broad and 7 m high ."`

## Web Search Capabilities

*   **Tool:** Tavily Search API is used for retrieving relevant web pages.
*   **Extraction Strategy:** A basic approach using `requests` to fetch page HTML and `BeautifulSoup` to parse and extract text content, primarily from `<main>`, `<article>`, or `<body>` tags. Content length per source is limited. Snippets from Tavily results are used as fallbacks if fetching/parsing fails.
*   **Example Questions (Not in KB):**
    *   `"Explain the concept of Lagrange multipliers with an example."`
    *   `"What is the Collatz conjecture?"`
    *   `"Derive the formula for the volume of a sphere using calculus."`

## Human-in-the-Loop (HITL)

*   **Mechanism:** Feedback buttons (👍 Correct, 👎 Incorrect, 🤔 Needs Improvement) are displayed below each assistant response in the Streamlit UI.
*   **Data Storage:** User feedback (query, response, rating, optional comments) is appended as a JSON object to the `feedback_log.jsonl` file in the project root.
*   **Purpose:** This logged data can be used manually or programmatically to evaluate the agent's performance and potentially fine-tune the models or prompts in the future (e.g., using DSPy).

## Tools & Frameworks Used

*   **Core Logic:** Python
*   **LLM Interaction:** LangChain, `langchain-google-genai`
*   **LLM:** Google Gemini (`gemini-2.0`)
*   **Embeddings:** Google (`text-embedding-004`)
*   **Vector Store:** FAISS (`faiss-cpu`)
*   **Web Search:** Tavily (`tavily-python`)
*   **Web Scraping:** Requests, BeautifulSoup4
*   **UI:** Streamlit
*   **Data Handling:** Pandas
*   **Environment:** `python-dotenv`

