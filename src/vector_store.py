import pandas as pd
import os
import sys # Import sys
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document

# Attempt absolute imports first (for when imported as a module)
try:
    from config import GOOGLE_API_KEY, EMBEDDING_MODEL_NAME, CSV_PATH, VECTOR_STORE_PATH
    from utils import get_logger
# If run directly via python -m src.vector_store, use relative imports
except ModuleNotFoundError:
    # Ensure the parent directory (math_agent) is in the path for relative imports to work correctly
    # This might be redundant if -m handles it, but can help in some environments
    # script_dir = os.path.dirname(__file__)
    # parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
    # if parent_dir not in sys.path:
    #     sys.path.insert(0, parent_dir)
    from .config import GOOGLE_API_KEY, EMBEDDING_MODEL_NAME, CSV_PATH, VECTOR_STORE_PATH
    from .utils import get_logger

logger = get_logger(__name__)

def load_csv_data(file_path: str) -> list[Document]:
    """Loads data from CSV and converts it into LangChain Documents."""
    try:
        # Assuming the CSV has 'question' and 'answer' columns (adjust if needed)
        # The header might be missing, so use header=None and assign names
        df = pd.read_csv(file_path, header=None, names=['question', 'answer'], on_bad_lines='skip', engine='python', quoting=1) # quoting=1 for QUOTE_MINIMAL
        df.dropna(subset=['question', 'answer'], inplace=True) # Drop rows where question or answer is missing

        documents = []
        for _, row in df.iterrows():
            # Combine question and answer for better context during retrieval
            content = f"Question: {row['question']}\nAnswer: {row['answer']}"
            # Use the question as metadata for potential future use
            metadata = {"source": os.path.basename(file_path), "question": row['question']}
            documents.append(Document(page_content=content, metadata=metadata))
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
    except FileNotFoundError:
        logger.error(f"CSV file not found at {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading or processing CSV {file_path}: {e}")
        return []

def create_or_load_vector_store(force_recreate: bool = False):
    """Creates a FAISS vector store from the CSV or loads an existing one."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)

    # Construct paths relative to the project root (math_agent directory)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    vector_store_full_path = os.path.join(project_root, VECTOR_STORE_PATH)
    csv_full_path = os.path.join(project_root, CSV_PATH)

    if os.path.exists(vector_store_full_path) and not force_recreate:
        try:
            logger.info(f"Loading existing vector store from {vector_store_full_path}")
            vector_store = FAISS.load_local(vector_store_full_path, embeddings, allow_dangerous_deserialization=True)
            logger.info("Successfully loaded vector store.")
            return vector_store
        except Exception as e:
            logger.warning(f"Failed to load existing vector store: {e}. Recreating...")

    logger.info(f"Creating new vector store from {csv_full_path}")
    documents = load_csv_data(csv_full_path) # Use the constructed path
    if not documents:
        logger.error("No documents loaded, cannot create vector store.")
        return None

    try:
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(vector_store_full_path) # Save using the constructed path
        logger.info(f"Successfully created and saved vector store to {vector_store_full_path}")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        return None

if __name__ == "__main__":
    # Example usage: Run this script directly to create the index
    # No need for specific imports here anymore, as the top-level try/except handles it.
    logger.info("Running vector_store.py as main script...")
    create_or_load_vector_store(force_recreate=True)
    logger.info("Finished running vector_store.py.")
