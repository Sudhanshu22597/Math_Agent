import streamlit as st
import sys
import os
# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from agent import MathAgent
from utils import get_logger
import json
from datetime import datetime

logger = get_logger(__name__)

# --- Feedback Storage ---
FEEDBACK_FILE = "feedback_log.jsonl"

def log_feedback(query, response, feedback, comments=""):
    """Logs user feedback to a JSON Lines file."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "response": response,
        "feedback": feedback, # "Correct", "Incorrect", "Needs Improvement"
        "comments": comments
    }
    try:
        with open(FEEDBACK_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.info(f"Feedback logged: {feedback}")
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")

# --- Streamlit App ---
st.set_page_config(page_title="Math Professor Agent", layout="wide")
st.title("🧠 Math Professor Agent")
st.caption("Ask me a math question! I'll check my knowledge base or search the web.")

# Initialize agent (cached for efficiency)
@st.cache_resource
def load_agent():
    try:
        logger.info("Initializing Math Agent...")
        agent = MathAgent()
        logger.info("Math Agent initialized successfully.")
        return agent
    except Exception as e:
        logger.error(f"Failed to initialize Math Agent: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not initialize the Math Agent. Please check logs and API keys. Error: {e}")
        return None

math_agent = load_agent()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_response" not in st.session_state:
    st.session_state.current_response = ""
if "current_query" not in st.session_state:
    st.session_state.current_query = ""
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask a math question..."):
    if not math_agent:
        st.error("Agent not initialized. Cannot process query.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process query and display response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
                try:
                    logger.info(f"Calling agent.process_query for: {prompt}")
                    full_response = math_agent.process_query(prompt)
                    logger.info("Agent processing complete.")
                except Exception as e:
                    logger.error(f"Error during agent processing: {e}", exc_info=True)
                    full_response = f"Sorry, an internal error occurred: {e}"

            message_placeholder.markdown(full_response)

        # Store response for feedback
        st.session_state.current_response = full_response
        st.session_state.current_query = prompt
        st.session_state.feedback_submitted = False # Reset feedback state for new response
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun() # Rerun to show feedback buttons immediately

# --- Feedback Section ---
# Show feedback options only if there's a response and feedback hasn't been submitted yet
if st.session_state.current_response and not st.session_state.feedback_submitted:
    st.write("---")
    st.write("**Was this response helpful?**")
    col1, col2, col3, col4 = st.columns([1,1,2, 4])

    with col1:
        if st.button("👍 Correct"):
            log_feedback(st.session_state.current_query, st.session_state.current_response, "Correct")
            st.session_state.feedback_submitted = True
            st.success("Feedback received!")
            st.rerun() # Rerun to hide buttons after submission

    with col2:
        if st.button("👎 Incorrect"):
            log_feedback(st.session_state.current_query, st.session_state.current_response, "Incorrect")
            st.session_state.feedback_submitted = True
            st.warning("Feedback received. We'll use this to improve.")
            st.rerun()

    with col3:
         if st.button("🤔 Needs Improvement"):
            # Optionally add a text area for comments
            comments = st.text_area("Optional: How can it be improved?", key="feedback_comments")
            if st.button("Submit Improvement Feedback", key="submit_improvement"):
                 log_feedback(st.session_state.current_query, st.session_state.current_response, "Needs Improvement", comments)
                 st.session_state.feedback_submitted = True
                 st.info("Feedback received. Thank you!")
                 st.rerun()

# Add a clear separation or hide the feedback section after submission if preferred
if st.session_state.feedback_submitted:
     st.write("---") # Keep separator
     # Optionally add a message: st.write("Thank you for your feedback!")
