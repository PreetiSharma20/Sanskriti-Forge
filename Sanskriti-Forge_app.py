import streamlit as st
from transformers import pipeline
from collections import deque

# Page configuration
st.set_page_config(page_title="Sanskriti-Forge", layout="wide")

# Title and introductory message
st.markdown("""
    <div style="text-align:center; color:#4B0082; font-size:30px; font-weight:bold;">
        🧠 **Sanskriti-Forge**: Indian Culture Chatbot
    </div>
    <div style="text-align:center; font-size:18px; color:#8B0000;">
        Your AI companion for exploring India's rich cultural heritage.
        Ask me anything about festivals, rituals, temples, mythology, art, or history! 🙏🇮🇳
    </div>
""", unsafe_allow_html=True)

# Indian Cultural Theme Color
st.markdown("""
    <style>
        body {
            background-color: #F9F4DB;
        }
        .stTextInput>div>div>input {
            background-color: #FFF5E1;
        }
        .stButton>button {
            background-color: #D4A5A5;
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        return pipeline("text-generation", model="gpt2")
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

nlp = load_model()

# Initialize conversation history as a deque (for efficient pop from the front)
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = deque(maxlen=10)  # Keeping last 10 messages

# Initialize input box in session state
if "input_box" not in st.session_state:
    st.session_state.input_box = ""  # Initialize input box in session_state

# UI for displaying conversation history
def display_conversation_history():
    st.sidebar.markdown("### Conversation History", unsafe_allow_html=True)
    if len(st.session_state.conversation_history) > 0:
        for idx, message in enumerate(st.session_state.conversation_history):
            st.sidebar.markdown(f"**User:** {message['query']}")
            st.sidebar.markdown(f"**Sanskriti-Forge:** {message['response']}")
            st.sidebar.markdown("-" * 50)

# Input field at the bottom
def display_input_field():
    with st.container():
        user_input = st.text_input("📝 Ask your cultural query here:", placeholder="e.g., Tell me about Diwali festival...", key="input_box")
        if user_input:
            st.session_state.input_box = ""  # Clear the input field after submission
            return user_input
        return None

# Response function
def get_response(user_input):
    try:
        result = nlp(user_input, max_length=200, do_sample=True, temperature=0.7)[0]['generated_text']
        return result.strip()
    except Exception as e:
        return f"Error processing your request: {str(e)}"

# Add to conversation history and update sidebar
def add_to_history(user_input, result):
    st.session_state.conversation_history.append({"query": user_input, "response": result})

# Display the conversation
def display_conversation(user_input, result):
    st.markdown(f"**User:** {user_input}")
    st.markdown(f"**Sanskriti-Forge:** {result}")

# Main function to handle chat flow
def main():
    # Display the conversation history on the sidebar
    display_conversation_history()

    # Display the prompt at the bottom of the page
    user_input = display_input_field()

    if user_input:
        with st.spinner("Generating cultural insights..."):
            result = get_response(user_input)
        
        # Add the conversation to history and display the conversation
        add_to_history(user_input, result)
        display_conversation(user_input, result)

# Run the main function
if __name__ == "__main__":
    main()

# Footer
st.markdown("<hr><center>Built with ❤️ for Bharat</center>", unsafe_allow_html=True)
