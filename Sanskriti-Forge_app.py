import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(page_title="Sanskriti-Forge", layout="centered")
st.title("🧠 Sanskriti-Forge: Indian Culture Chatbot")
st.markdown("""
    Welcome to **Sanskriti-Forge**, your AI companion for exploring India's rich cultural heritage.
    Ask me anything about festivals, rituals, temples, mythology, art, or history! 🙏🇮🇳
""")

# Load pre-trained model from Hugging Face
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        model = pipeline("text-generation", model="distilgpt2")  # Using a smaller model for faster performance
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")

nlp = load_model()

# Initialize conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Function to generate a response from the AI
def generate_response(user_input):
    conversation = "\n".join([f"{msg['role']}: {msg['text']}" for msg in st.session_state.conversation_history])
    conversation += f"\nUser: {user_input}"

    # Generate response from AI
    response = nlp(conversation, max_length=200, do_sample=True, temperature=0.7)[0]['generated_text']
    
    # Clean the response by removing the prompt part
