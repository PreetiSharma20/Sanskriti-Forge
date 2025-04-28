import streamlit as st
from transformers import pipeline
import os
import pandas as pd

# Install necessary dependencies (this is just for installation when running locally)
os.system('pip install streamlit torch transformers pandas')

# Page configuration
st.set_page_config(page_title="Sanskriti-Forge", layout="centered")
st.title("🧠 Sanskriti-Forge: Indian Culture Chatbot")
st.markdown("""
    Welcome to **Sanskriti-Forge**, your AI companion for exploring India's rich cultural heritage.
    Ask me anything about festivals, rituals, temples, mythology, art, or history! 🙏🇮🇳
""")

# Check if PyTorch is installed
import torch
st.write(f"PyTorch version: {torch.__version__}")

# Load pre-trained model from Hugging Face
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        model = pipeline("text-generation", model="gpt2")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")

nlp = load_model()

# Initialize chat history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Function to generate a response
def generate_response(user_input):
    # Combine previous history with the user input to make conversation flow better
    conversation = "\n".join(st.session_state.conversation_history) + "\nUser: " + user_input
    response = nlp(conversation, max_length=200, do_sample=True, temperature=0.7)[0]['generated_text']
    
    # Clean the response by removing the prompt part
    generated_response = response.split("User:")[-1].strip()
    return generated_response

# Chat input and displaying conversation
user_input = st.text_input("📝 Enter your cultural query:", placeholder="e.g., Tell me about Pongal festival")

# When user inputs something
if user_input:
    # Get chatbot's response
    with st.spinner("Generating cultural insights..."):
        chatbot_response = generate_response(user_input)
    
    # Store the user input and chatbot response in the session state to maintain history
    st.session_state.conversation_history.append(f"User: {user_input}")
    st.session_state.conversation_history.append(f"Sanskriti-Forge: {chatbot_response}")

    # Display the conversation
    for message in st.session_state.conversation_history:
        if message.startswith("User:"):
            st.markdown(f"**User:** {message[6:]}")
        else:
            st.markdown(f"**Sanskriti-Forge:** {message[16:]}")

# Optional: Save the conversation dataset
if st.button("📁 Download Knowledge Dataset"):
    df = pd.DataFrame(st.session_state.get("conversation_history", []), columns=["Conversation"])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "sanskriti_knowledge_pool.csv", "text/csv")

# Footer
st.markdown("""<hr><center>Built with ❤️ for Bharat</center>""", unsafe_allow_html=True)
