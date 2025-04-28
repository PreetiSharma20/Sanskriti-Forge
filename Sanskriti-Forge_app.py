import streamlit as st
from transformers import pipeline
import os

# Install necessary dependencies (this is just for installation when running locally)
os.system('pip install streamlit torch transformers pandas')

# Page configuration
st.set_page_config(page_title="Sanskriti-Forge", layout="centered")
st.title("🧠 Sanskriti-Forge: Indian Culture Chatbot")
st.markdown("""
    Welcome to **Sanskriti-Forge**, your AI companion for exploring India's rich cultural heritage.
    Ask me anything about festivals, rituals, temples, mythology, art, or history! 🙏🇮🇳
""")

# Load pre-trained model from Hugging Face, use a smaller model for quicker responses
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        model = pipeline("text-generation", model="distilgpt2")  # Using a smaller model for faster performance
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")

nlp = load_model()

# Initialize chat history in session state if not already
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Function to generate a response with a shorter max_length for faster replies
def generate_response(user_input, context=""):
    conversation = context + "\nUser: " + user_input
    response = nlp(conversation, max_length=100, do_sample=True, temperature=0.7)[0]['generated_text']
    
    # Clean the response by removing the prompt part
    generated_response = response.split("User:")[-1].strip()
    return generated_response

# Display chat history with user on the right and bot on the left
def display_chat():
    if 'conversation_history' in st.session_state:
        for i, message in enumerate(st.session_state.conversation_history):
            # Check if 'role' key exists in message
            if 'role' in message:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div style="text-align: right; background-color: #e0f7fa; padding: 10px; margin-bottom: 5px; border-radius: 8px; max-width: 70%; display: inline-block; clear: both;">
                        <b>User:</b> {message['text']}
                    </div>
                    """, unsafe_allow_html=True)

                elif message['role'] == 'bot':
                    st.markdown(f"""
                    <div style="text-align: left; background-color: #f1f8e9; padding: 10px; margin-bottom: 5px; border-radius: 8px; max-width: 70%; display: inline-block; clear: both;">
                        <b>Sanskriti-Forge:</b> {message['text']}
                    </div>
                    """, unsafe_allow_html=True)

# User input section
user_input = st.text_input("📝 Enter your cultural query:", placeholder="e.g., Tell me about Pongal festival")

# When user inputs something
if user_input:
    # Generate response from the bot
    with st.spinner("Generating cultural insights..."):
        response = generate_response(user_input)
    
    # Add user query and bot response to conversation history
    st.session_state.conversation_history.append({'role': 'user', 'text': user_input})
    st.session_state.conversation_history.append({'role': 'bot', 'text': response})

# Display chat interface
display_chat()

# Optional: Save the conversation dataset
if st.button("📁 Download Knowledge Dataset"):
    import pandas as pd
    df = pd.DataFrame([{"role": msg['role'], "text": msg['text']} for msg in st.session_state.conversation_history])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "sanskriti_knowledge_pool.csv", "text/csv")

# Footer
st.markdown("""<hr><center>Built with ❤️ for Bharat</center>""", unsafe_allow_html=True)
