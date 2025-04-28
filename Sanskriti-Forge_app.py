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
    generated_response = response.split("User:")[-1].strip()
    return generated_response

# Function to display chat with alternating user and AI messages
def display_chat():
    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div style="text-align: right; background-color: #e0f7fa; padding: 10px; margin-bottom: 5px; border-radius: 8px; max-width: 70%; display: inline-block;">
                <b>User:</b> {message['text']}
            </div>
            """, unsafe_allow_html=True)
        elif message['role'] == 'bot':
            st.markdown(f"""
            <div style="text-align: left; background-color: #f1f8e9; padding: 10px; margin-bottom: 5px; border-radius: 8px; max-width: 70%; display: inline-block;">
                <b>Sanskriti-Forge:</b> {message['text']}
            </div>
            """, unsafe_allow_html=True)

# Display the user input area at the bottom
user_input = st.text_input("📝 Enter your cultural query:", placeholder="e.g., Tell me about Pongal festival", key="user_input", help="Type your question and hit Enter!")

# When the user submits a message
if user_input:
    # Add user's message to conversation history
    st.session_state.conversation_history.append({'role': 'user', 'text': user_input})

    # Generate AI's response
    with st.spinner("Generating cultural insights..."):
        response = generate_response(user_input)
    
    # Add the AI's response to conversation history
    st.session_state.conversation_history.append({'role': 'bot', 'text': response})

    # Clear the input field after submitting
    # The next line is redundant, Streamlit will handle clearing input automatically when user presses Enter
    st.session_state.user_input = ""  # Reset input field

# Display the chat interface
display_chat()

# Optional: Save the conversation dataset
if st.button("📁 Download Knowledge Dataset"):
    import pandas as pd
    df = pd.DataFrame([{"role": msg['role'], "text": msg['text']} for msg in st.session_state.conversation_history])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "sanskriti_knowledge_pool.csv", "text/csv")

# Footer
st.markdown("""<hr><center>Built with ❤️ for Bharat</center>""", unsafe_allow_html=True)
