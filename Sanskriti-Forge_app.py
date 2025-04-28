import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(page_title="Sanskriti-Forge", layout="wide")
st.title("🧠 Sanskriti-Forge: Explore Indian Culture")
st.markdown("""
    Welcome to **Sanskriti-Forge**, your AI companion for exploring India's rich cultural heritage.
    Ask me anything about festivals, rituals, temples, mythology, art, or history! 🙏🇮🇳
""")

# Function to load pre-trained model
@st.cache_resource(show_spinner=True)
def load_model():
    try:
        # Load the model
        model = pipeline("text-generation", model="distilgpt2")  # Replace with the appropriate model for cultural queries
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

nlp = load_model()

# Initialize conversation history if not available
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Function to generate AI response
def generate_response(user_input):
    if not nlp:
        return "Error: Model could not be loaded."
    
    conversation = "\n".join([f"{msg['role']}: {msg['text']}" for msg in st.session_state.conversation_history])
    conversation += f"\nUser: {user_input}"

    try:
        # Generate response from the model
        response = nlp(conversation, max_length=200, do_sample=True, temperature=0.7)[0]['generated_text']
        generated_response = response.split("User:")[-1].strip()  # Clean up the response
        return generated_response
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error while generating the response."

# Function to display chat interface
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

# Sidebar for conversation history
def display_sidebar():
    st.sidebar.title("Conversation History")
    for message in reversed(st.session_state.conversation_history):
        if message['role'] == 'user':
            st.sidebar.markdown(f"**User:** {message['text']}")
        elif message['role'] == 'bot':
            st.sidebar.markdown(f"**Sanskriti-Forge:** {message['text']}")

# User input field at the bottom
user_input = st.text_input("📝 Enter your cultural query:", placeholder="e.g., Tell me about Pongal festival", key="user_input", help="Type your question and hit Enter!")

# When user submits the query
if user_input:
    # Add user input to conversation history
    st.session_state.conversation_history.append({'role': 'user', 'text': user_input})

    # Generate AI's response
    with st.spinner("Generating cultural insights..."):
        response = generate_response(user_input)
    
    # Add AI's response to conversation history
    st.session_state.conversation_history.append({'role': 'bot', 'text': response})

    # Clear the input field after submitting
    st.session_state.user_input = ""  # Reset input field

# Create layout with columns
col1, col2 = st.columns([2, 5])

# Display the conversation
with col2:
    display_chat()

# Display the sidebar with history
display_sidebar()

# Footer for the app
st.markdown("""<hr><center>Built with ❤️ for Bharat</center>""", unsafe_allow_html=True)
