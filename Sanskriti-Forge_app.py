import streamlit as st
from transformers import pipeline
import pandas as pd

# Page configuration
st.set_page_config(page_title="Sanskriti-Forge", layout="centered")
st.markdown(
    """
    <style>
    body {
        background-color: #ffefd5;  /* Warm background color (light cream) */
    }
    .title {
        color: #FF6F00;  /* Saffron color for the title */
        font-family: 'Georgia', serif;
        font-size: 3em;
        text-align: center;
    }
    .subtitle {
        color: #4CAF50;  /* Indian green color for the subtitle */
        font-family: 'Verdana', sans-serif;
        font-size: 1.5em;
        text-align: center;
    }
    .conversation {
        font-family: 'Verdana', sans-serif;
        font-size: 1.1em;
        background-color: #fff8e1;  /* Light yellow for chat bubble */
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        border: 2px solid #FF6F00;
    }
    .button {
        background-color: #FF6F00;  /* Saffron */
        color: white;
        border-radius: 10px;
        padding: 10px;
    }
    .footer {
        text-align: center;
        color: #FF6F00;
        font-family: 'Georgia', serif;
        font-size: 1.2em;
        padding-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add Indian-themed image
st.image("https://upload.wikimedia.org/wikipedia/commons/0/0b/Indian_Culture_Header_Image.jpg", use_column_width=True)

st.markdown(
    "<div class='title'>🧠 Sanskriti-Forge: Indian Culture Chatbot</div>", unsafe_allow_html=True
)
st.markdown(
    "<div class='subtitle'>Ask me anything about festivals, rituals, temples, mythology, art, or history! 🙏🇮🇳</div>", unsafe_allow_html=True
)

# Load pre-trained model from Hugging Face
@st.cache_resource(show_spinner=True)
def load_model():
    return pipeline("text-generation", model="gpt2")

nlp = load_model()

# Initialize session state for conversation history if not already present
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Create an input form with a button at the bottom to accept user input
with st.form(key="user_input_form"):
    user_input = st.text_input("📝 Enter your cultural query:", placeholder="e.g., Tell me about Pongal festival")
    submit_button = st.form_submit_button(label="Submit", use_container_width=True)

# Handle the conversation and dynamic responses
if submit_button and user_input:
    # Append user input to the conversation history
    st.session_state.conversation_history.append(f"User: {user_input}")

    # Generate the AI's response
    with st.spinner("Generating cultural insights..."):
        # Combine conversation history for context
        conversation = "\n".join(st.session_state.conversation_history)
        
        try:
            result = nlp(conversation, max_length=500, do_sample=True, temperature=0.7)[0]['generated_text']
            if not result:
                result = "I'm sorry, I couldn't generate a response. Please try again."
        except Exception as e:
            result = f"Error generating response: {str(e)}"

    # Append the AI's response to the conversation history
    st.session_state.conversation_history.append(f"Sanskriti-Forge: {result.strip()}")

    # Display the conversation history with Indian culture-themed styling
    st.markdown("### 🗨️ Conversation History:")
    for i, message in enumerate(st.session_state.conversation_history):
        if "User:" in message:
            st.markdown(f"<div class='conversation' style='border-left: 6px solid #4CAF50'>{message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='conversation' style='border-left: 6px solid #FF6F00'>{message}</div>", unsafe_allow_html=True)

    # Clear the input field (this will happen automatically with form submit)
    st.experimental_rerun()

# Optional: Save to knowledge pool (simple append to list)
if "dataset" not in st.session_state:
    st.session_state.dataset = []

# Ensure result is valid before saving
if user_input and result:
    st.session_state.dataset.append({"query": user_input, "response": result.strip()})

# Export dataset
if st.button("📁 Download Knowledge Dataset", key="download_button", use_container_width=True):
    df = pd.DataFrame(st.session_state.get("dataset", []))
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "sanskriti_knowledge_pool.csv", "text/csv")

# Footer with Indian culture theme
st.markdown("<div class='footer'>Built with ❤️ for Bharat</div>", unsafe_allow_html=True)
