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
    return pipeline("text-generation", model="gpt2")

nlp = load_model()

# Initialize session state for conversation history if not already present
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Create an input form with a button at the bottom to accept user input
with st.form(key="user_input_form"):
    user_input = st.text_input("📝 Enter your cultural query:", placeholder="e.g., Tell me about Pongal festival")
    submit_button = st.form_submit_button(label="Submit")

# Handle the conversation and dynamic responses
if submit_button and user_input:
    # Append user input to the conversation history
    st.session_state.conversation_history.append(f"User: {user_input}")

    # Generate the AI's response
    with st.spinner("Generating cultural insights..."):
        # Combine conversation history for context
        conversation = "\n".join(st.session_state.conversation_history)
        result = nlp(conversation, max_length=500, do_sample=True, temperature=0.7)[0]['generated_text']

    # Display the AI's response
    st.session_state.conversation_history.append(f"Sanskriti-Forge: {result.strip()}")

    # Display the conversation history (optional, for better interaction)
    st.markdown("### 🗨️ Conversation History:")
    for i, message in enumerate(st.session_state.conversation_history):
        st.markdown(f"{i + 1}. {message}")

    # Clear the input field (this will happen automatically with form submit)
    st.experimental_rerun()

# Optional: Save to knowledge pool (simple append to list)
if "dataset" not in st.session_state:
    st.session_state.dataset = []
st.session_state.dataset.append({"query": user_input, "response": result.strip()})

# Export dataset
if st.button("📁 Download Knowledge Dataset"):
    import pandas as pd
    df = pd.DataFrame(st.session_state.get("dataset", []))
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, "sanskriti_knowledge_pool.csv", "text/csv")

# Footer
st.markdown("""<hr><center>Built with ❤️ for Bharat</center>""", unsafe_allow_html=True)
