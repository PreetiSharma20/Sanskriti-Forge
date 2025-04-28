import streamlit as st
import os

# Ensure that the necessary libraries are installed
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
from transformers import pipeline

@st.cache_resource(show_spinner=True)
def load_model():
    try:
        model = pipeline("text-generation", model="gpt2")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")

nlp = load_model()

# User prompt
user_input = st.text_input("📝 Enter your cultural query:", placeholder="e.g., Tell me about Pongal festival")

# Response display
if user_input:
    with st.spinner("Generating cultural insights..."):
        result = nlp(user_input, max_length=200, do_sample=True, temperature=0.7)[0]['generated_text']
        
    st.markdown("""
    ### 🤖 Sanskriti-Forge says:
    """ + f"> {result.strip()}")

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
