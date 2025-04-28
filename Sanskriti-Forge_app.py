import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(page_title="Sanskriti-Forge", layout="centered")
st.title("🧠 Sanskriti-Forge: Indian Culture Chatbot")
st.markdown("""
Welcome to **Sanskriti-Forge**, your AI companion for exploring India's rich cultural heritage.
Ask me anything about festivals, rituals, temples, mythology, art, or history! 🙏🇮🇳
""")

# Function to load the model with exception handling
@st.cache_data(show_spinner=True)
def load_model():
    try:
        # Use a lighter version of GPT-2 (distilgpt2) for better performance
        model = pipeline("text-generation", model="distilgpt2")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

nlp = load_model()

# User prompt
user_input = st.text_input("📝 Enter your cultural query:", placeholder="e.g., Tell me about Pongal festival")

# Response display
if user_input:
    if nlp is None:
        st.error("There was an issue loading the model. Please try again later.")
    else:
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
