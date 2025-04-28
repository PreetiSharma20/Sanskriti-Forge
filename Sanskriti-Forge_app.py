import streamlit as st
from transformers import pipeline
import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup

# Page configuration
st.set_page_config(page_title="Sanskriti-Forge", layout="wide")
st.title("🧠 Sanskriti-Forge: Indian Culture Chatbot")

# Header image with container width
st.image("https://upload.wikimedia.org/wikipedia/commons/0/0b/Indian_Culture_Header_Image.jpg", use_container_width=True)

st.markdown("""
Welcome to **Sanskriti-Forge**, your AI companion for exploring India's rich cultural heritage.
Ask me anything about festivals, rituals, temples, mythology, art, or history! 🙏🇮🇳
""")

# Initialize Hugging Face model
@st.cache_resource(show_spinner=True)
def load_model():
    return pipeline("text-generation", model="gpt2")

nlp = load_model()

# Initialize session state for conversation tracking
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'dataset' not in st.session_state:
    st.session_state.dataset = []

# Function for web scraping and summarization
def web_search_and_scrape(query, num_results=3):
    urls = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=num_results):
            urls.append(result["href"])

    scraped_texts = []
    for url in urls:
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text() for p in paragraphs)
            scraped_texts.append(text[:1500])  # limit per site
        except:
            continue
    return ' '.join(scraped_texts)

# Summarize the content using Hugging Face BART model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    if len(text) < 100:
        return "Not enough information found to summarize."
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summary = ''
    for chunk in chunks:
        summary_piece = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        summary += summary_piece + ' '
    return summary.strip()

# Function for generating cultural understanding summary
def cultural_understanding_agent(prompt):
    scraped_data = web_search_and_scrape(prompt)
    summary = summarize_text(scraped_data)
    return summary

# Define function to handle user input and interaction
def handle_user_input(user_input):
    if user_input:
        # Display the user query
        st.session_state.conversation_history.append({"user": user_input})

        # Fetch cultural understanding or response
        with st.spinner("Processing your query..."):
            result = cultural_understanding_agent(user_input)

        # Display AI's response
        st.session_state.conversation_history.append({"bot": result})

# Input field for user query
user_input = st.text_input("📝 Enter your cultural query:", placeholder="e.g., Tell me about the festival of Diwali", key="user_input")

if user_input:
    handle_user_input(user_input)

# Sidebar for Conversation History Toggle
with st.sidebar:
    # History Toggle Button
    if st.button("📜 View Conversation History"):
        st.write("### Conversation History")
        for message in st.session_state.conversation_history:
            if "user" in message:
                st.markdown(f"**You**: {message['user']}")
            elif "bot" in message:
                st.markdown(f"**Sanskriti-Forge**: {message['bot']}")
                
# Footer with fixed prompt input at the bottom
st.markdown("""
<hr>
<center>Built with ❤️ for Bharat</center>
""", unsafe_allow_html=True)
