import streamlit as st
from transformers import pipeline
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

# Initialize Hugging Face models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis")

# Function to perform DuckDuckGo web search and scrape content
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

# Function to summarize the scraped text using Hugging Face BART model
def summarize_text(text):
    if len(text) < 100:
        return "Not enough information found to summarize."
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summary = ''
    for chunk in chunks:
        summary_piece = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        summary += summary_piece + ' '
    return summary.strip()

# Function to analyze sentiment of the summarized content
def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']

# Combining all backend functionality for cultural understanding
def cultural_understanding_agent(prompt):
    # Search and scrape relevant web data
    scraped_data = web_search_and_scrape(prompt)
    # Summarize the scraped data
    summary = summarize_text(scraped_data)
    # Analyze sentiment of the summary
    sentiment, score = analyze_sentiment(summary)
    return summary, sentiment, score

# Streamlit UI setup
st.set_page_config(page_title="Sanskriti-Forge", layout="centered")
st.title("🧠 Sanskriti-Forge: Indian Culture Chatbot")

# User input
user_input = st.text_input("📝 Enter your cultural query:", placeholder="e.g., Tell me about Pongal festival")

if user_input:
    with st.spinner("🔍 Searching the web and generating cultural insights..."):
        summary, sentiment, score = cultural_understanding_agent(user_input)
    
    # Display the summary
    st.markdown("### 🤖 Sanskriti-Forge's Cultural Insight:")
    st.write(summary)
    
    # Display sentiment analysis result
    st.markdown(f"### 🌟 Sentiment Analysis:")
    st.write(f"Sentiment: **{sentiment}** with a confidence score of {score:.2f}")
    
    # Option to save the result to session state (knowledge pool)
    if "dataset" not in st.session_state:
        st.session_state.dataset = []
    st.session_state.dataset.append({"query": user_input, "response": summary, "sentiment": sentiment, "score": score})

    # Download dataset button
    if st.button("📁 Download Knowledge Dataset"):
        import pandas as pd
        df = pd.DataFrame(st.session_state.get("dataset", []))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "sanskriti_knowledge_pool.csv", "text/csv")

# Footer
st.markdown("""<hr><center>Built with ❤️ for Bharat</center>""", unsafe_allow_html=True)
