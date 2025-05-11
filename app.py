import streamlit as st  # type: ignore
from newspaper import Article  # type: ignore
from transformers import pipeline, AutoTokenizer
import validators

st.set_page_config(page_title="News Summarizer", page_icon="📰")

# Load summarizer and tokenizer once
@st.cache_resource
def load_model_and_tokenizer():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    return summarizer, tokenizer

summarizer, tokenizer = load_model_and_tokenizer()

def get_article_text(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.title, article.text

def chunk_text(text, max_tokens=1024):
    """Split text into chunks based on token count."""
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    chunks = [input_ids[i:i + max_tokens] for i in range(0, len(input_ids), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def summarize_text(text):
    chunks = chunk_text(text)
    summary = ''
    for chunk in chunks:
        result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summary += result[0]['summary_text'] + ' '
    return summary.strip()

# Streamlit UI
st.title("📰 News Article Summarizer")

url = st.text_input("Enter a news article URL:")

if st.button("Summarize"):
    if not url:
        st.warning("Please enter a valid URL.")
    elif not validators.url(url):
        st.warning("Invalid URL format.")
    else:
        try:
            with st.spinner("Fetching and summarizing the article..."):
                title, text = get_article_text(url)

                st.subheader("📰 Article Title")
                st.write(title)

                st.subheader("📄 Original Article")
                with st.expander("Click to view full article text"):
                    st.write(text)

                st.subheader("📝 Summary")
                summary = summarize_text(text)
                st.success(summary)

                # Download button
                st.download_button("📥 Download Summary", summary, file_name="summary.txt")
        except Exception as e:
            st.error(f"Failed to summarize the article. Error: {e}")
