from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import streamlit as st

# Load your Help Guide PDF (already uploaded with the app)
HELP_GUIDE = "help-guide.pdf"

# Extract and chunk
@st.cache_resource
def load_chunks():
    reader = PdfReader(HELP_GUIDE)
    full_text = "".join([page.extract_text() for page in reader.pages])
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(full_text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

vectorstore = load_chunks()

st.set_page_config(page_title="Help Guide Assistant", layout="wide")
st.title("Help Guide Chat Assistant")

query = st.text_input("What do you want to know?")
if query:
    results = vectorstore.similarity_search(query, k=3)
    for i, result in enumerate(results):
        st.markdown(f"**Result {i+1}:**")
        st.write(result.page_content)
