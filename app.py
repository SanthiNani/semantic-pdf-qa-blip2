# 📘 Semantic PDF QA with BLIP-2 in Streamlit (CPU-compatible)

import streamlit as st
import fitz  # PyMuPDF
import io
import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from sentence_transformers import SentenceTransformer
import faiss

# 📌 Page setup
st.set_page_config(page_title="Semantic PDF QA", layout="wide")
st.title("📘 Semantic PDF QA with BLIP-2")

# 📤 PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# 🧠 Load models (force CPU for compatibility)
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
    )
    return embedder, summarizer, blip_processor, blip_model

embedder, summarizer, blip_processor, blip_model = load_models()

# 📄 PDF Parsing
def extract_text_and_images(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts, images = [], []
    for page in doc:
        text = page.get_text()
        texts.append(text)
        for img in page.get_images(full=True):
            base_image = doc.extract_image(img[0])
            image = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
            images.append(image)
    return texts, images

# 📌 Embedder + Semantic Search
class SemanticRetriever:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings, metas):
        self.index.add(np.array(embeddings).astype("float32"))
        self.metadata.extend(metas)

    def search(self, query_embedding, top_k=1):
        D, I = self.index.search(np.array(query_embedding).astype("float32"), top_k)
        return [self.metadata[i] for i in I[0]]

# 🧠 BLIP-2 QA
def answer_with_blip(image: Image.Image, context: str, query: str) -> str:
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to("cpu")
    output = blip_model.generate(**inputs, max_new_tokens=100)
    return blip_processor.decode(output[0], skip_special_tokens=True)

# 🧪 Main App Logic
if uploaded_file:
    st.success("✅ PDF uploaded successfully!")
    pdf_bytes = uploaded_file.read()
    texts, images = extract_text_and_images(pdf_bytes)

    # Embed text
    text_embeddings = embedder.encode(texts)
    retriever = SemanticRetriever()
    retriever.add(text_embeddings, texts)

    # User query
    query = st.text_input("🔍 Enter your question:")
    if query:
        query_embedding = embedder.encode([query])
        context = retriever.search(query_embedding, top_k=1)[0]

        # Summarize
        summary = summarizer(context, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]

        st.markdown("### 📌 Context Summary")
        st.info(summary)

        if images:
            st.image(images[0], caption="First image in PDF", use_column_width=True)
            answer = answer_with_blip(images[0], context, query)
            st.markdown("### 🤖 Image-Based Answer")
            st.success(answer)
        else:
            st.warning("⚠️ No images found in the PDF.")
            st.markdown("### 🤖 Answer from Text Only")
            st.success(summary)
