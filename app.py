import streamlit as st
import io, fitz, faiss, torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# BLIP-2 imports deferred inside class to avoid heavy load on start
class BLIP2Runner:
    def __init__(self):
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    def generate(self, image: Image.Image, context: str, query: str) -> str:
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=100)
        return self.processor.decode(output[0], skip_special_tokens=True)

def extract_text_and_images(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    texts, images = [], []
    for page in doc:
        texts.append(page.get_text())
        for img in page.get_images(full=True):
            base_image = doc.extract_image(img[0])
            image = Image.open(io.BytesIO(base_image["image"])).convert("RGB")
            images.append(image)
    return texts, images

# Load embedder and summarizer globally for speed
embedder = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization")

# Semantic Retriever class
class SemanticRetriever:
    def __init__(self, dim=384):
        import faiss
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings, metas):
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata.extend(metas)

    def search(self, query_embedding, top_k=1):
        D, I = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        return [self.metadata[i] for i in I[0]]

# Streamlit UI
st.title("📘 Semantic PDF QA with BLIP-2 + Summarization")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting text and images from PDF..."):
        texts, images = extract_text_and_images(uploaded_file)

    if len(texts) == 0:
        st.error("No text found in the PDF.")
    else:
        # Embed text and build semantic index
        embeddings = embedder.encode(texts)
        retriever = SemanticRetriever()
        retriever.add(embeddings, texts)

        query = st.text_input("Enter your question or ask for a summary:")

        if query:
            query_embedding = embedder.encode([query])
            context = retriever.search(query_embedding, top_k=1)[0]

            # Summarize the retrieved text chunk
            with st.spinner("Generating summary..."):
                summary = summarizer(context, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

            st.markdown("### 📌 Summary of Relevant Content:")
            st.write(summary)

            # If images exist, do BLIP-2 based QA on first image
            if images:
                with st.spinner("Generating answer with BLIP-2..."):
                    runner = BLIP2Runner()
                    answer = runner.generate(images[0], summary, query)
                st.markdown("### 💬 Answer based on image + summary:")
                st.write(answer)
            else:
                st.info("No images found in the PDF. Showing summary only.")

