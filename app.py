import streamlit as st
import os
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# CONFIG
# -----------------------------------
model = SentenceTransformer("models/all-MiniLM-L6-v2")
MODEL_PATH = "models/all-MiniLM-L6-v2"

st.set_page_config(page_title="AI Resume Screening Pro", page_icon="📄")
st.title("📄 AI Resume Screening System (Offline Mode)")
st.markdown("Fully Offline Semantic Resume Matching")

# -----------------------------------
# LOAD MODEL (LOCAL ONLY)
# -----------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found locally! Please place it inside models/ folder.")
        st.stop()
    return SentenceTransformer(MODEL_PATH)

model = load_model()

# -----------------------------------
# EXTRACT TEXT
# -----------------------------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

# -----------------------------------
# BETTER SIMILARITY FUNCTION
# -----------------------------------
def calculate_similarity(resume_text, job_description):

    resume_chunks = resume_text.split(".")
    resume_chunks = [chunk.strip() for chunk in resume_chunks if len(chunk) > 30]

    if not resume_chunks:
        return 0.0

    jd_embedding = model.encode([job_description])[0]

    chunk_embeddings = model.encode(resume_chunks)

    similarities = cosine_similarity(
        chunk_embeddings,
        [jd_embedding]
    ).flatten()

    top_matches = sorted(similarities, reverse=True)[:5]
    final_score = np.mean(top_matches)

    return round(float(final_score) * 100, 2)

# -----------------------------------
# SESSION STATE
# -----------------------------------
if "score" not in st.session_state:
    st.session_state.score = None

# -----------------------------------
# UI
# -----------------------------------
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description")

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Process"):
        if uploaded_file and job_description:
            resume_text = extract_text_from_pdf(uploaded_file)
            score = calculate_similarity(resume_text, job_description)
            st.session_state.score = score
        else:
            st.warning("Upload resume and paste job description.")

with col2:
    if st.button("🔄 Reset"):
        st.session_state.score = None
        st.experimental_rerun()

# -----------------------------------
# RESULT DISPLAY
# -----------------------------------
if st.session_state.score is not None:

    score = st.session_state.score

    st.markdown("---")
    st.subheader("📊 Match Result")

    st.progress(int(score))
    st.metric("Match Score", f"{score}%")

    if score > 85:
        st.success("🎯 Excellent Candidate Fit")
    elif score > 70:
        st.success("✅ Strong Match")
    elif score > 50:
        st.warning("⚠ Moderate Match")
    else:
        st.error("❌ Low Match")

    st.markdown("💡 Improve by adding relevant keywords from job description.")
