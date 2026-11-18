import streamlit as st
import pandas as pd
import html
from pathlib import Path
from nlp_engine import agricultural_nlp_pipeline

st.set_page_config(
    page_title="AgriNLP | Research Prototype",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: #f5f8f3;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    background: linear-gradient(120deg, #123c2b, #26734b);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 8px;
}
.hero p {
    color: #e1eee4;
    font-size: 16px;
}
.section-title {
    color: #174b32;
    font-size: 23px;
    font-weight: 700;
    margin-top: 20px;
}
.result-card {
    background: white;
    border: 1px solid #e0e9df;
    border-radius: 14px;
    padding: 20px;
    min-height: 125px;
    margin-bottom: 12px;
}
.result-label {
    color: #62806b;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.result-value {
    color: #183d2b;
    font-size: 17px;
    font-weight: 600;
    overflow-wrap: anywhere;
}
div.stButton > button {
    background: #246b45;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 22px;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #174b32;
    color: white;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🌱 AgriNLP")
    st.caption("Agricultural Language Intelligence")

    st.divider()
    st.markdown("### Supported languages")
    st.markdown("- Hindi-English")
    st.markdown("- Kannada-English")

    st.divider()
    st.markdown("### Research modules")
    st.markdown("""
    - Terminology normalization
    - Agricultural entity extraction
    - Query intent classification
    """)

    st.divider()
    st.caption("Academic Research Prototype")

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <h1>🌱 AgriNLP</h1>
    <p>Domain-Specific Terminology Normalization for
    Code-Mixed Agricultural Queries</p>
    <p>Transform informal agricultural queries into
    structured, understandable information.</p>
</div>
""", unsafe_allow_html=True)

# ---------- DATASET METRICS ----------
dataset_path = Path("/content/agricultural_queries_v1.csv")
dataset_count = 0

if dataset_path.exists():
    try:
        dataset_count = len(pd.read_csv(dataset_path))
    except Exception:
        pass

m1, m2, m3 = st.columns(3)

m1.metric("Research Dataset", f"{dataset_count} Queries")
m2.metric("Language Support", "2 Languages")
m3.metric("NLP Modules", "3 Modules")

st.divider()

# ---------- QUERY INPUT ----------
st.markdown('<div class="section-title">🔍 Agricultural Query Analysis</div>',
            unsafe_allow_html=True)

st.write("Enter an agricultural question in Hindi-English or Kannada-English.")

examples = [
    "Select an example...",
    "tamatar ke patte peele ho rahe hain",
    "nanna tomato gida ele yellow agide",
    "gehu mein rust disease dikh rahi hai",
    "bhattada gida mele keeta bandide",
    "gehu ki fasal ke liye fertilizer batao"
]

selected = st.selectbox("Try a sample query", examples)

default_query = "" if selected == examples[0] else selected

query = st.text_area(
    "Your agricultural query",
    value=default_query,
    placeholder="Example: tomato gidada ele haladi agide",
    height=110
)

analyze = st.button("🌿 Analyze Agricultural Query", use_container_width=True)

# ---------- ANALYSIS ----------
if analyze:
    if not query.strip():
        st.warning("Please enter an agricultural query.")
    else:
        try:
            result = agricultural_nlp_pipeline(query.strip())

            st.success("Query analysis completed successfully!")

            st.markdown('<div class="section-title">📊 Analysis Results</div>',
                        unsafe_allow_html=True)

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Original Query</div>
                    <div class="result-value">
                    {html.escape(str(result["original_query"]))}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Normalized Query</div>
                    <div class="result-value">
                    {html.escape(str(result["normalized_query"]))}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 🧩 Agricultural Entity Extraction")
            entities = result.get("entities", {})

            if isinstance(entities, (dict, list)):
                st.json(entities)
            else:
                st.write(entities)

            st.markdown("### 🎯 Query Intent Classification")
            st.info(str(result.get("intent", "Not identified")))

            st.caption(
                "Note: This prototype performs terminology normalization, "
                "not complete sentence translation."
            )

        except Exception as e:
            st.error(f"Analysis failed: {e}")

# ---------- RESEARCH INFORMATION ----------
st.divider()
st.markdown('<div class="section-title">📚 About the Research</div>',
            unsafe_allow_html=True)

st.write(
    "AgriNLP investigates domain-specific terminology normalization "
    "for code-mixed agricultural queries. It combines dictionary-based "
    "normalization, contextual rules and fuzzy matching with "
    "agricultural entity extraction and intent classification."
)

with st.expander("View Research Evaluation Information"):
    st.write(f"Dataset size: {dataset_count} manually authored examples.")
    st.write("Languages: Hindi-English and Kannada-English.")
    st.write("Evaluation results are preliminary and require independent testing.")

st.divider()
st.markdown(
    "<center style='color:#718574'>AgriNLP | Academic Research Prototype</center>",
    unsafe_allow_html=True
)
