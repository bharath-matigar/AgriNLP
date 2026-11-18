
import streamlit as st
import pandas as pd
import html
from pathlib import Path
from nlp_engine import agricultural_nlp_pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AgriNLP",
    page_icon="🌱",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "agricultural_queries_v1.csv"

# ---------------- LOAD DATASET ----------------
try:
    dataset = pd.read_csv(DATASET_PATH)
except Exception:
    dataset = pd.DataFrame()

# ---------------- SESSION STATE ----------------
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F5F4EF;
    color: #293D49;
}

.block-container {
    max-width: 1100px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #EAEDE8;
    border-right: 1px solid #DDE3DC;
}

/* Headings and text */
h1, h2, h3, h4 {
    color: #263C49 !important;
}

p, label, .stMarkdown {
    color: #596B70;
}

/* Brand */
.brand {
    font-size: 25px;
    font-weight: 750;
    color: #263C49;
    letter-spacing: -1px;
}

.brand span {
    color: #6D9992;
}

/* Small heading accent */
.eyebrow {
    color: #72948C;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

/* Introduction */
.intro {
    padding: 25px 0 28px 0;
    border-bottom: 1px solid #DDE4DE;
    margin-bottom: 30px;
}

.intro p {
    max-width: 750px;
    line-height: 1.8;
    font-size: 15px;
    color: #596B70;
}

/* Result cards */
.result-card {
    background: #FFFFFF;
    border: 1px solid #E1E7E2;
    border-radius: 10px;
    padding: 18px;
    min-height: 110px;
    margin-bottom: 12px;
}

.card-label {
    color: #72948C;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.card-value {
    color: #293D49;
    font-size: 16px;
    line-height: 1.7;
    overflow-wrap: anywhere;
}

/* Entity tags */
.entity-chip {
    display: inline-block;
    background: #EAF1ED;
    color: #426B64;
    border: 1px solid #D5E4DC;
    border-radius: 5px;
    padding: 5px 9px;
    margin: 3px;
    font-size: 13px;
}

/* Intent */
.intent-card {
    background: #EAF1ED;
    border-left: 3px solid #729A91;
    padding: 16px;
    border-radius: 5px;
    color: #2D514D;
    font-weight: 600;
}

/* Buttons */
div.stButton > button,
div.stFormSubmitButton > button {
    background: #294A55;
    color: #FFFFFF;
    border: 1px solid #294A55;
    border-radius: 6px;
    font-weight: 700;
    min-height: 42px;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #416A70;
    color: #FFFFFF;
    border-color: #416A70;
}

/* Input fields */
.stTextArea textarea {
    background: #FFFFFF;
    color: #293D49;
    border: 1px solid #D8E1DC;
    border-radius: 7px;
}

/* Select boxes */
[data-baseweb="select"] > div {
    background-color: #FFFFFF;
    border-color: #D8E1DC;
}

/* Dividers */
hr {
    border-color: #DDE4DE;
}

/* Footer */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        '<div class="brand">Agri<span>NLP</span></div>',
        unsafe_allow_html=True
    )

    st.caption("AGRICULTURAL QUERY RESEARCH")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Home", "Analyze Query", "Dataset", "About Research"]
    )

    st.divider()
    st.caption("LANGUAGE SUPPORT")
    st.caption("Hindi-English")
    st.caption("Kannada-English")


# ---------------- HEADER ----------------
st.markdown("""
<div class="intro">
    <div class="eyebrow">DOMAIN-SPECIFIC NLP RESEARCH</div>
    <h1>AgriNLP</h1>
    <p>
        Domain-Specific Terminology Normalization for
        Code-Mixed Agricultural Queries.
    </p>
    <p>
        A research prototype for understanding informal agricultural
        language through terminology normalization, entity extraction
        and intent classification.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- HOME ----------------
if page == "Home":

    st.markdown("## Project Overview")

    st.write(
        "AgriNLP aims to improve the understanding of agricultural "
        "queries written in informal Hindi-English and Kannada-English."
    )

    st.markdown("### Processing Workflow")

    columns = st.columns(4)

    workflow = [
        ("01", "Input", "Code-mixed query"),
        ("02", "Normalize", "Agricultural terminology"),
        ("03", "Extract", "Important entities"),
        ("04", "Classify", "Query intent")
    ]

    for column, (number, title, description) in zip(columns, workflow):
        with column:
            st.markdown(f"**{number}**")
            st.markdown(f"#### {title}")
            st.caption(description)

    st.divider()

    st.markdown("### Project Information")

    left, right = st.columns(2)

    with left:
        st.markdown("**Supported languages**")
        st.write("• Hindi-English")
        st.write("• Kannada-English")

    with right:
        st.markdown("**Core components**")
        st.write("• Terminology normalization")
        st.write("• Agricultural entity extraction")
        st.write("• Intent classification")

    st.info("Select Analyze Query from the sidebar to try the system.")


# ---------------- ANALYZE QUERY ----------------
elif page == "Analyze Query":

    st.markdown("## Agricultural Query Analysis")

    st.write(
        "Enter an informal agricultural query to identify "
        "its normalized terminology, entities and intent."
    )

    examples = [
        "Select an example",
        "tamatar ke patte peele ho rahe hain",
        "nanna tomato gida ele yellow agide",
        "gehu mein rust disease dikh rahi hai",
        "bhattada gida mele keeta bandide",
        "gehu ki fasal ke liye fertilizer batao"
    ]

    with st.form("query_form"):
        sample = st.selectbox("Sample queries", examples)

        query = st.text_area(
            "Enter your query",
            value="" if sample == examples[0] else sample,
            placeholder="Type your agricultural query here...",
            height=120
        )

        submitted = st.form_submit_button(
            "Analyze Query",
            use_container_width=True
        )

    if submitted:
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            try:
                st.session_state.analysis_result = (
                    agricultural_nlp_pipeline(query.strip())
                )
            except Exception as error:
                st.error(f"Analysis failed: {error}")

    result = st.session_state.analysis_result

    if result:
        st.divider()
        st.markdown("### Analysis Results")

        left, right = st.columns(2)

        with left:
            st.markdown(f"""
            <div class="result-card">
                <div class="card-label">Original Query</div>
                <div class="card-value">
                    {html.escape(str(result.get("original_query", "")))}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown(f"""
            <div class="result-card">
                <div class="card-label">Normalized Query</div>
                <div class="card-value">
                    {html.escape(str(result.get("normalized_query", "")))}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Extracted Agricultural Entities")

        entities = result.get("entities", {})

        if isinstance(entities, dict):
            columns = st.columns(3)

            for index, (category, values) in enumerate(entities.items()):
                with columns[index % 3]:
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="card-label">
                            {html.escape(category.replace("_", " ").title())}
                        </div>
                    """, unsafe_allow_html=True)

                    if values:
                        chips = "".join(
                            f'<span class="entity-chip">{html.escape(str(value))}</span>'
                            for value in values
                        )
                        st.markdown(chips, unsafe_allow_html=True)
                    else:
                        st.caption("No entity detected")

                    st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.write(entities)

        intent = str(result.get("intent", "Not identified"))

        st.markdown("### Detected Intent")

        st.markdown(f"""
        <div class="intent-card">
            {html.escape(intent.replace("_", " ").title())}
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "Note: This prototype performs terminology normalization, "
            "not complete sentence translation."
        )


# ---------------- DATASET ----------------
elif page == "Dataset":

    st.markdown("## Research Dataset")

    if dataset.empty:
        st.warning("Dataset file is missing or contains no records.")
    else:
        st.metric("Total queries", len(dataset))

        for column in dataset.columns:
            if any(word in column.lower() for word in ["language", "intent"]):
                st.markdown(f"### {column.replace('_', ' ').title()}")
                st.bar_chart(dataset[column].value_counts())

        st.markdown("### Dataset Records")

        st.dataframe(
            dataset,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Dataset examples are manually authored and require "
            "independent validation."
        )


# ---------------- ABOUT RESEARCH ----------------
elif page == "About Research":

    st.markdown("## About the Research")

    st.markdown("### Research Objective")

    st.write(
        "To improve the understanding of code-mixed agricultural "
        "queries through domain-specific terminology normalization."
    )

    st.divider()

    st.markdown("### Methodology")

    methodology = [
        ("Dictionary-based normalization",
         "Maps informal agricultural terminology to standard terms."),
        ("Contextual phrase rules",
         "Handles selected phrases using contextual patterns."),
        ("Fuzzy matching",
         "Handles approximate spelling variations."),
        ("Entity extraction",
         "Identifies crops, plant parts, diseases, pests and symptoms."),
        ("Intent classification",
         "Identifies the purpose of an agricultural query.")
    ]

    for title, description in methodology:
        with st.expander(title):
            st.write(description)

    st.markdown("### Research Scope")
    st.write("• Hindi-English agricultural queries")
    st.write("• Kannada-English agricultural queries")
    st.write("• Terminology normalization")
    st.write("• Agricultural entity extraction")
    st.write("• Query intent classification")

    st.divider()

    st.markdown("### Limitations")

    st.info(
        "This is an academic research prototype. Its evaluation "
        "requires independent testing and native-speaker validation."
    )


# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "AgriNLP | Domain-Specific Agricultural NLP Research Prototype"
)
