
import streamlit as st
import pandas as pd
import html
from pathlib import Path
from nlp_engine import agricultural_nlp_pipeline

st.set_page_config(
    page_title="AgriNLP | Research Lab",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "agricultural_queries_v1.csv"

try:
    dataset = pd.read_csv(DATASET_PATH) if DATASET_PATH.exists() else pd.DataFrame()
except Exception:
    dataset = pd.DataFrame()

if "result" not in st.session_state:
    st.session_state.result = None

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #101411;
    color: #E9EEE9;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #151B17;
    border-right: 1px solid #29332B;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3, h4 {
    font-family: 'Manrope', sans-serif !important;
    color: #F1F4EF !important;
}

p, label {
    color: #AEB9AE;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
    color: #EAF2E8;
    letter-spacing: -1px;
}

.brand span {
    color: #A5C58A;
}

.eyebrow {
    color: #A5C58A;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

.hero {
    background: linear-gradient(115deg, #202B22, #18231C);
    border: 1px solid #39483A;
    border-radius: 18px;
    padding: 38px;
    margin: 10px 0 30px 0;
}

.hero h1 {
    font-size: 42px;
    margin: 10px 0;
    letter-spacing: -1.5px;
}

.hero p {
    max-width: 720px;
    font-size: 15px;
    line-height: 1.8;
    color: #B8C5B7;
}

.metric {
    background: #19211B;
    border: 1px solid #303C32;
    border-radius: 14px;
    padding: 21px;
    min-height: 115px;
}

.metric-label {
    color: #9EAD9E;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.metric-value {
    font-family: 'Manrope', sans-serif;
    color: #B7D29B;
    font-size: 29px;
    font-weight: 800;
    margin-top: 9px;
}

.metric-note {
    color: #849184;
    font-size: 12px;
}

.panel {
    background: #19211B;
    border: 1px solid #303C32;
    border-radius: 14px;
    padding: 21px;
    margin-bottom: 14px;
}

.panel-label {
    color: #A5C58A;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 13px;
}

.panel-value {
    color: #F0F3ED;
    font-size: 17px;
    line-height: 1.7;
    overflow-wrap: anywhere;
}

.entity {
    display: inline-block;
    background: #263528;
    color: #C4DDB0;
    border: 1px solid #536B4B;
    border-radius: 7px;
    padding: 6px 11px;
    margin: 4px 4px 4px 0;
    font-size: 13px;
}

.intent {
    background: #263528;
    border: 1px solid #617A53;
    border-radius: 10px;
    padding: 17px;
    color: #C4DDB0;
    font-size: 17px;
    font-weight: 700;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: #78945F;
    color: #101610;
    border: 1px solid #A5C58A;
    border-radius: 8px;
    font-weight: 700;
    min-height: 45px;
    transition: all 0.2s ease;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #A5C58A;
    color: #101610;
    border-color: #C5D9B4;
}

.stTextArea textarea {
    background: #141B16;
    color: #F0F3ED;
    border: 1px solid #3A493B;
    border-radius: 9px;
}

[data-baseweb="select"] > div {
    background: #19211B;
    border-color: #3A493B;
}

[data-testid="stMetric"] {
    background: #19211B;
    border: 1px solid #303C32;
    border-radius: 12px;
    padding: 14px;
}

[data-testid="stMetricValue"] {
    color: #B7D29B;
}

hr {
    border-color: #303C32;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


def section_heading(kicker, heading, description=None):
    st.markdown(
        f'<div class="eyebrow">{html.escape(kicker)}</div>',
        unsafe_allow_html=True
    )
    st.subheader(heading)
    if description:
        st.caption(description)


def metric_card(label, value, note):
    st.markdown(f"""
    <div class="metric">
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value">{html.escape(str(value))}</div>
        <div class="metric-note">{html.escape(note)}</div>
    </div>
    """, unsafe_allow_html=True)


# SIDEBAR
with st.sidebar:
    st.markdown(
        '<div class="brand">Agri<span>NLP</span></div>',
        unsafe_allow_html=True
    )
    st.caption("AGRICULTURAL AI RESEARCH")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        ["Overview", "Query Analysis", "Dataset", "Research"],
        label_visibility="visible"
    )

    st.divider()
    st.markdown("**LANGUAGE COVERAGE**")
    st.caption("Hindi – English")
    st.caption("Kannada – English")

    st.divider()
    st.markdown("**RESEARCH MODULES**")
    st.caption("01  Terminology normalization")
    st.caption("02  Entity extraction")
    st.caption("03  Intent classification")

    st.divider()
    st.caption("Academic Research Prototype")


# HEADER
st.markdown("""
<div class="hero">
    <div class="eyebrow">AGRICULTURAL LANGUAGE INTELLIGENCE</div>
    <h1>Understanding agriculture,<br>one query at a time.</h1>
    <p>
        A domain-specific NLP research system designed to normalize
        informal code-mixed agricultural queries and extract meaningful
        agricultural information.
    </p>
</div>
""", unsafe_allow_html=True)


# OVERVIEW
if page == "Overview":

    section_heading(
        "SYSTEM OVERVIEW",
        "Research Dashboard",
        "An overview of the AgriNLP research prototype."
    )

    a, b, c = st.columns(3)

    with a:
        metric_card("Dataset queries", len(dataset), "Available CSV records")

    with b:
        metric_card("Languages", "02", "Hindi-English and Kannada-English")

    with c:
        metric_card("NLP modules", "03", "Core processing components")

    st.markdown("### Research Pipeline")

    steps = st.columns(4)

    pipeline = [
        ("01", "Input Query", "Informal agricultural language"),
        ("02", "Normalization", "Dictionary, context and fuzzy matching"),
        ("03", "Entity Extraction", "Crops, symptoms, pests and diseases"),
        ("04", "Intent Detection", "Agricultural query classification")
    ]

    for col, (number, title, detail) in zip(steps, pipeline):
        with col:
            st.markdown(f"""
            <div class="panel">
                <div class="eyebrow">{number}</div>
                <h4>{title}</h4>
                <p>{detail}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Research Scope")
    st.markdown("""
    - Normalize domain-specific agricultural terminology.
    - Support Hindi-English and Kannada-English code-mixed queries.
    - Extract important agricultural entities.
    - Identify the intent behind agricultural questions.
    """)

    if st.button("Open Query Analysis"):
        st.session_state.result = None
        st.session_state.go_analysis = True
        st.rerun()


# QUERY ANALYSIS
elif page == "Query Analysis":

    section_heading(
        "INTERACTIVE WORKSPACE",
        "Agricultural Query Analysis",
        "Enter a code-mixed agricultural query to begin processing."
    )

    samples = [
        "Select an example...",
        "tamatar ke patte peele ho rahe hain",
        "nanna tomato gida ele yellow agide",
        "gehu mein rust disease dikh rahi hai",
        "bhattada gida mele keeta bandide",
        "gehu ki fasal ke liye fertilizer batao"
    ]

    with st.form("query_form"):
        sample = st.selectbox("TRY A SAMPLE QUERY", samples)

        query = st.text_area(
            "YOUR AGRICULTURAL QUERY",
            value="" if sample == samples[0] else sample,
            placeholder="Example: tamatar ke patte peele ho rahe hain",
            height=120
        )

        submitted = st.form_submit_button(
            "Analyze Query",
            use_container_width=True
        )

    if submitted:
        if not query.strip():
            st.warning("Please enter an agricultural query.")
        else:
            try:
                st.session_state.result = agricultural_nlp_pipeline(
                    query.strip()
                )
            except Exception as error:
                st.error(f"Analysis failed: {error}")

    result = st.session_state.result

    if result:
        st.divider()

        section_heading(
            "ANALYSIS OUTPUT",
            "Processing Results"
        )

        left, right = st.columns(2)

        with left:
            st.markdown(f"""
            <div class="panel">
                <div class="panel-label">Original Query</div>
                <div class="panel-value">
                    {html.escape(str(result.get("original_query", "")))}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown(f"""
            <div class="panel">
                <div class="panel-label">Normalized Query</div>
                <div class="panel-value">
                    {html.escape(str(result.get("normalized_query", "")))}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Agricultural Entity Extraction")

        entities = result.get("entities", {})

        if isinstance(entities, dict):
            cols = st.columns(3)

            for index, (category, values) in enumerate(entities.items()):
                with cols[index % 3]:
                    st.markdown(f"""
                    <div class="panel">
                        <div class="panel-label">
                            {html.escape(category.replace("_", " ").title())}
                        </div>
                    """, unsafe_allow_html=True)

                    if values:
                        chips = "".join(
                            f'<span class="entity">{html.escape(str(value))}</span>'
                            for value in values
                        )
                        st.markdown(chips, unsafe_allow_html=True)
                    else:
                        st.caption("No entity detected")

                    st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.write(entities)

        intent = str(result.get("intent", "Not identified"))

        st.markdown("### Detected Query Intent")
        st.markdown(f"""
        <div class="intent">
            {html.escape(intent.replace("_", " ").title())}
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "AgriNLP performs terminology normalization, "
            "not complete sentence translation."
        )


# DATASET
elif page == "Dataset":

    section_heading(
        "RESEARCH DATA",
        "Agricultural Query Dataset",
        "Explore the examples used in the research prototype."
    )

    if dataset.empty:
        st.warning("Dataset file is missing or contains no records.")
    else:
        st.metric("Total available records", len(dataset))

        if "language" in dataset.columns:
            st.markdown("### Language Distribution")
            st.bar_chart(dataset["language"].value_counts())

        if "intent" in dataset.columns:
            st.markdown("### Intent Distribution")
            st.bar_chart(dataset["intent"].value_counts())

        st.markdown("### Dataset Records")
        st.dataframe(
            dataset,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Manually authored research examples. "
            "Native-speaker validation is still required."
        )


# RESEARCH
elif page == "Research":

    section_heading(
        "PROJECT DOCUMENTATION",
        "Research Methodology",
        "Understanding the approach behind AgriNLP."
    )

    st.markdown("""
    <div class="panel">
        <div class="panel-label">RESEARCH OBJECTIVE</div>
        <div class="panel-value">
            To improve the understanding of code-mixed agricultural
            queries through domain-specific terminology normalization.
        </div>
    </div>
    """, unsafe_allow_html=True)

    modules = [
        ("01", "Dictionary-Based Normalization",
         "Maps informal agricultural terms to standard terminology."),
        ("02", "Contextual Phrase Rules",
         "Uses selected phrase patterns to improve normalization."),
        ("03", "Fuzzy Matching",
         "Handles approximate spellings and terminology variations."),
        ("04", "Agricultural Entity Extraction",
         "Identifies crops, plant parts, diseases, pests and symptoms."),
        ("05", "Intent Classification",
         "Classifies queries into agricultural intent categories.")
    ]

    for number, title, description in modules:
        with st.expander(f"{number}  ·  {title}"):
            st.write(description)

    st.markdown("### Research Scope")
    st.write("Hindi-English and Kannada-English agricultural queries.")
    st.write("Terminology normalization, entity extraction and intent detection.")

    st.warning(
        "This is an academic research prototype. "
        "Evaluation results are preliminary and require independent "
        "testing and native-speaker validation."
    )


# FOOTER
st.divider()
st.markdown(
    "<center style='color:#849184;font-size:12px'>"
    "AgriNLP  ·  Academic Research Prototype  ·  AI for Agriculture"
    "</center>",
    unsafe_allow_html=True
)
