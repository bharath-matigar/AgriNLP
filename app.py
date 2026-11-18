
import streamlit as st
import pandas as pd
import html
from pathlib import Path
from nlp_engine import agricultural_nlp_pipeline

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AgriNLP | Research Lab",
    page_icon="🌿",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "agricultural_queries_v1.csv"

# =====================================================
# LOAD DATASET
# =====================================================

try:
    dataset = pd.read_csv(DATASET_PATH)
except Exception:
    dataset = pd.DataFrame()

# =====================================================
# SESSION STATE
# =====================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# =====================================================
# AESTHETIC DESIGN SYSTEM
# =====================================================

st.markdown("""
<style>

/* ---------- GLOBAL ---------- */

.stApp {
    background: #F5F4EF;
    color: #293D49;
}

.block-container {
    max-width: 1120px;
    padding-top: 2.1rem;
    padding-bottom: 3.5rem;
}

h1, h2, h3, h4 {
    color: #263C49 !important;
    letter-spacing: -0.7px;
}

p, label {
    color: #596B70;
    line-height: 1.75;
}

[data-testid="stCaptionContainer"] p {
    color: #84918E;
}

/* ---------- SIDEBAR ---------- */

[data-testid="stSidebar"] {
    background: #EAEDE8;
    border-right: 1px solid #DDE3DC;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

.brand {
    font-size: 29px;
    font-weight: 750;
    letter-spacing: -1.5px;
    color: #263C49;
}

.brand span {
    color: #6D9992;
}

.sidebar-note {
    font-size: 10px;
    letter-spacing: 2px;
    color: #81918C;
    font-weight: 700;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #40565B;
    font-size: 14px;
}

/* ---------- PAGE INTRO ---------- */

.topline {
    color: #72948C;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 2.5px;
    margin-bottom: 13px;
}

.hero {
    background: #1D3543;
    border-radius: 15px;
    padding: 36px 36px 32px 36px;
    margin: 10px 0 28px 0;
}

.hero-label {
    color: #A9C7BE;
    font-size: 10px;
    letter-spacing: 2.4px;
    font-weight: 700;
    margin-bottom: 16px;
}

.hero-title {
    color: #F7F5EF;
    font-size: 38px;
    line-height: 1.18;
    letter-spacing: -1.5px;
    font-weight: 650;
    max-width: 690px;
    margin-bottom: 14px;
}

.hero-description {
    color: #C5D2D0;
    font-size: 14px;
    line-height: 1.8;
    max-width: 650px;
}

.hero-footer {
    color: #9CB7B1;
    font-size: 10px;
    letter-spacing: 1.5px;
    margin-top: 28px;
}

/* ---------- SECTION HEADINGS ---------- */

.section-label {
    color: #72948C;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.section-description {
    color: #778580;
    font-size: 13px;
    margin-bottom: 18px;
}

/* ---------- WORKFLOW ---------- */

.workflow-card {
    background: #FFFFFF;
    border: 1px solid #E3E7E0;
    border-radius: 10px;
    padding: 19px 15px;
    min-height: 132px;
}

.workflow-number {
    color: #80A69B;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 1px;
    margin-bottom: 13px;
}

.workflow-title {
    color: #2B414D;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 5px;
}

.workflow-description {
    color: #82908B;
    font-size: 12px;
    line-height: 1.5;
}

/* ---------- QUERY WORKSPACE ---------- */

.workspace {
    background: #FFFFFF;
    border: 1px solid #E0E6E1;
    border-radius: 13px;
    padding: 23px;
    margin-top: 15px;
}

.workspace-label {
    color: #72948C;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

/* ---------- RESULT CARDS ---------- */

.result-card {
    background: #FFFFFF;
    border: 1px solid #E1E7E2;
    border-radius: 10px;
    padding: 19px;
    min-height: 112px;
    margin-bottom: 12px;
}

.result-label {
    color: #73968E;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 1.7px;
    margin-bottom: 12px;
}

.result-value {
    color: #2C414B;
    font-size: 15px;
    line-height: 1.7;
    overflow-wrap: anywhere;
}

/* ---------- ENTITY CHIPS ---------- */

.entity-chip {
    display: inline-block;
    background: #EAF1ED;
    color: #426B64;
    border: 1px solid #D5E4DC;
    border-radius: 5px;
    padding: 5px 9px;
    margin: 3px 3px 3px 0;
    font-size: 12px;
}

/* ---------- INTENT ---------- */

.intent-card {
    background: #EAF1ED;
    border-left: 3px solid #739A90;
    border-radius: 5px;
    padding: 17px;
    color: #2D514D;
    font-size: 15px;
    font-weight: 650;
}

/* ---------- METHODOLOGY ---------- */

.method-card {
    background: #FFFFFF;
    border: 1px solid #E2E7E1;
    border-radius: 9px;
    padding: 16px 18px;
    margin-bottom: 9px;
}

/* ---------- BUTTONS ---------- */

div.stButton > button,
div.stFormSubmitButton > button {
    background: #294A55;
    color: #FFFFFF;
    border: 1px solid #294A55;
    border-radius: 6px;
    min-height: 42px;
    font-weight: 650;
    transition: 0.2s ease;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #416A70;
    color: #FFFFFF;
    border-color: #416A70;
}

/* ---------- INPUTS ---------- */

.stTextArea textarea,
.stTextInput input {
    background: #FCFCFA;
    color: #293D49;
    border: 1px solid #D8E1DC;
    border-radius: 7px;
}

.stTextArea textarea:focus {
    border-color: #729A91;
}

[data-baseweb="select"] > div {
    background: #FCFCFA;
    border-color: #D8E1DC;
}

/* ---------- EXPANDERS ---------- */

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E1E7E2;
    border-radius: 8px;
}

/* ---------- DATAFRAME ---------- */

[data-testid="stDataFrame"] {
    border: 1px solid #E1E7E2;
    border-radius: 8px;
}

/* ---------- DIVIDERS ---------- */

hr {
    border-color: #DDE4DE;
}

/* ---------- METRICS ---------- */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E1E7E2;
    border-radius: 9px;
    padding: 15px;
}

[data-testid="stMetricLabel"] {
    color: #78928B;
}

[data-testid="stMetricValue"] {
    color: #293F4A;
}

/* ---------- FOOTER ---------- */

.footer {
    color: #87938D;
    font-size: 11px;
    letter-spacing: 0.3px;
    text-align: center;
    padding-top: 8px;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">Agri<span>NLP</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-note">RESEARCH LAB · 2026</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("**WORKSPACE**")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Analyze Query",
            "Dataset",
            "About Research"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("**LANGUAGE COVERAGE**")
    st.caption("Hindi · English")
    st.caption("Kannada · English")

    st.divider()

    st.caption("An academic NLP research prototype.")


# =====================================================
# COMMON HEADER
# =====================================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-label">
            AGRICULTURAL LANGUAGE INTELLIGENCE
        </div>

        <div class="hero-title">
            Understanding agriculture, one query at a time.
        </div>

        <div class="hero-description">
            Domain-specific terminology normalization for
            code-mixed agricultural queries, combining language
            processing with agricultural knowledge.
        </div>

        <div class="hero-footer">
            HINDI-ENGLISH &nbsp; · &nbsp; KANNADA-ENGLISH
        </div>
    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown(
        '<div class="topline">AGRINLP / RESEARCH WORKSPACE</div>',
        unsafe_allow_html=True
    )


# =====================================================
# HOME
# =====================================================

if page == "Home":

    st.markdown(
        '<div class="section-label">01 / PROJECT OVERVIEW</div>',
        unsafe_allow_html=True
    )

    st.markdown("## From informal language to structured meaning")

    st.markdown("""
    <div class="section-description">
        The system processes informal agricultural queries and
        identifies meaningful domain-specific information.
    </div>
    """, unsafe_allow_html=True)

    workflow = [
        ("01", "Input", "Code-mixed query"),
        ("02", "Normalize", "Terminology correction"),
        ("03", "Extract", "Agricultural entities"),
        ("04", "Classify", "Query intent")
    ]

    columns = st.columns(4)

    for column, (number, title, description) in zip(columns, workflow):

        with column:

            st.markdown(f"""
            <div class="workflow-card">
                <div class="workflow-number">{number} / STEP</div>
                <div class="workflow-title">{title}</div>
                <div class="workflow-description">{description}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    st.markdown(
        '<div class="section-label">02 / RESEARCH COMPONENTS</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown("### Language coverage")

        st.markdown("""
        <div class="result-card">
            <div class="result-label">LANGUAGE PAIR 01</div>
            <div class="result-value">Hindi + English</div>
        </div>

        <div class="result-card">
            <div class="result-label">LANGUAGE PAIR 02</div>
            <div class="result-value">Kannada + English</div>
        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("### Core capabilities")

        capabilities = [
            "Terminology normalization",
            "Agricultural entity extraction",
            "Intent classification"
        ]

        for item in capabilities:

            st.markdown(f"""
            <div class="method-card">
                <span style="color:#729A91;">●</span>
                &nbsp; {html.escape(item)}
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    st.markdown(
        '<div class="section-label">03 / TRY THE SYSTEM</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Explore query analysis")

    st.write(
        "Enter an agricultural query and observe how the system "
        "normalizes terminology and extracts information."
    )

    if st.button("Open Query Analysis →"):
        st.session_state.analysis_result = None
        st.session_state.open_analysis = True
        st.rerun()


# =====================================================
# ANALYZE QUERY
# =====================================================

elif page == "Analyze Query":

    st.markdown("## Query Analysis")

    st.markdown("""
    <div class="section-description">
        Test the agricultural NLP pipeline using informal,
        code-mixed queries.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="workspace">
        <div class="workspace-label">QUERY WORKSPACE</div>
        Enter a query in Hindi-English or Kannada-English.
    </div>
    """, unsafe_allow_html=True)

    examples = [
        "Select an example",
        "tamatar ke patte peele ho rahe hain",
        "nanna tomato gida ele yellow agide",
        "gehu mein rust disease dikh rahi hai",
        "bhattada gida mele keeta bandide",
        "gehu ki fasal ke liye fertilizer batao"
    ]

    with st.form("query_form"):

        sample = st.selectbox(
            "Choose a sample query",
            examples
        )

        query = st.text_area(
            "Agricultural query",
            value="" if sample == examples[0] else sample,
            placeholder="Example: tamatar ke patte peele ho rahe hain",
            height=115
        )

        submitted = st.form_submit_button(
            "Analyze query  →",
            use_container_width=True
        )

    if submitted:

        if not query.strip():

            st.warning("Please enter a query before analyzing.")

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

        st.markdown(
            '<div class="section-label">ANALYSIS OUTPUT</div>',
            unsafe_allow_html=True
        )

        st.markdown("### Query interpretation")

        left, right = st.columns(2)

        with left:

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">ORIGINAL QUERY</div>
                <div class="result-value">
                    {html.escape(str(result.get("original_query", "")))}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with right:

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">NORMALIZED QUERY</div>
                <div class="result-value">
                    {html.escape(str(result.get("normalized_query", "")))}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Extracted entities")

        entities = result.get("entities", {})

        if isinstance(entities, dict) and entities:

            columns = st.columns(3)

            for index, (category, values) in enumerate(entities.items()):

                with columns[index % 3]:

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">
                            {html.escape(category.replace("_", " ").upper())}
                        </div>
                    """, unsafe_allow_html=True)

                    if values:

                        if not isinstance(values, (list, tuple, set)):
                            values = [values]

                        chips = "".join(
                            f'<span class="entity-chip">{html.escape(str(value))}</span>'
                            for value in values
                        )

                        st.markdown(chips, unsafe_allow_html=True)

                    else:

                        st.caption("No entity detected")

                    st.markdown("</div>", unsafe_allow_html=True)

        elif isinstance(entities, dict):

            st.caption("No entities detected.")

        else:

            st.write(entities)

        intent = str(result.get("intent", "Not identified"))

        st.markdown("### Detected intent")

        st.markdown(f"""
        <div class="intent-card">
            {html.escape(intent.replace("_", " ").title())}
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.caption(
            "The prototype performs terminology normalization, "
            "not complete sentence translation."
        )


# =====================================================
# DATASET
# =====================================================

elif page == "Dataset":

    st.markdown("## Research Dataset")

    st.markdown("""
    <div class="section-description">
        A collection of code-mixed agricultural queries used
        for developing and examining the prototype.
    </div>
    """, unsafe_allow_html=True)

    if dataset.empty:

        st.warning(
            "Dataset file is missing or contains no records."
        )

    else:

        st.metric("Total query records", len(dataset))

        for column in dataset.columns:

            if any(
                word in column.lower()
                for word in ["language", "intent"]
            ):

                st.markdown(
                    f"### {column.replace('_', ' ').title()} distribution"
                )

                st.bar_chart(
                    dataset[column].value_counts()
                )

        st.markdown("### Dataset records")

        st.dataframe(
            dataset,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "The current examples are manually authored and "
            "require independent native-speaker validation."
        )


# =====================================================
# ABOUT RESEARCH
# =====================================================

elif page == "About Research":

    st.markdown("## About the Research")

    st.markdown("""
    <div class="section-description">
        Research objective, methodology and scope of the AgriNLP prototype.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">01 / OBJECTIVE</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Understanding code-mixed agricultural language")

    st.write(
        "The objective is to improve the understanding of informal "
        "agricultural queries through domain-specific terminology "
        "normalization, entity extraction and intent classification."
    )

    st.divider()

    st.markdown(
        '<div class="section-label">02 / METHODOLOGY</div>',
        unsafe_allow_html=True
    )

    methodology = [
        (
            "Dictionary-based normalization",
            "Maps informal agricultural terminology to standard terms."
        ),
        (
            "Contextual phrase rules",
            "Handles selected phrases using contextual patterns."
        ),
        (
            "Fuzzy matching",
            "Handles approximate spelling variations."
        ),
        (
            "Entity extraction",
            "Identifies crops, plant parts, diseases, pests and symptoms."
        ),
        (
            "Intent classification",
            "Identifies the purpose of an agricultural query."
        )
    ]

    for title, description in methodology:

        with st.expander(title):

            st.write(description)

    st.divider()

    st.markdown(
        '<div class="section-label">03 / RESEARCH SCOPE</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Languages and tasks")

    left, right = st.columns(2)

    with left:

        st.markdown("**Language pairs**")
        st.write("• Hindi-English")
        st.write("• Kannada-English")

    with right:

        st.markdown("**Processing tasks**")
        st.write("• Terminology normalization")
        st.write("• Entity extraction")
        st.write("• Intent classification")

    st.divider()

    st.markdown(
        '<div class="section-label">04 / LIMITATIONS</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This is an academic research prototype. Its evaluation "
        "requires independent testing, a suitable held-out dataset "
        "and native-speaker validation."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer">
    AGRINLP &nbsp; · &nbsp; AGRICULTURAL LANGUAGE RESEARCH
    <br><br>
    Domain-Specific Terminology Normalization
</div>
""", unsafe_allow_html=True)
