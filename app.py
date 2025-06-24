import os
import streamlit as st
import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile

from domains.doc_loader.routes import file_loader
from domains.workflows.routes import document_summarize_orchestrator


st.set_page_config(
    page_title="Agentic Document Intelligence System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4a4a4a;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a4a4a;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #6c757d;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border: 1px solid #dee2e6;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    }
    .entity-box {
        background-color: #e9ecef;
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #6c757d;
    }
    .summary-section {
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #dee2e6;
        padding-bottom: 1rem;
    }
    .key-points {
        background-color: #f1f8ff;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #6c757d;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .progress-label {
        font-size: 0.875rem;
        color: #6c757d;
        margin-bottom: 0.25rem;
    }
    .step-container {
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-radius: 0.25rem;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("<h1 class='main-header'>Agentic Document Intelligence System</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Upload documents to extract key information, summarize content, and analyze entities.</p>", unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2 = st.tabs(["Document Processing", "How It Works"])

with tab2:
    st.markdown("<h3 class='sub-header'>How This System Works</h3>", unsafe_allow_html=True)
    st.markdown("""
    This system processes documents in four main steps:

    1. **Document Upload** - Upload any supported document (PDF, DOCX, TXT, CSV, XLSX)
    2. **Content Extraction** - The system extracts text content from the uploaded file
    3. **Content Summarization** - The extracted content is analyzed and summarized
    4. **Entity Extraction** - (Optional) Entities like people, organizations, dates, and key topics are identified

    The system uses advanced AI to generate high-quality summaries and extract relevant information.
    """)

with st.sidebar:
    st.markdown("<h2 class='sub-header'>Configuration</h2>", unsafe_allow_html=True)

    # Document processing options with tooltips
    st.markdown("""
    <div class="tooltip">What are these settings?
        <span class="tooltiptext">Configure how your documents are processed</span>
    </div>
    """, unsafe_allow_html=True)

    extract_entities = st.checkbox(
        "Extract Entities", 
        value=True, 
        help="Extract people, organizations, dates, and relationships from the document"
    )

    token_max = st.slider(
        "Max Tokens per Chunk", 
        min_value=500, 
        max_value=4000, 
        value=1000, 
        step=100, 
        help="Maximum tokens for each summary chunk. Higher values may produce more detailed summaries but take longer to process."
    )

    # Advanced options (collapsible)
    with st.expander("Advanced Options"):
        st.markdown("""
        <div class="tooltip">What are entity types?
            <span class="tooltiptext">Select which types of information to extract from your documents</span>
        </div>
        """, unsafe_allow_html=True)

        entity_types = st.multiselect(
            "Entity Types to Extract",
            ["people", "organizations", "locations", "dates", "events", "products", "key_metrics", "technical_terms"],
            default=["people", "organizations", "locations", "dates", "events"],
            help="Select which types of entities to extract from the document"
        )

with tab1:
    st.markdown("<h3>Upload Your Documents</h3>", unsafe_allow_html=True)
    st.markdown("<p>Supported formats: PDF, DOCX, TXT, CSV, XLSX</p>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drag and drop files here", 
        accept_multiple_files=True, 
        type=["pdf", "docx", "txt", "csv", "xlsx"]
    )

if uploaded_files:
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            results = []

            for uploaded_file in uploaded_files:
                with NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name

                try:
                    file_path = temp_file_path
                    file_type = Path(uploaded_file.name).suffix.lower().replace('.', '')

                    file_progress = st.progress(0)
                    st.text(f"Processing: {uploaded_file.name}")

                    result = asyncio.run(document_summarize_orchestrator(
                        file_paths=file_path,
                        extract_entities=extract_entities,
                        token_max=token_max
                    ))

                    results.append({
                        "filename": uploaded_file.name,
                        "result": result
                    })

                    file_progress.progress(100)

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")

                finally:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)

            if results:
                st.markdown("<h2 class='sub-header'>Results</h2>", unsafe_allow_html=True)

                for result_item in results:
                    filename = result_item["filename"]
                    result = result_item["result"]

                    with st.expander(f"Results for {filename}", expanded=True):
                        st.markdown("<div class='result-container'>", unsafe_allow_html=True)

                        st.markdown("<h3>Summary</h3>", unsafe_allow_html=True)
                        if result.get("status") == "success" and result.get("summary"):
                            st.markdown(result["summary"])
                        else:
                            st.warning("No summary generated or processing failed.")

                        if extract_entities and result.get("entities") and isinstance(result["entities"], dict) and result["entities"].get("status") == "success":
                            st.markdown("<h3>Extracted Entities</h3>", unsafe_allow_html=True)

                            if result["entities"].get("entities"):
                                st.markdown("<h4>People & Organizations</h4>", unsafe_allow_html=True)

                                people_orgs = [e for e in result["entities"]["entities"] 
                                              if e.get("type") in ["person", "people", "organization", "organizations"]]

                                if people_orgs:
                                    cols = st.columns(3)
                                    for i, entity in enumerate(people_orgs):
                                        with cols[i % 3]:
                                            st.markdown(f"<div class='entity-box'><strong>{entity.get('name', 'Unknown')}</strong><br>{entity.get('type', 'Unknown')}</div>", unsafe_allow_html=True)
                                else:
                                    st.info("No people or organizations found.")

                            if result["entities"].get("dates"):
                                st.markdown("<h4>Key Dates</h4>", unsafe_allow_html=True)
                                for date in result["entities"]["dates"]:
                                    st.markdown(f"**{date.get('date', 'Unknown date')}**: {date.get('context', '')}")

                            if result["entities"].get("key_topics"):
                                st.markdown("<h4>Key Topics</h4>", unsafe_allow_html=True)
                                st.write(", ".join(result["entities"]["key_topics"]))

                            if result["entities"].get("sentiment"):
                                st.markdown("<h4>Document Sentiment</h4>", unsafe_allow_html=True)
                                sentiment = result["entities"]["sentiment"]
                                if isinstance(sentiment, dict) and "polarity" in sentiment:
                                    st.markdown(f"**Polarity**: {sentiment['polarity']}")
                                    if "explanation" in sentiment:
                                        st.markdown(f"**Explanation**: {sentiment['explanation']}")
                                else:
                                    st.json(sentiment)

                        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Please upload one or more documents to begin processing.")
