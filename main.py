import asyncio
import os
from pathlib import Path
from typing import Dict, Any

import streamlit as st
from dotenv import load_dotenv

from workflows.agents.graph import get_entities_extraction_graph
from workflows.agents.models import EntityExtractionState
from workflows.settings import config_settings

load_dotenv()

st.set_page_config(
    page_title="Document Summarization & Entity Extraction",
    page_icon="📄",
    layout="wide"
)


def create_upload_directory():
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    return upload_dir


def validate_file(uploaded_file) -> tuple[bool, str]:
    if not uploaded_file:
        return False, "No file provided"

    if not uploaded_file.name:
        return False, "File has no name"

    file_size_mb = uploaded_file.size / (1024 * 1024) if uploaded_file.size else 0
    if file_size_mb > config_settings.MAX_FILE_SIZE_MB:
        return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({config_settings.MAX_FILE_SIZE_MB}MB)"

    if file_size_mb == 0:
        return False, "File appears to be empty"

    file_extension = uploaded_file.name.split('.')[-1].lower() if '.' in uploaded_file.name else ''
    if not file_extension:
        return False, "File has no extension"

    supported_types = ['txt', 'pdf', 'docx', 'xlsx', 'csv']
    if file_extension not in supported_types:
        return False, f"File type '{file_extension}' not supported. Supported types: {', '.join(supported_types)}"

    return True, "File validation passed"


def save_uploaded_file(uploaded_file, upload_dir: Path) -> tuple[str, str, str]:
    is_valid, message = validate_file(uploaded_file)
    if not is_valid:
        raise ValueError(message)

    try:
        file_path = upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            content = uploaded_file.getbuffer()
            if len(content) == 0:
                raise ValueError("File content is empty")
            f.write(content)

        file_extension = uploaded_file.name.split('.')[-1].lower()

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise ValueError("Failed to save file or file is empty after saving")

        return str(file_path), uploaded_file.name, file_extension

    except Exception as e:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise ValueError(f"Error saving file: {str(e)}")


async def process_document(file_path: str, file_name: str, file_type: str) -> Dict[str, Any]:
    try:
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        if os.path.getsize(file_path) == 0:
            raise ValueError("File is empty")

        # Create initial state for the entities extraction graph
        state = EntityExtractionState(
            file_path=file_path,
            file_type=file_type,
            file_name=file_name,
            original_file_name=file_name,
            contents=[],
            final_summary="",
            entities={}
        )

        # Use the main entities extraction graph which handles the complete flow:
        # load_documents → summarization → extract_entities
        graph = get_entities_extraction_graph()
        if not graph:
            raise ValueError("Failed to initialize entities extraction graph")

        # Execute the complete workflow
        result = await graph.ainvoke(state)

        if not result:
            raise ValueError("Entities extraction process failed to return results")

        # Validate results
        if not result.get("contents"):
            raise ValueError("No content could be extracted from the document")

        final_summary = result.get("final_summary", "")
        if not final_summary or not final_summary.strip():
            raise ValueError("Failed to generate summary from document")

        # Calculate content metrics for display
        content_length = 0
        if result.get("contents"):
            if isinstance(result["contents"][0], str):
                content_length = sum(len(content) for content in result["contents"])
            else:
                # If contents are Document objects
                content_length = sum(
                    len(doc.page_content) for doc in result["contents"] if hasattr(doc, 'page_content'))

        if content_length < config_settings.MIN_CONTENT_LENGTH:
            raise ValueError(
                f"Document content too short ({content_length} characters). Minimum required: {config_settings.MIN_CONTENT_LENGTH}")

        if content_length > config_settings.MAX_CONTENT_LENGTH:
            st.warning(f"Document is very large ({content_length} characters). Processing may take longer.")

        return {
            "final_summary": result["final_summary"],
            "entities": result["entities"],
            "document_count": len(result["contents"]),
            "content_length": content_length,
            "processing_status": "success"
        }

    except Exception as e:
        return {
            "final_summary": f"Error processing document: {str(e)}",
            "entities": {
                "entities": [],
                "dates": [],
                "key_topics": [],
                "sentiment": {"error": str(e)},
                "relationships": []
            },
            "document_count": 0,
            "content_length": 0,
            "processing_status": "error",
            "error_message": str(e)
        }


def display_results(results: Dict[str, Any]):
    processing_status = results.get("processing_status", "unknown")

    if processing_status == "error":
        st.error("❌ Document Processing Failed")
        st.error(results.get("error_message", "Unknown error occurred"))
        return

    st.success("✅ Document Processing Completed Successfully")

    content_length = results.get("content_length", 0)
    document_count = results.get("document_count", 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Document Segments", document_count)
    with col2:
        st.metric("Content Length", f"{content_length:,} chars")
    with col3:
        processing_time = "N/A"
        st.metric("Status", "✅ Success")

    st.subheader("📋 Document Summary")
    final_summary = results.get("final_summary", "")
    if final_summary and final_summary.strip():
        if final_summary.startswith("Error"):
            st.error(final_summary)
        else:
            st.write(final_summary)

            summary_length = len(final_summary)
            st.caption(f"Summary length: {summary_length} characters")
    else:
        st.warning("No summary was generated")

    st.subheader("🏷️ Extracted Entities")
    entities = results.get("entities", {})

    if not entities:
        st.warning("No entities were extracted")
        return

    if isinstance(entities, dict):
        entity_tabs = st.tabs(
            ["📊 Overview", "👥 People", "🏢 Organizations", "📍 Locations", "📅 Dates", "🎯 Events", "💡 Key Topics",
             "😊 Sentiment", "🔗 Relationships"])

        with entity_tabs[0]:
            total_entities = sum(len(v) if isinstance(v, list) else (1 if v else 0) for v in entities.values())
            st.metric("Total Entities Found", total_entities)

            if entities.get("key_topics"):
                st.write("**Main Topics:**")
                for topic in entities["key_topics"][:5]:
                    st.write(f"• {topic}")

        entity_sections = [
            ("entities", "👥 People & General Entities", entity_tabs[1]),
            ("organizations", "🏢 Organizations", entity_tabs[2]) if "organizations" in entities else ("entities",
                                                                                                      "🏢 Organizations",
                                                                                                      entity_tabs[2]),
            ("locations", "📍 Locations", entity_tabs[3]) if "locations" in entities else ("entities", "📍 Locations",
                                                                                          entity_tabs[3]),
            ("dates", "📅 Dates", entity_tabs[4]),
            ("events", "🎯 Events", entity_tabs[5]) if "events" in entities else ("entities", "🎯 Events",
                                                                                 entity_tabs[5]),
            ("key_topics", "💡 Key Topics", entity_tabs[6]),
            ("sentiment", "😊 Sentiment Analysis", entity_tabs[7]),
            ("relationships", "🔗 Relationships", entity_tabs[8])
        ]

        for key, title, tab in entity_sections:
            with tab:
                value = entities.get(key, [])
                if value:
                    if isinstance(value, list):
                        if len(value) == 0:
                            st.info("No items found in this category")
                        else:
                            for i, item in enumerate(value, 1):
                                if isinstance(item, dict):
                                    st.json(item)
                                else:
                                    st.write(f"{i}. {item}")
                    elif isinstance(value, dict):
                        if not value:
                            st.info("No information available")
                        elif "error" in value:
                            st.error(f"Error: {value['error']}")
                        else:
                            st.json(value)
                    else:
                        st.write(value)
                else:
                    st.info("No items found in this category")
    else:
        st.json(entities)

    if document_count > 0:
        st.info(f"📄 Successfully processed {document_count} document segments containing {content_length:,} characters")
    else:
        st.warning("No document segments were processed")


def main():
    st.title("📄 Document Summarization & Entity Extraction")
    st.write("Upload a document to get an AI-powered summary and extract key entities")

    upload_dir = create_upload_directory()

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'pdf', 'docx', 'xlsx', 'csv'],
        help="Supported formats: TXT, PDF, DOCX, XLSX, CSV"
    )

    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")

        if st.button("Process Document", type="primary"):
            try:
                with st.spinner("Saving file..."):
                    file_path, file_name, file_type = save_uploaded_file(uploaded_file, upload_dir)

                with st.spinner("Processing document..."):
                    results = asyncio.run(process_document(file_path, file_name, file_type))

                st.success("Processing completed!")
                display_results(results)

            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

            finally:
                if 'file_path' in locals() and os.path.exists(file_path):
                    os.remove(file_path)


if __name__ == "__main__":
    main()
