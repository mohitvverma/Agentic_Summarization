from typing import Dict, Any, List, Union, Literal
from pathlib import Path

from loguru import logger
from domains.workflows.models import OrchestratorState, DocumentInfo
from domains.workflows.tool import load_document, extract_document_entities, summarize_document, process_images
from domains.workflows.utils import get_attribute, create_result_dict, extract_summary_text
from langgraph.graph import END, START, StateGraph


def next_document(state: OrchestratorState):
    return {
        "current_document_index": state.current_document_index + 1,
        "results": get_attribute(state, "results", [])
    }


def decide_next_action(state: OrchestratorState) -> Literal["extract_document_entities", "summarize_document", "next_document", "end"]:
    status = get_attribute(state, "status")

    if status == "error":
        return "end"

    if status == "document_loaded":
        if get_attribute(state, "extract_entities", True):
            return "extract_document_entities"
        else:
            return "summarize_document"
    elif status == "entities_extracted":
        return "summarize_document"
    elif status == "document_summarized":
        return "next_document"
    elif status == "completed":
        return "end"
    else:
        return "end"


def check_more_documents(state: OrchestratorState) -> Literal["load_document", "end"]:
    if state.current_document_index < len(state.documents):
        return "load_document"
    else:
        return "end"


def create_orchestrator_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("load_document", load_document)
    graph.add_node("extract_document_entities", extract_document_entities)
    graph.add_node("summarize_document", summarize_document)
    graph.add_node("next_document", next_document)

    graph.add_edge(START, "load_document")

    conditional_edges = {
        "extract_document_entities": "extract_document_entities",
        "summarize_document": "summarize_document",
        "next_document": "next_document",
        "end": END
    }

    graph.add_conditional_edges("load_document", decide_next_action, conditional_edges)
    graph.add_conditional_edges("extract_document_entities", decide_next_action, conditional_edges)
    graph.add_conditional_edges("summarize_document", decide_next_action, conditional_edges)
    graph.add_conditional_edges("next_document", check_more_documents, {
        "load_document": "load_document",
        "end": END
    })

    return graph.compile()


async def run_orchestrator_graph(
    file_paths: Union[str, List[str]] = None,
    extract_entities: bool = True,
    token_max: int = 1000,
    images_path: Union[str, List[str]] = None
) -> Dict[str, Any]:
    try:
        documents = []

        if file_paths:
            if isinstance(file_paths, str):
                file_paths = [file_paths]

            logger.info(f"Processing {len(file_paths)} document file(s)")

            for file_path in file_paths:
                file_type = str(Path(file_path).suffix.lower().replace('.', ''))
                file_name = str(Path(file_path).name)
                original_file_name = str(Path(file_path).stem)

                documents.append(DocumentInfo(
                    file_path=file_path,
                    file_type=file_type,
                    file_name=file_name,
                    original_file_name=original_file_name
                ))

        # Normalize images_path to a list if it's a string
        if images_path and isinstance(images_path, str):
            images_path = [images_path]

        if images_path:
            logger.info(f"Processing {len(images_path)} image file(s)")

        initial_state = OrchestratorState(
            documents=documents,
            extract_entities=extract_entities,
            token_max=token_max,
            images_path=images_path
        )

        # Process images if available
        image_documents = []
        if images_path:
            logger.info("Processing images before document workflow")
            image_state = await process_images(initial_state)

            # Update initial state with image documents
            if "documents" in image_state:
                initial_state.documents = image_state["documents"]

            # Store image documents for later use
            if "image_documents" in image_state:
                image_documents = image_state["image_documents"]

        orchestrator_graph = create_orchestrator_graph()
        final_state = None
        intermediate_results = []

        async for step in orchestrator_graph.astream(
            initial_state.dict(),
            {"recursion_limit": 20},
        ):
            final_state = step

            results = get_attribute(step, "results", [])
            if results:
                intermediate_results = results

        if final_state and get_attribute(final_state, "status") != "error":
            results = []

            next_document_results = get_attribute(get_attribute(final_state, "next_document", {}), "results", [])
            direct_results = get_attribute(final_state, "results", [])

            if next_document_results:
                results = next_document_results
            elif direct_results:
                results = direct_results
            elif intermediate_results:
                results = intermediate_results

            # If we have no results but we processed image documents, check if they were processed by the workflow
            if len(results) == 0 and image_documents and len(image_documents) > 0:
                # The image documents should have been processed by the workflow
                # If not, they might be in the intermediate_results
                if intermediate_results:
                    results = intermediate_results

            # Check if we have a consolidated image document in the final state
            consolidated_image_document = get_attribute(final_state, "consolidated_image_document")
            if consolidated_image_document:
                logger.info("Found consolidated image document in final state")
                # Create a result from the consolidated document
                consolidated_doc = get_attribute(consolidated_image_document, "content", [])[0]
                if consolidated_doc:
                    # Extract metadata from the consolidated document
                    metadata = {}
                    if hasattr(consolidated_doc, 'metadata'):
                        metadata = consolidated_doc.metadata
                    elif isinstance(consolidated_doc, dict) and "metadata" in consolidated_doc:
                        metadata = consolidated_doc["metadata"]

                    # Create a result with the consolidated information
                    consolidated_result = create_result_dict(
                        file_path="multiple_images",
                        file_type="json",
                        status="success",
                        summary=consolidated_doc.page_content if hasattr(consolidated_doc, 'page_content') else consolidated_doc.get("page_content", ""),
                        metadata={
                            "is_consolidated": True,
                            "image_count": metadata.get("image_count", len(image_documents)),
                            "product_name": metadata.get("product_name", ""),
                            "quantity_estimation": metadata.get("quantity_estimation", ""),
                            "confidence_level": metadata.get("confidence_level", "")
                        }
                    )

                    # Add key details if available
                    if "key_details" in metadata:
                        consolidated_result["key_details"] = metadata["key_details"]

                    # Replace results with just the consolidated result
                    results = [consolidated_result]

            logger.info(f"Processing completed with {len(results)} result(s)")

            if len(results) == 0:
                # Default file path and type if no file_paths provided
                file_path = None
                file_type = "unknown"

                if file_paths:
                    file_path = file_paths[0] if isinstance(file_paths, list) else file_paths
                    file_type = str(Path(file_path).suffix.lower().replace('.', ''))
                elif images_path:
                    # Use the first image path if no file_paths provided
                    file_path = images_path[0] if isinstance(images_path, list) else images_path
                    file_type = str(Path(file_path).suffix.lower().replace('.', ''))

                summary = None
                metadata = {}
                entities = None

                for attr in ["summary", "final_summary"]:
                    if get_attribute(final_state, attr):
                        summary = get_attribute(final_state, attr)
                        if attr == "final_summary":
                            summary = extract_summary_text(summary)
                        break

                entities = get_attribute(final_state, "entities")
                if not entities:
                    entities = get_attribute(get_attribute(final_state, "extract_document_entities", {}), "entities")

                metadata = get_attribute(final_state, "metadata", {})

                if not summary and hasattr(final_state, "documents"):
                    for doc in get_attribute(final_state, "documents", []):
                        content = get_attribute(doc, "content", [])
                        if content and len(content) > 0:
                            result = create_result_dict(
                                file_path=get_attribute(doc, "file_path", file_path),
                                file_type=get_attribute(doc, "file_type", file_type),
                                summary="Document processed successfully",
                                metadata={"document_count": len(content)},
                                entities=entities
                            )
                            results = [result]
                            break

                if summary and len(results) == 0:
                    result = create_result_dict(
                        file_path=file_path,
                        file_type=file_type,
                        summary=summary,
                        metadata=metadata,
                        entities=entities
                    )
                    results = [result]

                if len(results) == 0:
                    logger.warning("Could not find any summary or document content in final state")

                    # Check if we have image documents first
                    if image_documents and len(image_documents) > 0:
                        # Create a result for the first image document
                        img_doc = image_documents[0]
                        result = create_result_dict(
                            file_path=img_doc.file_path,
                            file_type=img_doc.file_type,
                            status="success",
                            summary="Image processed successfully"
                        )

                        # If the document has content, add it to the result
                        if img_doc.content and len(img_doc.content) > 0:
                            doc = img_doc.content[0]
                            # Check if doc is a Document object or a dict
                            if hasattr(doc, 'page_content'):
                                result["summary"] = doc.page_content
                                if hasattr(doc, 'metadata') and "key_points" in doc.metadata:
                                    result["key_points"] = doc.metadata["key_points"]
                            elif isinstance(doc, dict) and "page_content" in doc:
                                result["summary"] = doc["page_content"]
                                if "metadata" in doc and "key_points" in doc["metadata"]:
                                    result["key_points"] = doc["metadata"]["key_points"]

                        results = [result]
                    else:
                        # Create a default result if no image results
                        result = create_result_dict(
                            file_path=file_path,
                            file_type=file_type,
                            summary="Document processed successfully, but no summary was generated"
                        )

                        for intermediate_result in intermediate_results:
                            if isinstance(intermediate_result, dict):
                                if intermediate_result.get("summary"):
                                    result["summary"] = intermediate_result.get("summary")

                                    for field in ["summary_json", "key_points", "topics", "products", "quantities"]:
                                        if intermediate_result.get(field):
                                            result[field] = intermediate_result.get(field)

                                    if intermediate_result.get("metadata"):
                                        result["metadata"].update(intermediate_result.get("metadata"))

                                if not entities and intermediate_result.get("entities"):
                                    entities = intermediate_result.get("entities")

                        if entities:
                            result["entities"] = entities

                        results = [result]

            # The image documents should have been processed by the workflow
            # No need to add them separately as they're already included in the results

            if len(results) == 1:
                return results[0]
            else:
                return {
                    "status": "success",
                    "results": results,
                    "count": len(results)
                }
        else:
            return {
                "status": "error",
                "message": get_attribute(final_state, "error", "Unknown error in orchestration")
            }

    except Exception as e:
        logger.error(f"Error in run_orchestrator_graph: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
