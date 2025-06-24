import operator
import json
from typing import List, Literal, Dict, Any, Optional, Union, cast

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pathlib import Path

from domains.utils import initialize_chat_model
from domains.workflows.models import (
    SummaryOutput, 
    SummaryState, 
    OverallSummaryState,
    DocumentInfo,
    OrchestratorState,
    OverallState
)
from domains.workflows.prompt import initialize_entity_extraction_prompt, initialize_summary_prompt
from domains.doc_loader.routes import file_loader

def create_summarization_graph(token_max: int = 1000):
    """
    Create a graph for summarizing documents using a map-reduce approach.

    Args:
        token_max: Maximum number of tokens for each summary chunk

    Returns:
        A compiled StateGraph for document summarization
    """
    llm = initialize_chat_model()

    map_prompt = ChatPromptTemplate.from_messages(
        [("human", "Write a concise summary of the following:\n\n{context}")]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    reduce_template = """
    The following is a set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary
    of the main themes.
    """
    reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    def length_function(documents: List[Document]) -> int:
        return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

    async def generate_summary(state: SummaryState):
        if hasattr(state, "content"):
            content = state.content
        else:
            content = state["content"]
        response = await map_chain.ainvoke({"context": content})
        return {"summaries": [response]}

    def map_summaries(state: OverallState):
        if hasattr(state, "contents"):
            contents = state.contents
        else:
            contents = state["contents"]
        return [
            Send("generate_summary", {"content": content}) for content in contents
        ]

    def collect_summaries(state: OverallState):
        if hasattr(state, "summaries"):
            summaries = state.summaries
        else:
            summaries = state["summaries"]
        return {
            "collapsed_summaries": [Document(page_content=summary) for summary in summaries]
        }

    async def collapse_summaries(state: OverallState):
        if hasattr(state, "collapsed_summaries"):
            collapsed_summaries = state.collapsed_summaries
        else:
            collapsed_summaries = state["collapsed_summaries"]
        doc_lists = split_list_of_docs(
            collapsed_summaries, length_function, token_max
        )
        results = []
        for doc_list in doc_lists:
            results.append(await acollapse_docs(doc_list, reduce_chain.ainvoke))

        return {"collapsed_summaries": results}

    def should_collapse(
        state: OverallState,
    ) -> Literal["collapse_summaries", "generate_final_summary"]:
        if hasattr(state, "collapsed_summaries"):
            collapsed_summaries = state.collapsed_summaries
        else:
            collapsed_summaries = state["collapsed_summaries"]
        num_tokens = length_function(collapsed_summaries)
        if num_tokens > token_max:
            return "collapse_summaries"
        else:
            return "generate_final_summary"

    async def generate_final_summary(state: OverallState):
        if hasattr(state, "collapsed_summaries"):
            collapsed_summaries = state.collapsed_summaries
        else:
            collapsed_summaries = state["collapsed_summaries"]

        docs_text = "\n\n".join([doc.page_content for doc in collapsed_summaries])
        response = await reduce_chain.ainvoke({"docs": docs_text})
        return {"final_summary": response}

    # Nodes:
    graph = StateGraph(OverallState)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)

    # Edges:
    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)

    return graph.compile()


def create_orchestrator_graph():
    async def load_document(state: OrchestratorState):
        try:
            if state.current_document_index >= len(state.documents):
                return {"status": "completed"}

            current_doc = state.documents[state.current_document_index]
            logger.info(f"Loading document: {current_doc.file_path}")

            documents = file_loader(
                pre_signed_url=current_doc.file_path,
                file_name=current_doc.file_name,
                original_file_name=current_doc.original_file_name,
                file_type=current_doc.file_type
            )

            updated_docs = state.documents.copy()
            updated_docs[state.current_document_index] = DocumentInfo(
                file_path=current_doc.file_path,
                file_type=current_doc.file_type,
                file_name=current_doc.file_name,
                original_file_name=current_doc.original_file_name,
                content=documents
            )

            return {"documents": updated_docs, "status": "document_loaded"}
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            return {"error": str(e), "status": "error"}

    async def extract_document_entities(state: OrchestratorState):
        try:
            if state.status == "error":
                return {}

            current_doc = state.documents[state.current_document_index]
            if not current_doc.content:
                return {"error": "No document content to extract entities from", "status": "error"}

            entities_result = await extract_entities(current_doc.content)

            result = {
                "file_path": current_doc.file_path,
                "file_type": current_doc.file_type,
                "status": "processing",
                "entities": entities_result
            }

            updated_results = state.results.copy()
            if state.current_document_index < len(updated_results):
                updated_results[state.current_document_index] = result
            else:
                updated_results.append(result)

            return {"results": updated_results, "status": "entities_extracted"}
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {"error": str(e), "status": "error"}

    async def summarize_document(state: OrchestratorState):
        try:
            logger.debug(f"Starting summarize_document with state type: {type(state)}")
            logger.debug(f"State status: {state.status}")
            logger.debug(f"State results: {state.results}")
            logger.debug(f"State results type: {type(state.results)}")
            logger.debug(f"State results length: {len(state.results)}")

            if state.status == "error":
                return {}

            current_doc = state.documents[state.current_document_index]
            if not current_doc.content:
                return {"error": "No document content to summarize", "status": "error"}

            contents = [doc.page_content for doc in current_doc.content]

            try:
                summarization_graph = create_summarization_graph(token_max=state.token_max)

                final_state = None
                logger.debug("Starting summarization graph streaming")

                async for step in summarization_graph.astream(
                    {"contents": contents},
                    {"recursion_limit": 10},
                ):
                    logger.debug(f"Received step from summarization graph: {type(step)}")
                    final_state = step
                    logger.debug(f"Using step as final_state: {type(final_state)}")

                logger.debug("Finished summarization graph streaming")
                logger.debug(f"Processing step state type: {type(final_state)}")
                logger.debug(f"Processing step state: {final_state}")
            except Exception as e:
                logger.error(f"Error running summarization graph: {str(e)}")
                return {
                    "results": state.results,
                    "status": "error",
                    "error": f"Error running summarization graph: {str(e)}"
                }

            summary_result = {
                "status": "error",
                "summary": None,
                "metadata": {
                    "document_count": len(current_doc.content),
                    "total_tokens": sum(initialize_chat_model().get_num_tokens(doc.page_content) for doc in current_doc.content),
                }
            }

            final_summary = None
            if hasattr(final_state, "final_summary"):
                final_summary = final_state.final_summary
            elif isinstance(final_state, dict) and "final_summary" in final_state:
                final_summary = final_state["final_summary"]

            if final_summary is not None:
                logger.debug(f"Final summary type: {type(final_summary)}")
                logger.debug(f"Final summary: {final_summary}")

                if isinstance(final_summary, str):
                    summary_result["summary"] = final_summary
                    summary_result["summary_json"] = {"summary": final_summary}
                elif isinstance(final_summary, SummaryOutput):
                    summary_result["summary"] = final_summary.summary
                    summary_result["key_points"] = final_summary.key_points
                    summary_result["topics"] = final_summary.topics
                    summary_result["metadata"].update(final_summary.metadata)
                    summary_result["summary_json"] = final_summary.dict()
                elif isinstance(final_summary, dict) and "summary" in final_summary:
                    summary_result["summary"] = final_summary["summary"]
                    summary_result["key_points"] = final_summary.get("key_points", [])
                    summary_result["topics"] = final_summary.get("topics", [])
                    if "metadata" in final_summary:
                        summary_result["metadata"].update(final_summary["metadata"])
                    summary_result["summary_json"] = final_summary
                else:
                    # Fallback to string representation
                    try:
                        summary_result["summary"] = str(final_summary)
                        summary_result["summary_json"] = {"summary": str(final_summary)}
                    except Exception as e:
                        logger.error(f"Error converting summary to string: {str(e)}")
                        summary_result["summary"] = "Error: Could not convert summary to string"
                        summary_result["summary_json"] = {"summary": "Error: Could not convert summary to string"}

                summary_result["status"] = "success"
            else:
                summary_result["status"] = "error"
                summary_result["error"] = "No final summary generated"

            # Update the result for this document
            updated_results = state.results.copy()
            if state.current_document_index < len(updated_results):
                current_result = updated_results[state.current_document_index]
                current_result.update({
                    "status": summary_result["status"],
                    "summary": summary_result.get("summary"),
                    "metadata": summary_result.get("metadata", {}),
                })

                if "summary_json" in summary_result:
                    current_result["summary_json"] = summary_result["summary_json"]
                if "key_points" in summary_result:
                    current_result["key_points"] = summary_result["key_points"]
                if "topics" in summary_result:
                    current_result["topics"] = summary_result["topics"]

                updated_results[state.current_document_index] = current_result
            else:
                updated_results.append({
                    "file_path": current_doc.file_path,
                    "file_type": current_doc.file_type,
                    "status": summary_result["status"],
                    "summary": summary_result.get("summary"),
                    "metadata": summary_result.get("metadata", {}),
                    "summary_json": summary_result.get("summary_json"),
                    "key_points": summary_result.get("key_points", []),
                    "topics": summary_result.get("topics", [])
                })

            return {"results": updated_results, "status": "document_summarized"}
        except Exception as e:
            logger.error(f"Error summarizing document: {str(e)}")
            return {"error": str(e), "status": "error"}

    def next_document(state: OrchestratorState):
        return {"current_document_index": state.current_document_index + 1}

    def decide_next_action(state: OrchestratorState) -> Literal["extract_document_entities", "summarize_document", "next_document", "end"]:
        if state.status == "error":
            return "end"

        if state.status == "document_loaded":
            if state.extract_entities:
                return "extract_document_entities"
            else:
                return "summarize_document"
        elif state.status == "entities_extracted":
            return "summarize_document"
        elif state.status == "document_summarized":
            return "next_document"
        elif state.status == "completed":
            return "end"
        else:
            return "end"

    def check_more_documents(state: OrchestratorState) -> Literal["load_document", "end"]:
        if state.current_document_index < len(state.documents):
            return "load_document"
        else:
            return "end"

    graph = StateGraph(OrchestratorState)
    graph.add_node("load_document", load_document)
    graph.add_node("extract_document_entities", extract_document_entities)
    graph.add_node("summarize_document", summarize_document)
    graph.add_node("next_document", next_document)

    # Define the edges
    graph.add_edge(START, "load_document")
    graph.add_conditional_edges("load_document", decide_next_action, {
        "extract_document_entities": "extract_document_entities",
        "summarize_document": "summarize_document",
        "next_document": "next_document",
        "end": END
    })
    graph.add_conditional_edges("extract_document_entities", decide_next_action, {
        "extract_document_entities": "extract_document_entities",
        "summarize_document": "summarize_document",
        "next_document": "next_document",
        "end": END
    })
    graph.add_conditional_edges("summarize_document", decide_next_action, {
        "extract_document_entities": "extract_document_entities",
        "summarize_document": "summarize_document",
        "next_document": "next_document",
        "end": END
    })
    graph.add_conditional_edges("next_document", check_more_documents, {
        "load_document": "load_document",
        "end": END
    })

    return graph.compile()


async def run_summarization_graph(documents: List[Document], token_max: int = 1000) -> Dict[str, Any]:
    """
    Run the summarization graph on a list of documents.

    Args:
        documents: List of Document objects to summarize
        token_max: Maximum number of tokens for each summary chunk

    Returns:
        Dictionary containing the final summary and metadata
    """
    result = {
        "status": "processing",
        "summary": None,
        "metadata": {
            "document_count": len(documents),
            "total_tokens": sum(initialize_chat_model().get_num_tokens(doc.page_content) for doc in documents),
        }
    }

    try:
        summarization_graph = create_summarization_graph(token_max=token_max)
        document_contents = [doc.page_content for doc in documents]
        final_state = None

        # Run the graph with the document contents
        async for step in summarization_graph.astream(
            {"contents": document_contents},
            {"recursion_limit": 10},
        ):
            final_state = step
            if isinstance(step, dict):
                logger.debug(f"Processing step: {list(step.keys())}")
            else:
                logger.debug(f"Processing step of type: {type(step)}")

        # Process the final state to extract the summary
        if final_state:
            # Check if final_state has final_summary attribute or key
            final_summary = None
            if hasattr(final_state, "final_summary"):
                final_summary = final_state.final_summary
            elif isinstance(final_state, dict) and "final_summary" in final_state:
                final_summary = final_state["final_summary"]

            if final_summary is not None:
                logger.debug(f"Final summary type: {type(final_summary)}")
                logger.debug(f"Final summary: {final_summary}")

                # Store the summary in the result
                if isinstance(final_summary, str):
                    result["summary"] = final_summary
                elif isinstance(final_summary, SummaryOutput):
                    result["summary"] = final_summary.summary
                    result["key_points"] = final_summary.key_points
                    result["topics"] = final_summary.topics
                    result["metadata"].update(final_summary.metadata)
                elif isinstance(final_summary, dict) and "summary" in final_summary:
                    result["summary"] = final_summary["summary"]
                    result["key_points"] = final_summary.get("key_points", [])
                    result["topics"] = final_summary.get("topics", [])
                    if "metadata" in final_summary:
                        result["metadata"].update(final_summary["metadata"])
                else:
                    try:
                        result["summary"] = str(final_summary)
                    except Exception as e:
                        logger.error(f"Error converting summary to string: {str(e)}")
                        result["summary"] = "Error: Could not convert summary to string"

                result["status"] = "success"
            else:
                result["status"] = "error"
                result["error"] = "No final summary generated"
        else:
            result["status"] = "error"
            result["error"] = "Failed to generate summary"

    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        result["status"] = "error"
        result["error"] = str(e)

    return result


async def extract_entities(documents: List[Document], entity_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Extract entities and other key information from documents.

    Args:
        documents: List of Document objects to analyze
        entity_types: Optional list of specific entity types to extract

    Returns:
        Dictionary containing extracted entities and metadata
    """
    llm = initialize_chat_model(temperature=0.0)  # Use zero temperature for factual extraction

    if not entity_types:
        entity_types = [
            "people", "organizations", "locations", "dates", "events", 
            "products", "key_metrics", "technical_terms"
        ]

    extraction_prompt = initialize_entity_extraction_prompt()

    try:
        extraction_chain = extraction_prompt | llm | JsonOutputParser()
    except Exception as e:
        logger.warning(f"Error creating JsonOutputParser with EntityExtraction: {str(e)}")
        extraction_chain = extraction_prompt | llm | JsonOutputParser()

    result = {
        "status": "processing",
        "entities": [],
        "dates": [],
        "key_topics": [],
        "sentiment": {},
        "relationships": [],
        "metadata": {
            "document_count": len(documents),
            "entity_types_requested": entity_types,
        }
    }

    try:
        all_extractions = []

        for i, doc in enumerate(documents):
            try:
                logger.debug(f"Extracting entities from document chunk {i+1}/{len(documents)}")
                extraction = await extraction_chain.ainvoke({"text": doc.page_content})
                logger.debug(f"Extracting entities: {extraction}")
                logger.debug(f"Extraction type: {type(extraction)}")

                if isinstance(extraction, dict):
                    from domains.workflows.models import EntityExtraction
                    extraction = EntityExtraction(
                        entities=extraction.get("entities", []),
                        dates=extraction.get("dates", []),
                        key_topics=extraction.get("key_topics", []),
                        sentiment=extraction.get("sentiment", {}),
                        relationships=extraction.get("relationships", [])
                    )

                all_extractions.append(extraction)

            except Exception as e:
                logger.error(f"Error extracting entities from document chunk {i+1}: {str(e)}")

        if all_extractions:
            entity_map = {}
            date_map = {}
            topics = set()
            relationship_map = {}

            for extraction in all_extractions:
                if isinstance(extraction, EntityExtraction):
                    # Process entities
                    for entity in extraction.entities:
                        if "name" in entity and "type" in entity:
                            key = f"{entity['name']}_{entity['type']}"
                            entity_map[key] = entity

                    for date in extraction.dates:
                        if "date" in date:
                            key = date["date"]
                            date_map[key] = date

                    topics.update(extraction.key_topics)

                    if extraction.sentiment and len(extraction.sentiment) > len(result["sentiment"]):
                        result["sentiment"] = extraction.sentiment

                    for rel in extraction.relationships:
                        if "source" in rel and "target" in rel and "type" in rel:
                            key = f"{rel['source']}_{rel['type']}_{rel['target']}"
                            relationship_map[key] = rel

                elif isinstance(extraction, dict):
                    # Process entities
                    for entity in extraction.get("entities", []):
                        if "name" in entity and "type" in entity:
                            key = f"{entity['name']}_{entity['type']}"
                            entity_map[key] = entity

                    for date in extraction.get("dates", []):
                        if "date" in date:
                            key = date["date"]
                            date_map[key] = date

                    topics.update(extraction.get("key_topics", []))

                    sentiment = extraction.get("sentiment", {})
                    if sentiment and len(sentiment) > len(result["sentiment"]):
                        result["sentiment"] = sentiment

                    for rel in extraction.get("relationships", []):
                        if "source" in rel and "target" in rel and "type" in rel:
                            key = f"{rel['source']}_{rel['type']}_{rel['target']}"
                            relationship_map[key] = rel

            result["entities"] = list(entity_map.values())
            result["dates"] = list(date_map.values())
            result["key_topics"] = list(topics)
            result["relationships"] = list(relationship_map.values())

            result["status"] = "success"
        else:
            result["status"] = "error"
            result["error"] = "No entities extracted from any document"

    except Exception as e:
        logger.error(f"Error in entity extraction: {str(e)}")
        result["status"] = "error"
        result["error"] = str(e)

    return result


async def summarize_content(documents: List[Document], token_max: int = 1000) -> Dict[str, Any]:
    """
    Summarize a list of documents using a map-reduce approach with LangGraph.

    Args:
        documents: List of Document objects to summarize
        token_max: Maximum number of tokens for each summary chunk

    Returns:
        Dictionary containing the final summary and metadata
    """
    return await run_summarization_graph(documents, token_max)


async def run_orchestrator_graph(
    file_paths: Union[str, List[str]],
    extract_entities: bool = True,
    token_max: int = 1000
) -> Dict[str, Any]:
    """
    Run the orchestrator graph to process documents.

    Args:
        file_paths: Single file path or list of file paths to process
        extract_entities: Whether to extract entities from documents
        token_max: Maximum tokens for summarization chunks

    Returns:
        Dictionary with processing results
    """
    try:
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        logger.info(f"Processing {len(file_paths)} document file(s)")

        documents = []
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

        initial_state = OrchestratorState(
            documents=documents,
            extract_entities=extract_entities,
            token_max=token_max
        )

        orchestrator_graph = create_orchestrator_graph()
        final_state = None

        async for step in orchestrator_graph.astream(
            initial_state.dict(),
            {"recursion_limit": 20},
        ):
            final_state = step

        # Process the results
        logger.debug(f"Final state: {final_state}")
        if final_state and final_state.get("status") != "error":
            results = final_state.get("results", [])
            logger.debug(f"Results from final state: {results}")
            logger.debug(f"Results type: {type(results)}")
            logger.debug(f"Results length: {len(results)}")

            if len(results) == 0 and final_state.get("summary"):
                logger.debug(f"Creating result from summary: {final_state.get('summary')}")
                results = [{
                    "file_path": file_paths[0] if isinstance(file_paths, list) else file_paths,
                    "file_type": str(Path(file_paths[0] if isinstance(file_paths, list) else file_paths).suffix.lower().replace('.', '')),
                    "status": "success",
                    "summary": final_state.get("summary"),
                    "metadata": final_state.get("metadata", {})
                }]
                logger.debug(f"Created results: {results}")

            if len(results) == 1:
                logger.debug(f"Returning single result: {results[0]}")
                return results[0]
            else:
                response = {
                    "status": "success",
                    "results": results,
                    "count": len(results)
                }
                logger.debug(f"Returning multiple results: {response}")
                return response
        else:
            return {
                "status": "error",
                "message": final_state.get("error", "Unknown error in orchestration")
            }

    except Exception as e:
        logger.error(f"Error in run_orchestrator_graph: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
