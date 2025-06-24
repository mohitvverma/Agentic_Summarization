import operator
import json
from typing import List, Literal, Dict, Any, Optional, Union, cast, TypeVar, Generic

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
    DocumentInfo,
    OrchestratorState,
    OverallState,
    EntityExtraction
)
from domains.workflows.prompt import initialize_entity_extraction_prompt, initialize_summary_prompt
from domains.doc_loader.routes import file_loader
from domains.workflows.utils import (
    create_result_dict,
    extract_summary_text,
    get_attribute
)


def create_summarization_graph(token_max: int = 1000):
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
        content = get_attribute(state, "content", "")
        response = await map_chain.ainvoke({"context": content})
        return {"summaries": [response]}

    def map_summaries(state: OverallState):
        contents = get_attribute(state, "contents", [])
        return [
            Send("generate_summary", {"content": content}) for content in contents
        ]

    def collect_summaries(state: OverallState):
        summaries = get_attribute(state, "summaries", [])
        return {
            "collapsed_summaries": [Document(page_content=summary) for summary in summaries]
        }

    async def collapse_summaries(state: OverallState):
        collapsed_summaries = get_attribute(state, "collapsed_summaries", [])
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
        collapsed_summaries = get_attribute(state, "collapsed_summaries", [])
        num_tokens = length_function(collapsed_summaries)
        if num_tokens > token_max:
            return "collapse_summaries"
        else:
            return "generate_final_summary"

    async def generate_final_summary(state: OverallState):
        collapsed_summaries = get_attribute(state, "collapsed_summaries", [])
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
        if get_attribute(state, "status") == "error":
            return {}

        current_doc = state.documents[state.current_document_index]
        if not get_attribute(current_doc, "content"):
            return {"error": "No document content to extract entities from", "status": "error"}

        entities_result = await extract_entities(current_doc.content)

        result = create_result_dict(
            file_path=current_doc.file_path,
            file_type=current_doc.file_type,
            status="processing",
            entities={
                "status": get_attribute(entities_result, "status", "success"),
                "entities": get_attribute(entities_result, "entities", []),
                "dates": get_attribute(entities_result, "dates", []),
                "key_topics": get_attribute(entities_result, "key_topics", []),
                "sentiment": get_attribute(entities_result, "sentiment", {}),
                "relationships": get_attribute(entities_result, "relationships", [])
            }
        )

        updated_results = get_attribute(state, "results", []).copy() if get_attribute(state, "results") else []
        if state.current_document_index < len(updated_results):
            updated_results[state.current_document_index].update(result)
        else:
            updated_results.append(result)

        return {"results": updated_results, "status": "entities_extracted"}
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        return {"error": str(e), "status": "error"}


async def summarize_document(state: OrchestratorState):
    try:
        if get_attribute(state, "status") == "error":
            return {}

        current_doc = state.documents[state.current_document_index]
        if not get_attribute(current_doc, "content"):
            return {"error": "No document content to summarize", "status": "error"}

        contents = [doc.page_content for doc in current_doc.content]
        summarization_graph = create_summarization_graph(token_max=state.token_max)
        final_state = None

        try:
            logger.info(f"Summarizing document: {current_doc.file_path}")
            async for step in summarization_graph.astream(
                {"contents": contents},
                {"recursion_limit": 10},
            ):
                final_state = step
        except Exception as e:
            logger.error(f"Error running summarization graph: {str(e)}")
            return {
                "results": get_attribute(state, "results", []),
                "status": "error",
                "error": f"Error running summarization graph: {str(e)}"
            }

        final_summary = get_attribute(final_state, "final_summary")

        metadata = {
            "document_count": len(current_doc.content),
            "total_tokens": sum(initialize_chat_model().get_num_tokens(doc.page_content) for doc in current_doc.content),
        }

        summary_result = create_result_dict(
            file_path=current_doc.file_path,
            file_type=current_doc.file_type,
            status="success",
            metadata=metadata
        )

        if final_summary is not None:
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
            summary_result["status"] = "error"
            summary_result["error"] = "No final summary generated"
            return {
                "results": get_attribute(state, "results", []),
                "status": "error",
                "error": "No final summary generated"
            }

        updated_results = get_attribute(state, "results", []).copy() if get_attribute(state, "results") else []
        if state.current_document_index < len(updated_results):
            current_result = updated_results[state.current_document_index]
            current_result.update(summary_result)
        else:
            updated_results.append(summary_result)

        return {"results": updated_results, "status": "document_summarized"}
    except Exception as e:
        logger.error(f"Error summarizing document: {str(e)}")
        return {"error": str(e), "status": "error"}


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
                logger.info(f"Extracting entities from document chunk {i+1}/{len(documents)}")
                extraction = await extraction_chain.ainvoke({"text": doc.page_content})

                if isinstance(extraction, dict):
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

            # Process all extractions to consolidate entities
            for extraction in all_extractions:
                # Standardize extraction handling regardless of type
                entities = get_attribute(extraction, "entities", [])
                dates = get_attribute(extraction, "dates", [])
                key_topics = get_attribute(extraction, "key_topics", [])
                sentiment = get_attribute(extraction, "sentiment", {})
                relationships = get_attribute(extraction, "relationships", [])

                # Process entities
                for entity in entities:
                    if "name" in entity and "type" in entity:
                        key = f"{entity['name']}_{entity['type']}"
                        entity_map[key] = entity

                # Process dates
                for date in dates:
                    if "date" in date:
                        key = date["date"]
                        date_map[key] = date

                # Process topics
                topics.update(key_topics)

                # Process sentiment
                if sentiment and len(sentiment) > len(result["sentiment"]):
                    result["sentiment"] = sentiment

                # Process relationships
                for rel in relationships:
                    if "source" in rel and "target" in rel and "type" in rel:
                        key = f"{rel['source']}_{rel['type']}_{rel['target']}"
                        relationship_map[key] = rel

            # Update result with consolidated data
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
