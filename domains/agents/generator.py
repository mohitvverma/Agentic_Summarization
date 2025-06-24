import json

from typing import Dict, Any, List, Union, Literal, Optional
from pathlib import Path

from loguru import logger
from domains.utils import initialize_chat_model
from domains.workflows.models import OrchestratorState, DocumentInfo
from domains.workflows.tool import load_document, extract_document_entities, summarize_document
from domains.workflows.utils import get_attribute, create_result_dict, extract_summary_text
from domains.workflows.prompt import initialize_orchestrator_agent_prompt
from langgraph.graph import END, START, StateGraph


def next_document(state: OrchestratorState):
    return {
        "current_document_index": state.current_document_index + 1,
        "results": get_attribute(state, "results", [])
    }


async def orchestrator_agent(state: OrchestratorState) -> Dict[str, Any]:
    try:
        llm = initialize_chat_model()
        prompt = initialize_orchestrator_agent_prompt()

        total_documents = len(state.documents)

        entities_extracted = False
        document_summarized = False

        results = get_attribute(state, "results", [])
        if results and state.current_document_index < len(results):
            current_result = results[state.current_document_index]

            if get_attribute(current_result, "entities"):
                entities_extracted = True

            if get_attribute(current_result, "summary"):
                document_summarized = True

        prompt_input = {
            "current_document_index": state.current_document_index + 1,  # 1-indexed for human readability
            "total_documents": total_documents,
            "status": get_attribute(state, "status", "processing"),
            "extract_entities": get_attribute(state, "extract_entities", True),
            "entities_extracted": entities_extracted,
            "document_summarized": document_summarized,
            "instructions": get_attribute(state, "instructions", "Process the documents by extracting entities and generating summaries.")
        }

        response = await llm.ainvoke(prompt.format_prompt(**prompt_input).to_string())

        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        try:
            parsed_response = json.loads(content)
            next_action = parsed_response.get("next_action")
            reasoning = parsed_response.get("reasoning", "No reasoning provided")
        except:
            if "load_document" in content.lower():
                next_action = "load_document"
            elif "extract_document_entities" in content.lower():
                next_action = "extract_document_entities"
            elif "summarize_document" in content.lower():
                next_action = "summarize_document"
            elif "next_document" in content.lower():
                next_action = "next_document"
            elif "end" in content.lower():
                next_action = "end"
            else:
                next_action = decide_next_action_static(state)
            reasoning = "Extracted from text response"

        logger.info(f"Orchestrator agent decided next action: {next_action}")
        logger.info(f"Reasoning: {reasoning}")

        return {"next_action": next_action}
    except Exception as e:
        logger.error(f"Error in orchestrator agent: {str(e)}")
        next_action = decide_next_action_static(state)
        return {"next_action": next_action}


def decide_next_action_static(state: OrchestratorState) -> Literal["extract_document_entities", "summarize_document", "next_document", "end"]:
    status = get_attribute(state, "status")

    if status == "error":
        return "end"

    # Check if entities have already been extracted and if document has been summarized
    entities_extracted = False
    document_summarized = False

    # Get the results for the current document if available
    results = get_attribute(state, "results", [])
    if results and state.current_document_index < len(results):
        current_result = results[state.current_document_index]
        # Check if entities exist in the current result
        if get_attribute(current_result, "entities"):
            entities_extracted = True
        # Check if summary exists in the current result
        if get_attribute(current_result, "summary"):
            document_summarized = True

    # If both entities have been extracted and document has been summarized, move to next document
    if entities_extracted and document_summarized:
        return "next_document"

    if status == "document_loaded":
        if get_attribute(state, "extract_entities", True) and not entities_extracted:
            return "extract_document_entities"
        elif not document_summarized:
            return "summarize_document"
        else:
            return "next_document"
    elif status == "entities_extracted":
        if not document_summarized:
            return "summarize_document"
        else:
            return "next_document"
    elif status == "document_summarized":
        if get_attribute(state, "extract_entities", True) and not entities_extracted:
            return "extract_document_entities"
        else:
            return "next_document"
    elif status == "completed":
        return "end"
    else:
        return "end"


def decide_next_action(state: OrchestratorState) -> Literal["extract_document_entities", "summarize_document", "next_document", "end"]:
    """
    Determine the next action based on the current state.
    This function checks if a next_action has been set by the orchestrator agent,
    and falls back to the static decision logic if not.
    """
    next_action = get_attribute(state, "next_action")

    if next_action in ["extract_document_entities", "summarize_document", "next_document", "end"]:
        return next_action

    # Fall back to static decision logic
    return decide_next_action_static(state)


def check_more_documents(state: OrchestratorState) -> Literal["load_document", "end"]:
    if state.current_document_index < len(state.documents):
        return "load_document"
    else:
        return "end"


def create_orchestrator_graph(use_agent: bool = True):
    """
    Create the orchestrator graph.

    Args:
        use_agent: Whether to use the dynamic orchestrator agent for decision making.
                  If False, falls back to static decision logic.

    Returns:
        The compiled graph.
    """
    graph = StateGraph(OrchestratorState)

    # Add the core processing nodes
    graph.add_node("load_document", load_document)
    graph.add_node("extract_document_entities", extract_document_entities)
    graph.add_node("summarize_document", summarize_document)
    graph.add_node("next_document", next_document)

    # Add the orchestrator agent node if enabled
    if use_agent:
        graph.add_node("orchestrator_agent", orchestrator_agent)

    # Define the edges
    graph.add_edge(START, "load_document")

    # If using the agent, route through it after each processing step
    if use_agent:
        # After loading a document, consult the agent
        graph.add_edge("load_document", "orchestrator_agent")

        # After the agent decides, route to the appropriate node
        graph.add_conditional_edges("orchestrator_agent", decide_next_action, {
            "extract_document_entities": "extract_document_entities",
            "summarize_document": "summarize_document",
            "next_document": "next_document",
            "end": END
        })
        graph.add_edge("extract_document_entities", "orchestrator_agent")

        graph.add_edge("summarize_document", "orchestrator_agent")

        graph.add_conditional_edges("next_document", check_more_documents, {
            "load_document": "load_document",
            "end": END
        })
    else:
        # Use the static decision logic if the agent is disabled
        graph.add_conditional_edges("load_document", decide_next_action_static, {
            "extract_document_entities": "extract_document_entities",
            "summarize_document": "summarize_document",
            "next_document": "next_document",
            "end": END
        })
        graph.add_conditional_edges("extract_document_entities", decide_next_action_static, {
            "extract_document_entities": "extract_document_entities",
            "summarize_document": "summarize_document",
            "next_document": "next_document",
            "end": END
        })
        graph.add_conditional_edges("summarize_document", decide_next_action_static, {
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


async def run_orchestrator_graph(
    file_paths: Union[str, List[str]],
    extract_entities: bool = True,
    token_max: int = 1000,
    instructions: Optional[str] = None,
    use_agent: bool = True
) -> Dict[str, Any]:
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

        # Initialize state and graph
        initial_state = OrchestratorState(
            documents=documents,
            extract_entities=extract_entities,
            token_max=token_max,
            instructions=instructions
        )

        # Create the orchestrator graph with or without the agent
        orchestrator_graph = create_orchestrator_graph(use_agent=use_agent)
        final_state = None
        intermediate_results = []

        # Run the graph
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

            next_document = get_attribute(final_state, "next_document", {})
            if isinstance(next_document, dict) and "results" in next_document:
                results = next_document["results"]
            else:
                results = get_attribute(final_state, "results", [])

            logger.info(f"Processing completed with {len(results)} result(s)")

            if len(results) == 0 and intermediate_results:
                results = intermediate_results

            if len(results) == 0:
                file_path = file_paths[0] if isinstance(file_paths, list) else file_paths
                file_type = str(Path(file_path).suffix.lower().replace('.', ''))

                # Check if there's a summarize_document result in the final state
                summarize_document_result = get_attribute(final_state, "summarize_document", {})
                if isinstance(summarize_document_result, dict) and "results" in summarize_document_result:
                    results = summarize_document_result["results"]
                    logger.info(f"Found summary in summarize_document result: {results}")
                    # If we found results, no need to continue with the rest of the checks
                    if results and len(results) > 0:
                        return results[0] if len(results) == 1 else {
                            "status": "success",
                            "results": results,
                            "count": len(results)
                        }

                if get_attribute(final_state, "summary"):
                    summary = get_attribute(final_state, "summary")
                    metadata = get_attribute(final_state, "metadata", {})
                    entities = get_attribute(final_state, "entities")

                    result = create_result_dict(
                        file_path=file_path,
                        file_type=file_type,
                        summary=summary,
                        metadata=metadata,
                        entities=entities
                    )
                    results = [result]

                elif get_attribute(final_state, "final_summary"):
                    summary = get_attribute(final_state, "final_summary")
                    summary_text = extract_summary_text(summary)
                    metadata = get_attribute(final_state, "metadata", {})
                    entities = get_attribute(final_state, "entities")

                    result = create_result_dict(
                        file_path=file_path,
                        file_type=file_type,
                        summary=summary_text,
                        metadata=metadata,
                        entities=entities
                    )
                    results = [result]

                elif hasattr(final_state, "documents"):
                    for doc in get_attribute(final_state, "documents", []):
                        content = get_attribute(doc, "content", [])
                        if content and len(content) > 0:
                            result = create_result_dict(
                                file_path=get_attribute(doc, "file_path", file_path),
                                file_type=get_attribute(doc, "file_type", file_type),
                                summary="Document processed successfully",
                                metadata={"document_count": len(content)},
                                entities=get_attribute(final_state, "entities")
                            )
                            results = [result]
                            break

                if len(results) == 0:
                    logger.warning("Could not find any summary or document content in final state")

                    result = create_result_dict(
                        file_path=file_path,
                        file_type=file_type,
                        summary="Document processed successfully, but no summary was generated"
                    )

                    for intermediate_result in intermediate_results:
                        if isinstance(intermediate_result, dict) and intermediate_result.get("summary"):
                            result["summary"] = intermediate_result.get("summary")

                            # Copy additional fields if available
                            for field in ["summary_json", "key_points", "topics"]:
                                if intermediate_result.get(field):
                                    result[field] = intermediate_result.get(field)

                            if intermediate_result.get("metadata"):
                                result["metadata"].update(intermediate_result.get("metadata"))
                            break

                    entities = get_attribute(final_state, "entities")
                    if not entities:
                        extract_entities_result = get_attribute(final_state, "extract_document_entities", {})
                        entities = get_attribute(extract_entities_result, "entities")

                    if not entities:
                        for intermediate_result in intermediate_results:
                            if isinstance(intermediate_result, dict) and intermediate_result.get("entities"):
                                entities = intermediate_result.get("entities")
                                break

                    if entities:
                        result["entities"] = entities

                    results = [result]

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
