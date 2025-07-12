from typing import Literal

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from workflows.agents.models import EntityExtractionState
from workflows.agents.tools import (
    generate_summary,
    collapse_summaries,
    generate_final_summary,
    length_function,
    load_documents,
    extract_entities
)
from workflows.settings import config_settings


def collect_summaries(state: EntityExtractionState):
    # When using Send objects, LangGraph automatically collects the results
    # The individual summary strings from generate_summary calls are collected
    # into a list and passed to this function
    summaries = state.get("summaries", [])

    # If summaries is empty, it might be that the results are in a different format
    # Let's check if we have individual summary results to collect
    if not summaries:
        # This shouldn't happen with proper Send object handling, but let's be safe
        summaries = []

    state["collapsed_summaries"] = [
        Document(page_content=summary) for summary in summaries if summary
    ]
    return state


def should_collapse_summaries(
        state: EntityExtractionState
) -> Literal["collapse_summaries", "generate_final_summary"]:
    summaries = state.get("collapsed_summaries", [])
    num_tokens = length_function(summaries)
    return (
        "collapse_summaries"
        if num_tokens > config_settings.MAXIMUM_TOKEN_FOR_SUMMARIZATION
        else "generate_final_summary"
    )


def get_entities_extraction_graph():
    graph = StateGraph(EntityExtractionState)

    graph.add_node('load_documents', load_documents)
    graph.add_node('generate_summary', generate_summary)
    graph.add_node('collect_summaries', collect_summaries)
    graph.add_node('collapse_summaries', collapse_summaries)
    graph.add_node('generate_final_summary', generate_final_summary)
    graph.add_node('extract_entities', extract_entities)

    # Simplified workflow without Send objects
    graph.add_edge(START, "load_documents")
    graph.add_edge("load_documents", "generate_summary")
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges(
        "collect_summaries", should_collapse_summaries
    )
    graph.add_conditional_edges(
        "collapse_summaries", should_collapse_summaries
    )
    graph.add_edge("generate_final_summary", "extract_entities")
    graph.add_edge("extract_entities", END)

    return graph.compile()


if __name__ == "__main__":
    import asyncio


    async def run_workflow():
        graph = get_entities_extraction_graph()
        state = EntityExtractionState(
            file_type='pdf',
            file_path="/Users/mohitverma/Downloads/Technical Assessment - Multi-Agent Orchestration System - Mohit Verma - Outlook.pdf",
            file_name='dummy.pdf',
            original_file_name='dummy.pdf',
            contents=[],
            final_summary="",
            entities={},
            summaries=[],
            collapsed_summaries=[],
        )
        print("Graph created successfully")
        print(f"Graph type: {type(graph)}")

        print("\nExecuting workflow...")
        try:
            result = await graph.ainvoke(state)

            print("\n" + "=" * 50)
            print("WORKFLOW EXECUTION RESULTS")
            print("=" * 50)

            if result.get("contents"):
                print(f"✅ Documents loaded: {len(result['contents'])} segments")
                total_content = sum(len(str(content)) for content in result["contents"])
                print(f"✅ Total content length: {total_content} characters")

            if result.get("final_summary"):
                print(f"✅ Summary generated: {len(result['final_summary'])} characters")
                print(f"Summary preview: {result['final_summary'][:200]}...")

            if result.get("entities"):
                print(f"✅ Entities extracted: {len(result['entities'])} categories")
                for key, value in result["entities"].items():
                    if isinstance(value, list):
                        print(f"  - {key}: {len(value)} items")
                    else:
                        print(f"  - {key}: {value}")

            print("=" * 50)
            print("✅ Workflow completed successfully!")

        except Exception as e:
            print(f"❌ Error executing workflow: {str(e)}")
            print("Graph creation was successful, but workflow execution failed.")


    asyncio.run(run_workflow())
