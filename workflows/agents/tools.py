from typing import List

from langchain.chains.combine_documents import (
    acollapse_docs,
    split_list_of_docs
)
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from loguru import logger

from workflows.agents.models import EntityExtractionState
from workflows.agents.prompts import initialize_entity_extraction_prompt
from workflows.loader.routes import file_loader
from workflows.settings import config_settings
from workflows.utils import initialize_chat_model


async def load_documents(
        state: EntityExtractionState,
):
    file_paths = state.get("file_path", [])
    file_name = state.get("file_name", [])
    file_type = state.get("file_type", [])
    original_file_name = state.get("original_file_name", [])

    try:
        contents = file_loader(
            pre_signed_url=file_paths,
            file_name=file_name,
            original_file_name=original_file_name,
            file_type=file_type
        )

        state["contents"] = contents
        return state

    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise ValueError(f"Error loading documents: {str(e)}")


def length_function(documents: List[Document]) -> int:
    llm = initialize_chat_model()

    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)


async def generate_summary(state: EntityExtractionState):
    try:
        # Handle both full state (with contents array) and Send object (with single context)
        if "context" in state:
            # Called via Send object - single content piece
            content = state.get("context", "")
            if not content or not content:
                logger.warning("No context provided for summary generation")
                return ""

            llm = initialize_chat_model()
            if not llm:
                raise ValueError("Failed to initialize language model")

            SUMMARIZE_TEMPLATE = """
            You are a professional document summarization expert with expertise in content analysis and synthesis.

            TASK: Create a concise yet comprehensive summary of the provided content segment.

            GUIDELINES:
            - Capture the main ideas, key points, and essential information
            - Maintain factual accuracy and preserve important details
            - Use clear, professional language
            - Ensure the summary is self-contained and coherent
            - Preserve critical data, names, dates, and numerical information
            - Eliminate redundant information while retaining substance

            CONTENT TO SUMMARIZE:
            {context}

            OUTPUT: Provide a well-structured summary that effectively represents the content's core information.
            """

            def initialize_summarize_prompt():
                return PromptTemplate(
                    template=SUMMARIZE_TEMPLATE,
                    input_variables=["context"],
                    output_parser=StrOutputParser()
                )

            summarize_chain = initialize_summarize_prompt() | llm | StrOutputParser()

            try:
                summary = await summarize_chain.ainvoke({"context": content})
                if summary and summary:
                    return summary
                else:
                    logger.warning(f"Empty summary generated for content: {content[:100]}...")
                    return "No summary could be generated for this content segment"

            except Exception as content_error:
                logger.error(f"Error summarizing content segment: {str(content_error)}")
                raise ValueError(f"Error summarizing content segment: {str(content_error)}")

        else:
            # Called with full state - multiple contents (legacy mode)
            contents = state.get("contents", [])
            if not contents:
                logger.warning("No contents provided for summary generation")
                return {"summaries": []}

            llm = initialize_chat_model()
            if not llm:
                raise ValueError("Failed to initialize language model")

            SUMMARIZE_TEMPLATE = """
            You are a professional document summarization expert with expertise in content analysis and synthesis.

            TASK: Create a concise yet comprehensive summary of the provided content segment.

            GUIDELINES:
            - Capture the main ideas, key points, and essential information
            - Maintain factual accuracy and preserve important details
            - Use clear, professional language
            - Ensure the summary is self-contained and coherent
            - Preserve critical data, names, dates, and numerical information
            - Eliminate redundant information while retaining substance

            CONTENT TO SUMMARIZE:
            {context}

            OUTPUT: Provide a well-structured summary that effectively represents the content's core information.
            """

            def initialize_summarize_prompt():
                return PromptTemplate(
                    template=SUMMARIZE_TEMPLATE,
                    input_variables=["context"],
                    output_parser=StrOutputParser()
                )

            summarize_chain = initialize_summarize_prompt() | llm | StrOutputParser()

            valid_contents = [content for content in contents if content and content]

            if not valid_contents:
                logger.warning("No valid content found for summarization")
                return {"summaries": []}

            response = []
            for content in valid_contents:
                try:
                    summary = await summarize_chain.ainvoke({"context": content})
                    if summary and summary:
                        response.append(summary)

                    else:
                        logger.warning(f"Empty summary generated for content: {content[:100]}...")
                        response.append("No summary could be generated for this content segment")

                except Exception as content_error:
                    logger.error(f"Error summarizing content segment: {str(content_error)}")
                    raise ValueError(f"Error summarizing content segment: {str(content_error)}")

            return {"summaries": response}

    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise ValueError(f"Error generating summary: {str(e)}")


async def collapse_summaries(state: EntityExtractionState):
    try:
        llm = initialize_chat_model()

        def calculate_summary_length(documents: List[Document]) -> int:
            return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

        llm = initialize_chat_model()

        collapse_summaries = state["collapsed_summaries"]

        doc_lists = split_list_of_docs(
            docs=collapse_summaries,
            length_func=calculate_summary_length,
            token_max=config_settings.MAX_SUMMARY_TOKENS,
        )

        results = [
            await acollapse_docs(doc_list, llm) for doc_list in doc_lists
        ]

        logger.info(f"Resuls: {results}")
        # Ensure results are proper Document objects with page_content attribute
        collapsed_docs = []
        for result in results:
            if isinstance(result, Document):
                collapsed_docs.append(result)
            elif isinstance(result, str):
                collapsed_docs.append(Document(page_content=result))
            else:
                # Handle other potential return types
                collapsed_docs.append(Document(page_content=str(result)))

        return {
            "collapsed_summaries": collapsed_docs
        }

    except Exception as e:
        logger.error(f"Error collapsing summaries: {str(e)}")
        raise e


async def generate_final_summary(state: EntityExtractionState):
    try:
        if not state.get("collapsed_summaries"):
            logger.warning("No collapsed summaries available for final summary generation")
            return {"final_summary": "No content available for summarization"}

        llm = initialize_chat_model()
        if not llm:
            raise ValueError("Failed to initialize language model")

        collapse_summaries = state["collapsed_summaries"]

        MAP_REDUCE_TEMPLATE = """
        You are an expert document analyst tasked with creating a comprehensive final summary.

        TASK: Synthesize the following summaries into a single, coherent, and comprehensive document summary.

        REQUIREMENTS:
        - Maintain all critical information and key insights
        - Ensure logical flow and coherence
        - Eliminate redundancies while preserving important details
        - Create a self-contained summary that captures the document's essence
        - Preserve important data, dates, and specific details

        SUMMARIES TO COMBINE:
        {context}

        OUTPUT: Provide a well-structured, comprehensive summary that effectively represents the entire document.
        """

        def initialize_map_reduce_prompt():
            return PromptTemplate(
                template=MAP_REDUCE_TEMPLATE,
                input_variables=["context"],
                output_parser=StrOutputParser()
            )

        map_chain = initialize_map_reduce_prompt() | llm | StrOutputParser()

        all_docs = "\n\n".join(
            [doc.page_content for doc in collapse_summaries if doc.page_content.strip()]
        )

        if not all_docs:
            logger.warning("No valid content found in collapsed summaries")
            return {"final_summary": "No valid content available for summarization"}

        response = await map_chain.ainvoke({"context": all_docs})

        if not response or not response:
            logger.error("Empty response from final summary generation")
            return {"final_summary": "Failed to generate summary - empty response"}

        state["final_summary"] = response
        return state

    except Exception as e:
        logger.error(f"Error generating final summary: {str(e)}")
        raise ValueError(f"Error generating final summary: {str(e)}")


async def extract_entities(state: EntityExtractionState):
    try:
        final_summary = state.get("final_summary", "")
        if not final_summary or not final_summary:
            logger.warning("No final summary available for entity extraction")
            return state

        if len(final_summary) < config_settings.MIN_CONTENT_LENGTH:
            logger.warning(f"Summary too short for meaningful entity extraction: {len(final_summary)} characters")
            return state

        llm = initialize_chat_model()
        if not llm:
            raise ValueError("Failed to initialize language model for entity extraction")

        chain = initialize_entity_extraction_prompt() | llm | JsonOutputParser()

        response = await chain.ainvoke({"text": final_summary})

        if not response:
            logger.error("Empty response from entity extraction")
            return state

        if not isinstance(response, dict):
            logger.error(f"Invalid response format from entity extraction: {type(response)}")
            return state

        required_keys = ["entities", "dates", "key_topics", "sentiment", "relationships"]
        for key in required_keys:
            if key not in response:
                response[key] = [] if key != "sentiment" else {}

        state["entities"] = response
        return state

    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        raise ValueError(f"Error extracting entities: {str(e)}")
