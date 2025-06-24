# from langchain_core.output_parsers import JsonOutputParser
# from langchain.prompts import PromptTemplate
#
#
# ENTITY_EXTRACTION_PROMPT = """
# You are an expert in extracting structured information from unstructured text.
# Your task is to analyze the given text and identify relevant entities, organizing them into the following predefined types:
#
# entity_types = [
#     "people", "organizations", "locations", "dates", "events",
#     "products", "key_metrics", "technical_terms"
# ]
#
#
# Return your response in a well-formatted JSON object with the following fields:
# - entities: a list of identified entities, each with its type and relevant attributes
# - dates: a list of extracted dates with associated context
# - key_topics: a list of core themes or topics discussed in the text
# - sentiment: a concise analysis of the overall sentiment (e.g., positive, negative, neutral)
# - relationships: logical or contextual relationships between entities (e.g., person to organization, product to event)
#
# Analyze the following text and extract the required information:
#
# {text}
# """
#
#
# SUMMARY_PROMPT = """
# You are a summarization assistant. Provided below are multiple text segments or partial summaries:
#
# {context}
#
# Your task is to synthesize these into one concise, coherent final summary and return it as a JSON object.
#
# Make sure to:
# - Capture the most important points and insights.
# - Remove redundancy or irrelevant information.
# - Maintain a logical and readable structure.
# - Ensure clarity and completeness in the consolidated summary.
#
# Return your response as a valid JSON object with the following structure:
# {{
#   "summary": "The main summary text that captures the key information from all segments",
#   "key_points": ["Key point 1", "Key point 2", "Key point 3", ...],
#   "topics": ["Main topic 1", "Main topic 2", ...],
#   "metadata": {{
#     "source_count": number of source segments,
#     "additional_info": "Any other relevant information"
#   }}
# }}
# """
#
#
# def initialize_entity_extraction_prompt() -> PromptTemplate:
#     """
#     Initialize the entity extraction prompt template.
#
#     Returns:
#         PromptTemplate: The initialized prompt template for entity extraction.
#     """
#     return PromptTemplate(
#         input_variables=["text"],
#         template=ENTITY_EXTRACTION_PROMPT,
#         output_parser=JsonOutputParser()
#     )
#
#
# def initialize_summary_prompt() -> PromptTemplate:
#     """
#     Initialize the summary prompt template.
#     :return:
#     """
#     return PromptTemplate(
#         input_variables=["context"],
#         template=SUMMARY_PROMPT,
#         output_parser=JsonOutputParser()
#     )

from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


class EntityExtractionSchema(BaseModel):
    entities: List[Dict[str, Any]] = Field(description="List of identified entities with their types and attributes", default_factory=list)
    dates: List[Dict[str, Any]] = Field(description="List of extracted dates with context", default_factory=list)
    key_topics: List[str] = Field(description="Core themes or topics from the text", default_factory=list)
    sentiment: Dict[str, Any] = Field(description="Overall sentiment analysis", default_factory=dict)
    relationships: List[Dict[str, Any]] = Field(description="Contextual relationships between entities", default_factory=list)


class SummarySchema(BaseModel):
    summary: str = Field(description="Main summary text capturing key information")
    key_points: List[str] = Field(description="List of essential points from the content", default_factory=list)
    topics: List[str] = Field(description="Main topics identified", default_factory=list)
    metadata: Dict[str, Any] = Field(description="Additional metadata about the summary", default_factory=dict)


# Moved templates to separate constants for better maintainability
ENTITY_EXTRACTION_TEMPLATE = """
You are an expert in extracting structured information from unstructured text.
Your task is to analyze the given text and identify relevant entities.

Entity Types to Consider:
- People: Names and roles of individuals
- Organizations: Company names, institutions, groups
- Locations: Places, addresses, geographical references
- Dates: Temporal references, timeframes
- Events: Significant occurrences or happenings
- Products: Items, services, offerings
- Key Metrics: Numbers, statistics, measurements
- Technical Terms: Domain-specific terminology

Input Text:
{text}

Ensure your response follows this exact JSON schema:
{format_instructions}
"""

SUMMARY_TEMPLATE = """
You are a professional summarization assistant tasked with synthesizing multiple text segments.

Context to Summarize:
{context}

Requirements:
1. Consolidate key information into a coherent summary
2. Eliminate redundancies and preserve essential points
3. Maintain clear logical flow and readability
4. Include all crucial insights from source material

Provide your response in the following JSON schema:
{format_instructions}
"""


def initialize_entity_extraction_prompt() -> PromptTemplate:
    try:
        parser = JsonOutputParser(pydantic_object=EntityExtractionSchema)
        return PromptTemplate(
            template=ENTITY_EXTRACTION_TEMPLATE,
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            output_parser=parser
        )
    except Exception as e:
        raise ValueError(f"Failed to initialize entity extraction prompt: {str(e)}")


def initialize_summary_prompt() -> PromptTemplate:
    try:
        parser = JsonOutputParser(pydantic_object=SummarySchema)
        return PromptTemplate(
            template=SUMMARY_TEMPLATE,
            input_variables=["context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            output_parser=parser
        )
    except Exception as e:
        raise ValueError(f"Failed to initialize summary prompt: {str(e)}")
