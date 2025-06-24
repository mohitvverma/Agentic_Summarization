from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
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


class OrchestratorAgentSchema(BaseModel):
    next_action: str = Field(
        description="The next action to take in the document processing workflow",
        examples=["load_document", "extract_document_entities", "summarize_document", "next_document", "end"]
    )
    reasoning: str = Field(
        description="Explanation for why this action was chosen"
    )


ORCHESTRATOR_AGENT_TEMPLATE = """
You are an intelligent document processing orchestrator. Your job is to determine the next step in processing documents based on the current state and user instructions.

Current State:
- Document Index: {current_document_index} of {total_documents}
- Current Status: {status}
- Extract Entities: {extract_entities}
- Entities Already Extracted: {entities_extracted}
- Document Already Summarized: {document_summarized}

User Instructions: {instructions}

Available Actions:
- load_document: Load the next document for processing
- extract_document_entities: Extract entities from the current document (only if not already done)
- summarize_document: Generate a summary of the current document (only if not already done)
- next_document: Move to the next document
- end: End the processing workflow

Processing Guidelines:
1. Do not extract entities more than once for the same document
2. Do not summarize a document more than once
3. After both entity extraction and summarization are complete, move to the next document
4. If all documents are processed, end the workflow

Based on the current state and user instructions, determine the most appropriate next action.

{format_instructions}
"""


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


def initialize_orchestrator_agent_prompt() -> PromptTemplate:
    try:
        parser = JsonOutputParser(pydantic_object=OrchestratorAgentSchema)
        return PromptTemplate(
            template=ORCHESTRATOR_AGENT_TEMPLATE,
            input_variables=[
                "current_document_index", 
                "total_documents", 
                "status", 
                "extract_entities", 
                "entities_extracted", 
                "document_summarized", 
                "instructions"
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()},
            output_parser=parser
        )
    except Exception as e:
        raise ValueError(f"Failed to initialize orchestrator agent prompt: {str(e)}")
