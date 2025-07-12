from typing import Dict, Any, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class EntityExtractionSchema(BaseModel):
    entities: List[Dict[str, Any]] = Field(
        description="List of identified entities with their types and attributes", default_factory=list)
    dates: List[Dict[str, Any]] = Field(
        description="List of extracted dates with context", default_factory=list
    )
    key_topics: List[str] = Field(
        description="Core themes or topics from the text", default_factory=list
    )
    sentiment: Dict[str, Any] = Field(
        description="Overall sentiment analysis", default_factory=dict
    )
    relationships: List[Dict[str, Any]] = Field(
        description="Contextual relationships between entities", default_factory=list
    )


class SummarySchema(BaseModel):
    summary: str = Field(
        description="Main summary text capturing key information"
    )
    key_points: List[str] = Field(
        description="List of essential points from the content", default_factory=list
    )
    topics: List[str] = Field(
        description="Main topics identified", default_factory=list
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata about the summary", default_factory=dict
    )


ENTITY_EXTRACTION_TEMPLATE = """
You are an advanced information extraction specialist with expertise in natural language processing and structured data analysis.

TASK: Extract comprehensive structured information from the provided text with high precision and completeness.

EXTRACTION GUIDELINES:
1. PEOPLE: Extract full names, titles, roles, affiliations, and any descriptive attributes
2. ORGANIZATIONS: Identify companies, institutions, agencies, departments, and organizational units
3. LOCATIONS: Extract geographical references including cities, countries, addresses, regions, and landmarks
4. DATES: Capture all temporal references with context (deadlines, events, periods, durations)
5. EVENTS: Identify meetings, conferences, incidents, milestones, and significant occurrences
6. PRODUCTS: Extract services, software, hardware, publications, and offerings with specifications
7. KEY METRICS: Numbers, percentages, financial figures, quantities, and measurements with units
8. TECHNICAL TERMS: Domain-specific terminology, acronyms, and specialized vocabulary
9. RELATIONSHIPS: Connections between entities (partnerships, hierarchies, dependencies)
10. SENTIMENT: Overall tone and emotional context of the content

QUALITY REQUIREMENTS:
- Ensure accuracy and avoid hallucination
- Provide context for ambiguous entities
- Include confidence indicators where applicable
- Maintain consistency in entity naming
- Handle edge cases like partial information gracefully

INPUT TEXT TO ANALYZE:
{text}

OUTPUT FORMAT:
Respond strictly in the following JSON schema format:
{format_instructions}
"""

SUMMARY_TEMPLATE = """
You are an expert document summarization specialist with advanced analytical capabilities and deep understanding of information synthesis.

TASK: Create a comprehensive, well-structured summary that captures the essence and critical details of the provided content.

SUMMARIZATION STRATEGY:
1. CONTENT ANALYSIS: Thoroughly analyze the input to identify main themes, arguments, and supporting details
2. INFORMATION HIERARCHY: Prioritize information based on relevance, importance, and impact
3. SYNTHESIS: Combine related concepts and eliminate redundant information while preserving meaning
4. COHERENCE: Ensure logical flow and clear connections between ideas
5. COMPLETENESS: Include all essential points, key findings, and actionable insights

QUALITY STANDARDS:
- Maintain factual accuracy and avoid interpretation bias
- Use clear, professional language appropriate for the content domain
- Preserve important numerical data, dates, and specific details
- Ensure the summary is self-contained and understandable
- Balance brevity with comprehensiveness
- Highlight critical insights and implications

CONTENT TO SUMMARIZE:
{context}

SPECIAL INSTRUCTIONS:
- If content contains technical information, preserve technical accuracy
- For business documents, emphasize decisions, actions, and outcomes
- For research content, highlight methodology, findings, and conclusions
- For narrative content, maintain chronological or logical sequence

OUTPUT REQUIREMENTS:
Provide your response strictly following this JSON schema:
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
