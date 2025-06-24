from typing import List, Dict, Any, Annotated, Literal, Optional, Union, TypedDict
from pydantic import BaseModel, Field
import operator
from langchain_core.documents import Document


class EntityExtraction(BaseModel):
    entities: List[Dict[str, Any]] = Field(
        description="List of extracted entities with their types and attributes"
    )
    dates: List[Dict[str, Any]] = Field(
        description="List of dates mentioned in the document with context"
    )
    key_topics: List[str] = Field(
        description="List of key topics or themes identified in the document"
    )
    sentiment: Dict[str, Any] = Field(
        description="Overall sentiment analysis of the document"
    )
    relationships: List[Dict[str, Any]] = Field(
        description="Relationships identified between entities"
    )


class SummaryOutput(BaseModel):
    summary: str = Field(
        description="The main summary text"
    )
    key_points: List[str] = Field(
        description="List of key points extracted from the document",
        default_factory=list
    )
    topics: List[str] = Field(
        description="Main topics covered in the document",
        default_factory=list
    )
    metadata: Dict[str, Any] = Field(
        description="Additional metadata about the summary",
        default_factory=dict
    )

class SummaryState(BaseModel):
    """State for a single document summary task"""
    content: str = Field(description="Content to summarize")


class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


class SummaryState(TypedDict):
    content: str


class OverallSummaryState(BaseModel):
    """Overall state for the summarization graph"""
    contents: List[str] = Field(description="List of document contents to summarize")
    summaries: Annotated[List[str], operator.add] = Field(
        description="List of generated summaries", default_factory=list
    )
    collapsed_summaries: List[Dict[str, Any]] = Field(
        description="List of collapsed summaries as Document objects", default_factory=list
    )
    final_summary: Optional[Union[str, Dict[str, Any], SummaryOutput]] = Field(
        description="Final generated summary", default=None
    )


class DocumentInfo(BaseModel):
    """Information about a document to process"""
    file_path: str = Field(description="Path to the document file")
    file_type: str = Field(description="Type of the document file")
    file_name: str = Field(description="Name of the document file")
    original_file_name: str = Field(description="Original name of the document file")
    content: Optional[List[Any]] = Field(
        description="Document content as a list of Document objects", default=None
    )


class OrchestratorState(BaseModel):
    """State for the document processing orchestrator"""
    documents: List[DocumentInfo] = Field(
        description="List of documents to process", default_factory=list
    )
    extract_entities: bool = Field(
        description="Whether to extract entities from documents", default=True
    )
    token_max: int = Field(
        description="Maximum tokens for summarization chunks", default=1000
    )
    current_document_index: int = Field(
        description="Index of the current document being processed", default=0
    )
    results: List[Dict[str, Any]] = Field(
        description="List of processing results", default_factory=list
    )
    status: str = Field(
        description="Status of the orchestration process", default="processing"
    )
    error: Optional[str] = Field(
        description="Error message if processing failed", default=None
    )
