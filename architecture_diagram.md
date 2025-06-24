# Agentic Document Intelligence System - Architecture Diagram

```
+---------------------+
|                     |
|  User Interface     |
|  (Streamlit)        |
|                     |
+----------+----------+
           |
           v
+----------+----------+
|                     |
|  Orchestrator       |
|  Component          |
|                     |
+-----+-------+-------+
      |       |
      |       |
      v       v
+-----+--+ +--+------+
|        | |         |
|Extract | |Summarize|
|Component| |Component|
|        | |         |
+--------+ +---------+
```

## Component Descriptions

### User Interface (Streamlit)
- Provides a web-based interface for users to upload documents
- Displays processing results including summaries and extracted entities
- Allows configuration of processing options

### Orchestrator Component
- Coordinates the document processing workflow
- Routes documents to appropriate components
- Manages state and results throughout processing

### Extract Component
- Extracts entities, dates, and relationships from documents
- Identifies key topics and themes
- Performs sentiment analysis on document content

### Summarize Component
- Implements map-reduce approach for document summarization
- Breaks documents into manageable chunks
- Generates concise summaries of document content

## Data Flow

1. User uploads document(s) through the UI
2. Orchestrator loads and prepares documents
3. Documents are processed by Extract and Summarize components
4. Results are collected and formatted by the Orchestrator
5. UI displays the processed results to the user