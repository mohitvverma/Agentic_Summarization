import operator
import json
import re
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
from domains.settings import config_settings
from domains.workflows.models import (
    SummaryOutput,
    SummaryState,
    DocumentInfo,
    OrchestratorState,
    OverallState,
    EntityExtraction
)
from domains.workflows.prompt import initialize_entity_extraction_prompt, initialize_image_analysis_prompt, initialize_multi_image_analysis_prompt
from domains.doc_loader.routes import file_loader, process_image
from domains.workflows.utils import (
    create_result_dict,
    get_attribute,
    summary_generation_prompt
)


async def load_document(state: OrchestratorState):
    try:
        if state.current_document_index >= len(state.documents):
            return {"status": "completed"}

        current_doc = state.documents[state.current_document_index]
        logger.info(f"Loading document: {current_doc.file_path}")

        # Check if this is an image document or consolidated document that already has content
        if (current_doc.file_type in config_settings.SUPPORTED_FILE_TYPES or
            current_doc.file_path == "consolidated_analysis") and get_attribute(current_doc, "content"):
            logger.info(f"Document already has content, skipping loading: {current_doc.file_path}")
            return {"status": "document_loaded"}

        # Check if this is a consolidated analysis document
        if current_doc.file_path == "consolidated_analysis" or current_doc.file_type == "json":
            # Consolidated documents already have their content set, so we don't need to load them
            logger.info(f"Document is a consolidated analysis, skipping external loading")
            return {"status": "document_loaded"}

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
        logger.info(f"Extracted entities: {entities_result}")

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


async def summarize_document(state: OrchestratorState):
    try:
        if get_attribute(state, "status") == "error":
            return {}

        current_doc = state.documents[state.current_document_index]
        if not get_attribute(current_doc, "content"):
            return {"error": "No document content to summarize", "status": "error"}

        # Convert dictionary documents to Document objects if needed
        processed_docs = []
        for doc in current_doc.content:
            if isinstance(doc, dict) and "page_content" in doc:
                doc_content = doc["page_content"]
                doc_metadata = doc.get("metadata", {})
                processed_docs.append(Document(page_content=doc_content, metadata=doc_metadata))
            elif isinstance(doc, dict) and "content" in doc:
                # Handle nested document structure
                doc_content = doc["content"]
                doc_metadata = doc.get("metadata", {})
                processed_docs.append(Document(page_content=doc_content, metadata=doc_metadata))
            else:
                processed_docs.append(doc)

        contents = [doc.page_content for doc in processed_docs]
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
            "total_tokens": sum(initialize_chat_model().get_num_tokens(doc.page_content) for doc in processed_docs),
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


async def consolidate_image_analyses(image_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        if not image_analyses or len(image_analyses) < 2:
            # If there's only one or zero analyses, no need to consolidate
            return image_analyses[0] if image_analyses else {}

        logger.info(f"Consolidating information from {len(image_analyses)} image analyses")

        product_names = []
        quantities = []

        # Format the image analyses for the prompt
        analyses_text = []
        for i, analysis in enumerate(image_analyses):
            if 'name' in analysis:
                product_names.append(analysis['name'])
            elif 'summary' in analysis:
                # Fall back to parsing from summary if needed
                summary = analysis.get('summary', '')
                product_name_match = re.search(r'Product Name[:\s]+([^\n]+)', summary)
                if product_name_match:
                    product_names.append(product_name_match.group(1).strip())

            # Extract quantity
            if 'quantity' in analysis:
                quantities.append(analysis['quantity'])
            elif 'summary' in analysis:
                # Fall back to parsing from summary if needed
                summary = analysis.get('summary', '')
                quantity_match = re.search(r'Estimated Total Quantity[:\s]+([^\n]+)', summary)
                if quantity_match:
                    quantities.append(quantity_match.group(1).strip())

            summary = analysis.get('summary', '')
            detailed_summary = analysis.get('detailed_summary', '')

            analysis_text = f"IMAGE {i+1} ANALYSIS:\n"
            analysis_text += f"Product Name: {analysis.get('name', '')}\n"
            analysis_text += f"Quantity: {analysis.get('quantity', '')}\n"
            analysis_text += f"View: {analysis.get('image_view', '')}\n"
            analysis_text += f"Detailed Summary: {detailed_summary}\n"
            analysis_text += f"Summary: {summary}\n"

            key_points = analysis.get('key_points', [])
            if key_points:
                analysis_text += "Key Points:\n"
                for point in key_points:
                    analysis_text += f"- {point}\n"

            analyses_text.append(analysis_text)

        llm = initialize_chat_model()
        multi_image_prompt = initialize_multi_image_analysis_prompt()

        chain = multi_image_prompt | llm

        response = await chain.ainvoke(
            {
                "image_summaries": image_analyses,
            }
        )

        logger.debug("Consolidated image analysis response: {}".format(response))

        content = response.content if hasattr(response, 'content') else str(response)

        if isinstance(content, str):
            try:
                # Check for JSON code block
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                if json_match:
                    content = json_match.group(1)

                # Parse the JSON content
                consolidated_analysis = json.loads(content)
                logger.debug(f"Successfully parsed JSON content: {consolidated_analysis}")
            except Exception as e:
                logger.error(f"Error parsing JSON content: {str(e)}")
                consolidated_analysis = {
                    "unified_summary": content,
                    "key_details": []
                }
        elif isinstance(content, dict):
            # If content is already a dictionary, use it directly
            consolidated_analysis = content
        else:
            # Fallback for any other type
            consolidated_analysis = {
                "unified_summary": str(content),
                "key_details": []
            }

        # Post-process the consolidated analysis to ensure accuracy

        # Validate product name
        if product_names:
            # Count occurrences of each product name
            name_counts = {}
            for name in product_names:
                name_counts[name] = name_counts.get(name, 0) + 1

            # Use the most frequent product name
            most_common_name = max(name_counts.items(), key=operator.itemgetter(1))[0]

            # Override the product name in the consolidated analysis
            consolidated_analysis["product_name"] = most_common_name

            # Update the consolidated description and unified summary to use the correct product name
            if "consolidated_description" in consolidated_analysis:
                consolidated_analysis["consolidated_description"] = f"The product is identified as '{most_common_name}' from multiple images taken from different angles. " + \
                    "The images show the product packaging from various perspectives, allowing for a comprehensive analysis of the quantity and arrangement."

            if "unified_summary" in consolidated_analysis:
                consolidated_analysis["unified_summary"] = f"Analysis of multiple images confirms the product is '{most_common_name}'. " + \
                    "By examining the product from different angles, we can determine the quantity and arrangement with higher confidence."

        # Validate quantity estimation
        if quantities:
            # Extract numeric values from quantities where possible
            numeric_quantities = []
            for q in quantities:
                # Check if the quantity is already a numeric string
                if str(q).isdigit():
                    numeric_quantities.append(int(q))
                else:
                    # Try to extract numeric values using regex
                    nums = re.findall(r'\d+', str(q))
                    if nums:
                        numeric_quantities.extend([int(n) for n in nums])

            if numeric_quantities:
                # Use the maximum quantity as a conservative estimate
                max_quantity = max(numeric_quantities)

                # Override the quantity estimation in the consolidated analysis
                consolidated_analysis["quantity_estimation"] = str(max_quantity)

                # Add a note about the quantity estimation to the key details
                if "key_details" not in consolidated_analysis:
                    consolidated_analysis["key_details"] = []

                consolidated_analysis["key_details"].append(
                    f"Quantity estimation based on the maximum count observed across different angles: {max_quantity} boxes"
                )

        return consolidated_analysis

    except Exception as e:
        logger.error(f"Error consolidating image analyses: {str(e)}")
        return {"error": str(e)}


async def process_images(state: OrchestratorState) -> Dict[str, Any]:
    try:
        images_path = get_attribute(state, "images_path")
        if not images_path:
            return {}  # No images to process

        if isinstance(images_path, str):
            images_path = [images_path]

        logger.info(f"Processing {len(images_path)} image(s)")
        from domains.workflows.prompt import MULTI_IMAGE_ANALYSIS_TEMPLATE
        llm = initialize_chat_model()
        image_analysis_prompt = initialize_image_analysis_prompt()

        documents = get_attribute(state, "documents", []).copy()
        image_documents = []
        image_analyses = []

        for img_path in images_path:
            try:
                logger.info(f"Processing image: {img_path}")
                image_data = await process_image(
                    file_path=img_path,
                    process_type="base64",
                    image_type=Path(img_path).suffix.lower().replace('.', '')
                )

                prompt = summary_generation_prompt(
                    image_url=image_data.get("image_url", f"file://{img_path}"),
                    template=MULTI_IMAGE_ANALYSIS_TEMPLATE
                )

                response = await llm.ainvoke(prompt)
                content = response.content
                content = content.strip().replace("```json", "").replace("```", "").strip()
                logger.debug("Image analysis response summary: {}".format(content))

                if isinstance(content, str):
                    try:
                        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                        if json_match:
                            content = json_match.group(1)

                        analysis = json.loads(content)
                    except:
                        analysis = {
                            "summary": content,
                            "key_points": []
                        }
                else:
                    analysis = {
                        "summary": str(content),
                        "key_points": []
                    }

                image_analyses.append(analysis)

                detailed_description = analysis.get("detailed_description", "")
                if not detailed_description and analysis.get("summary"):
                    detailed_description = analysis.get("summary")

                if detailed_description:
                    file_type = Path(img_path).suffix.lower().replace('.', '')
                    file_name = Path(img_path).name
                    original_file_name = Path(img_path).stem

                    doc = Document(
                        page_content=detailed_description,
                        metadata={
                            "source": img_path,
                            "file_name": file_name,
                            "original_file_name": original_file_name,
                            "file_type": file_type,
                            "is_image": True,
                            "key_points": analysis.get("key_points", [])
                        }
                    )

                    image_doc_info = DocumentInfo(
                        file_path=img_path,
                        file_type=file_type,
                        file_name=file_name,
                        original_file_name=original_file_name,
                        content=[doc]
                    )

                    documents.append(image_doc_info)
                    image_documents.append(image_doc_info)
                    logger.info(f"Added image document for {img_path}")

            except Exception as e:
                logger.error(f"Error processing image {img_path}: {str(e)}")

        consolidated_result = None
        if len(image_analyses) > 1:
            logger.info(f"Consolidating analyses from {len(image_analyses)} images")
            consolidated_analysis = await consolidate_image_analyses(image_analyses)

            if consolidated_analysis and not "error" in consolidated_analysis:
                # Create a consolidated description from the analysis
                consolidated_description = consolidated_analysis.get("consolidated_description", "")
                if not consolidated_description and consolidated_analysis.get("unified_summary"):
                    consolidated_description = consolidated_analysis.get("unified_summary")

                # Check if we have a product_name and estimated_total_quantity
                if "product_name" in consolidated_analysis and "estimated_total_quantity" in consolidated_analysis:
                    consolidated_description = f"Analysis of multiple images confirms the product is '{consolidated_analysis['product_name']}'. "
                    consolidated_description += f"The total quantity is estimated to be {consolidated_analysis['estimated_total_quantity']} boxes. "

                    # Add overlap logic explanation if available
                    if "overlap_logic_explanation" in consolidated_analysis:
                        consolidated_description += f"{consolidated_analysis['overlap_logic_explanation']} "

                    # Add confidence level if available
                    if "confidence" in consolidated_analysis:
                        consolidated_description += f"Confidence level: {consolidated_analysis['confidence']}."
                elif "unified_summary" in consolidated_analysis:
                    consolidated_description = consolidated_analysis["unified_summary"]

                if consolidated_description:
                    # Use the first image path as the source for the consolidated document
                    img_path = images_path[0]
                    file_type = Path(img_path).suffix.lower().replace('.', '')

                    # Create a special filename to indicate this is a consolidated analysis
                    file_name = "consolidated_analysis.json"
                    original_file_name = "consolidated_analysis"

                    # Extract product name and quantity from the consolidated analysis
                    product_name = consolidated_analysis.get("product_name", "")
                    quantity = str(consolidated_analysis.get("estimated_total_quantity", ""))
                    confidence = consolidated_analysis.get("confidence", "")

                    # Extract key details from the view analysis if available
                    key_details = consolidated_analysis.get("key_details", [])
                    if not key_details and "view_analysis" in consolidated_analysis:
                        key_details = []
                        for view in consolidated_analysis["view_analysis"]:
                            if "unique_features" in view:
                                key_details.append(f"View {view.get('view_id', '')}: {view.get('unique_features', '')}")

                    consolidated_doc = Document(
                        page_content=consolidated_description,
                        metadata={
                            "source": "multiple_images",
                            "file_name": file_name,
                            "original_file_name": original_file_name,
                            "file_type": "json",
                            "is_image": True,
                            "is_consolidated": True,
                            "image_count": len(image_analyses),
                            "product_name": product_name,
                            "quantity_estimation": quantity,
                            "confidence_level": confidence,
                            "key_details": key_details
                        }
                    )

                    # Add the consolidated document to the list of documents
                    consolidated_doc_info = DocumentInfo(
                        file_path="consolidated_analysis",
                        file_type="json",
                        file_name=file_name,
                        original_file_name=original_file_name,
                        content=[consolidated_doc]
                    )

                    documents.append(consolidated_doc_info)
                    consolidated_result = consolidated_doc_info
                    logger.info("Added consolidated document for multiple images")

        # Update the state with the new documents
        result = {
            "documents": documents,
            "image_documents": image_documents
        }

        if consolidated_result:
            result["consolidated_image_document"] = consolidated_result

        return result

    except Exception as e:
        logger.error(f"Error in process_images: {str(e)}")
        return {"error": str(e), "status": "error"}


async def extract_entities(documents: List[Union[Document, Dict[str, Any]]], entity_types: Optional[List[str]] = None) -> Dict[str, Any]:
    llm = initialize_chat_model(temperature=0.0)

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
                if isinstance(doc, dict) and "page_content" in doc:
                    doc_content = doc["page_content"]
                    doc_metadata = doc.get("metadata", {})
                    doc = Document(page_content=doc_content, metadata=doc_metadata)
                elif isinstance(doc, dict) and "content" in doc:
                    # Handle nested document structure
                    doc_content = doc["content"]
                    doc_metadata = doc.get("metadata", {})
                    doc = Document(page_content=doc_content, metadata=doc_metadata)

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

            for extraction in all_extractions:
                # Standardize extraction handling regardless of type
                entities = get_attribute(extraction, "entities", [])
                dates = get_attribute(extraction, "dates", [])
                key_topics = get_attribute(extraction, "key_topics", [])
                sentiment = get_attribute(extraction, "sentiment", {})
                relationships = get_attribute(extraction, "relationships", [])

                for entity in entities:
                    if "name" in entity and "type" in entity:
                        key = f"{entity['name']}_{entity['type']}"
                        entity_map[key] = entity

                for date in dates:
                    if "date" in date:
                        key = date["date"]
                        date_map[key] = date

                # Process topics
                topics.update(key_topics)

                if sentiment and len(sentiment) > len(result["sentiment"]):
                    result["sentiment"] = sentiment

                for rel in relationships:
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
