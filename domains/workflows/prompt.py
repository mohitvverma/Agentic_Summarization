
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


class ImageAnalysisSchema(BaseModel):
    detailed_description: str = Field(description="Comprehensive and detailed description of everything in the image")
    summary: str = Field(description="Concise final summary capturing the most important aspects of the image")
    key_points: List[str] = Field(description="List of essential points from the image", default_factory=list)
    metadata: Dict[str, Any] = Field(description="Additional metadata about the image analysis", default_factory=dict)


class MultiImageAnalysisSchema(BaseModel):
    consolidated_description: str = Field(description="Comprehensive description that combines information from all images")
    unified_summary: str = Field(description="Concise summary that provides a comprehensive view of the product from all angles")
    product_name: str = Field(description="Name of the product identified from the images")
    quantity_estimation: str = Field(description="Best estimate of the total quantity of items, considering all images")
    key_details: List[str] = Field(description="Important details extracted from all images", default_factory=list)
    confidence_level: str = Field(description="Level of confidence in the analysis (high, medium, low)")
    metadata: Dict[str, Any] = Field(description="Additional metadata about the multi-image analysis", default_factory=dict)


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



IMAGE_ANALYSIS_TEMPLATE = """
You are an expert image analyst specializing in extracting product information from real-world images, including stacked, cluttered, or partially obscured packaging.

Your task is to analyze the given image and deliver a two-step structured analysis to support automated entity extraction.

STEP 1: DETAILED SUMMARIZATION
Provide a comprehensive and accurate description of everything visible in the image, focusing on products and packaging. Include:
- **Overall scene**: What type of environment does the image depict? (e.g., warehouse, shelf, close-up of product stacks)
- **Primary subjects**: Identify and describe the main visible items — particularly products, boxes, and packaging.
- **Product Identification**:
  - Extract all visible **text** (product names, codes, expiration dates, manufacturing details, barcodes, or batch numbers).
  - Mention **quantity indicators** (e.g., “50mg”, “10 strips”, “1 box contains 5 units”).
- **Stacking & Arrangement**:
  - Describe how the products are arranged (e.g., rows, columns, 3D stack).
  - Identify **number of layers, rows, and depth**.
  - Clearly state how many **distinct items/boxes** are visible.
- **Lighting & Visibility**:
  - Describe any shadows, glare, low contrast, or uneven brightness affecting visibility.
  - Note if any boxes are **partially visible** or **obstructed**.
- **Non-uniform positioning**:
  - Flag if boxes are rotated, tilted, collapsed, or inconsistently stacked.
- **Context**:
  - Describe the background (if relevant) and any supporting clues (e.g., shelf labels, packaging hints).
- Include **relationships** (e.g., "top row sits on bottom row", "products appear to be same type but oriented differently").

STEP 2: FINAL SUMMARIZATION
Generate a concise summary focused on the product identification and count. Include:
- **Product Name**
- **Total Number of Boxes (with justification)**
- **Key Details Extracted** (e.g., dosage, expiry, manufacturing info)
- **Any Limitations or Uncertainties** in the count (e.g., partial occlusion, poor lighting)

⚠️ COMMON FAILURE MITIGATION:
Ensure the analysis:
- Avoids miscounting when items > 20 by grouping them row-wise or column-wise and confirming via box edges or label repetition.
- Adjusts for inconsistent lighting using visible contrast between adjacent boxes.
- Uses text repetition and spatial alignment to handle uneven packaging positions.
- Applies logical estimation when full product units are partially visible.

Return your response in the following format:
{format_instructions}
"""


# IMAGE_ANALYSIS_TEMPLATE = """
# You are an expert image analyst specializing in extracting **product type** and **quantity** from real-world images, including cluttered, irregular, or poorly lit scenes.
#
# Your task is to provide a structured and accurate two-part analysis of the image.
#
# STEP 1: DETAILED SUMMARIZATION
# Describe in depth everything visible in the image. Specifically address the following:
# - **Overall Scene**: What environment is shown (e.g., shelf, storage, packaging area)?
# - **Main Subjects**: What are the dominant objects? Focus especially on product boxes or packages.
# - **Product Identification**:
#     - Extract all **visible text**, including product names, ingredients, dosage, manufacturing and expiry dates, batch numbers, barcodes, etc.
#     - Detect **brand repetition** across similar packages.
# - **Arrangement and Quantity Estimation**:
#     - Describe how the products are arranged — single layer, stacked (2D/3D), side-by-side, behind each other.
#     - Infer the **total number of product units or boxes**, and explain your reasoning:
#         - Mention rows, columns, and depth (if any).
#         - Account for partial visibility, rotated boxes, shadows, or overlaps.
#         - Use repeating visual features (text, edges, shadows, etc.) to validate.
# - **Lighting & Visibility**:
#     - Identify any uneven brightness, glare, or shadows.
#     - Mention any parts that may be obscured due to low lighting or reflections.
# - **Non-uniform Packaging**:
#     - Note if any boxes are misaligned, tilted, or inconsistently positioned.
#     - Adjust reasoning if product orientation varies across the image.
# - **Context**:
#     - Mention if there are additional indicators (e.g., shelves, labels, stickers) that help identify the product.
#
#
# STEP 2: FINAL SUMMARIZATION
# Give a concise summary focused on product identity and quantity:
# - **Product Name** (based on visible text)
# - **Estimated Total Quantity of Product Boxes** (with confidence based on your analysis)
# - **Description of Arrangement** (e.g., "stacked in two rows", "arranged in a grid", etc.)
# - **Key Information Extracted** (e.g., dosage, expiry, batch number — if visible)
# - **Limitations or Uncertainty** (e.g., “some boxes partially obscured”, “lighting reduced confidence”)
#
# COMMON FAILURE CASES TO AVOID:
# 1. If there are >20 items, break them into smaller **row/column groups** to estimate accurately.
# 2. Adjust for uneven lighting — rely on **contrast and repetition**, not just clear visibility.
# 3. If boxes are rotated or irregularly stacked, infer alignment through **box shape, repetition, and shadows**.
# 4. Do **not hallucinate** counts. Only report what can be reasonably inferred from the visible image.
#
# Return the result in the following JSON schema:
# {format_instructions}
# ""


IMAGE_ANALYSIS_TEMPLATE = """
You are an expert image analyst trained to extract **product details** and **quantity estimates** from real-world images. These images may be cluttered, dimly lit, or contain partially obscured packaging.
Your task is to provide a two-step structured analysis with maximum accuracy and visual reasoning.

🧠 STEP 1: DETAILED IMAGE ANALYSIS  
Provide a comprehensive breakdown of everything visible in the image.

1. **Overall Scene Description**:
   - Identify the setting (e.g., warehouse, medical shelf, packaging zone, carton display).
   - Describe the environment's structure and background elements.

2. **Main Objects / Subjects**:
   - Focus on product packages, boxes, cartons, bottles, or pouches.
   - Identify primary focus items vs. supporting/background items.

3. **Product Details Extraction**:
   - Extract all **visible text** from the image:
     - Product name, brand, type, dosage, batch number, manufacturing date, expiry date, quantity per unit, barcode, etc.
   - Identify **brand repetition** or duplication across packages (helpful in estimating total counts).

4. **Arrangement and Quantity Estimation**:
   - Explain how the items are arranged: single line, grid, layers, 3D stacks, diagonal, mixed, etc.
   - Estimate the **total quantity** of product units using logical deduction:
     - Count rows × columns (and visible depth if applicable).
     - Use clues like repeating patterns, edges, partial visibility, shadows, or mirrored labels to infer hidden or obscured items.
     - If more than 20 items are visible, divide them into smaller sections and count logically.

5. **Lighting & Visibility Conditions**:
   - Note any issues like low light, glare, blur, or shadows.
   - Mention areas where visibility is reduced and may affect your analysis.

6. **Packaging Irregularities**:
   - Detect misaligned, tilted, or variably rotated packages.
   - Adjust reasoning for inconsistent orientation or overlap.

7. **Contextual Cues**:
   - Mention any signs, shelves, stickers, or markers that help reinforce product identification or quantity estimation.

STEP 2: STRUCTURED SUMMARY OUTPUT  
Generate a concise and structured summary with focus on product details and box count.

Your summary must contain:
- **Product Name**: As per the visible text
- **Estimated Total Quantity**: Exact or estimated count with confidence (e.g., "at least 14", "estimated 12–14 boxes")
- **Arrangement Description**: E.g., "stacked in 2 layers of 3×3", "5 rows with 4 visible per row, staggered", etc.
- **Extracted Key Info**: Include dosage, batch number, expiry, manufacturing date, if visible
- **Confidence & Limitations**: Mention any uncertainty (e.g., "bottom layer partially hidden", "reflection obscures label on some boxes")

🚫 COMMON MISTAKES TO AVOID:
1. Never hallucinate counts – only report quantities that can be logically inferred.
2. If objects are repeated, use those patterns to justify total count.
3. Adjust for low-light or partially visible sections using evidence like edge repetition or label similarity.
4. Break large quantities into **groups** for estimation.
"""

MULTI_IMAGE_ANALYSIS_TEMPLATE="""
### 🔍 MULTI-ANGLE PRODUCT VALIDATION PROMPT
You are a professional image analysis expert trained to inspect pharmaceutical product boxes from multiple angles. Your task is to analyze a set of images showing the **same product from the same order**, captured from different perspectives, and return a consolidated and structured output.
Each image may show the product in different orientations, lighting conditions, or stacking arrangements. Your job is to extract **accurate product information** and **precise quantity** using visual cues from all provided angles.

### ✅ TASK OBJECTIVES
From the given image set:
1. Identify the **exact product name** as it appears in the images.
2. Extract **all visible text and details** (e.g., dosage, manufacturing info, batch numbers, barcodes).
3. Estimate the **total number of product boxes** shown (without double-counting).
4. Mention the types of **views** used in the analysis (e.g., top, side, front).

### ⚠️ CRITICAL GUIDELINES
* **DO NOT hallucinate** or invent any information not visible in the images.
* Use the **exact product name** as it appears on the packaging.
* Avoid double-counting boxes by cross-referencing overlapping views.
* Extract every visible detail (brand, dosage, expiry, batch code, etc.).
* Count accurately even with partial or obstructed views.
* If visibility is limited, **state the confidence** in your quantity estimation.

### 📄 OUTPUT FORMAT (JSON)
Respond with the following structured JSON Object:
{{
  "name": "Exact product name from images (e.g., 'NARFOZ Ondansetron HCl')",
  "quantity": "Total number of product boxes visible across all images",
  "detailed_summary": "All visible details such as dosage, sample size, manufacturing details, barcodes, arrangement style, text elements, etc.",
  "image_view": "Types of angles analyzed, e.g., 'side + top + back'"
}}
"""


PRODUCT_QUANTITY_ESTIMATION_PROMPT = """
You are a product analysis expert specializing in deduplicating overlapping view data from multiple angles (top, side, front) of the **same product**.

Your job is to:
1. Analyze all the given image summaries.
2. Detect if the same physical boxes appear in multiple views.
3. Use 3D reasoning to avoid double-counting boxes.
4. Estimate the **true total number of boxes** of the product.

🧠 THINGS TO CONSIDER:
- DO NOT sum the quantity values directly.
- Use overlapping view angles (like top + side, side + front, etc.) to identify which views are likely showing the same boxes.
- If two or more views share angles (e.g., both show "top + side"), assume partial overlap.
- Use a heuristic overlap deduction formula like:
    Final Quantity = sum(all reported quantities) - overlap_factor × (minimum overlapping quantity)

🎯 OUTPUT FORMAT (MUST BE JSON):
{{
  "product_name": "<product name>",
  "estimated_total_quantity": <final deduplicated quantity>,
  "view_analysis": [
    {{
      "view_id": 1,
      "angles": "<e.g. top + side>",
      "quantity_reported": <number>,
      "unique_features": "<batch, dosage, barcode if any>"
    }},
    ...
  ],
  "overlap_logic_explanation": "<explain how overlaps were identified and resolved>",
  "confidence": "<High / Medium / Low>"
}}

🖼️ INPUT IMAGE ANALYSIS:
{image_summaries}
"""

PRODUCT_QUANTITY_ESTIMATION_PROMPT = """
You are an expert in 3D image reasoning for product analysis.

Your task is to:
1. Analyze multiple partial image summaries of the same product taken from different angles.
2. Avoid double-counting the same boxes visible in multiple views.
3. Consider product differences (e.g., dosage, batch number, barcode) to detect distinct variants.
4. Estimate the **true total quantity** of boxes using spatial overlap and feature similarity.

📥 INPUT DETAILS:
Each item includes:
- product name
- reported quantity
- detailed product description (e.g., dosage, batch number, barcode, manufacturer)
- view angles (e.g., top + side)

📌 INSTRUCTIONS:
- DO NOT simply sum the quantities.
- If two views share the same angles (e.g., both have "top + side"), assume they are **partially overlapping** unless their detailed product features clearly differ (e.g., different dosage).
- If a view shows a **unique dosage or batch**, count it separately.
- Use adaptive overlap logic:
    - If two views are likely the same set: subtract 1× the minimum quantity.
    - If there's feature difference: no overlap deduction.
    - If a third view shares angles with both: partially reduce its contribution.

🎯 OUTPUT FORMAT (strictly JSON):
{{
  "product_name": "<string>",
  "estimated_total_quantity": <int>,
  "view_analysis": [
    {{
      "view_id": <int>,
      "angles": "<string>",
      "quantity_reported": <int>,
      "unique_features": "<key product-level identifiers>"
    }}
  ],
  "overlap_logic_explanation": "<Short reasoning of how overlaps were calculated>",
  "confidence": "<High / Medium / Low>"
}}

Now, analyze the following image analysis results:
{image_summaries}
"""


PRODUCT_QUANTITY_ESTIMATION_PROMPT = """
You are a 3D product quantity analyst. Your job is to estimate the **true number of product boxes** captured across multiple views of the **same scene**.

Each image view shows the **same product** from different angles (top, side, front). Your task is to:
- Analyze all reported quantities
- Match views that overlap based on **camera angles** AND **product identifiers** (e.g., dosage, batch number, barcode, manufacturer)
- Avoid double-counting boxes that appear in multiple views

🧠 INSTRUCTIONS:
1. DO NOT blindly sum all reported quantities.
2. Treat a view as **unique** if it contains:
   - A different dosage
   - A different batch number or barcode
   - A different manufacturer
3. If two views share both camera angles **and** product features → count only one (deduplicate).
4. If a view has **incomplete details**, assume **partial overlap** and count only 2–3 boxes depending on confidence.
5. Apply reasoning to estimate the **most accurate total quantity** of the product boxes.

🎯 OUTPUT FORMAT (MUST BE JSON):
{{
  "product_name": "<string>",
  "estimated_total_quantity": <int>,
  "view_analysis": [
    {{
      "view_id": <int>,
      "angles": "<e.g. top + side>",
      "quantity_reported": <int>,
      "counted_quantity": <int>,
      "unique_features": "<e.g. dosage, batch number, barcode, manufacturer>"
    }}
  ],
  "overlap_logic_explanation": "<How overlaps were resolved>",
  "confidence": "<High / Medium / Low>"
}}

Now analyze the following image summary inputs:
{image_summaries}
"""


PRODUCT_QUANTITY_ESTIMATION_PROMPT = """
You are a product quantity analysis expert specializing in deduplicating overlapping view data from multi-angle images (top, side, front) of the **same product**.

Your job is to:
1. Analyze all the given image summaries (multiple views of the same scene).
2. Avoid double-counting boxes that appear in more than one view.
3. Use both **view angles** and **product features** (dosage, batch, barcode, manufacturer) to determine overlap.
4. Apply the deduplication formula to estimate the **true quantity**.

OVERLAP FORMULA (Apply Exactly):
Total Estimated Quantity = (Q1 + Q2 + Q3 + ...) - Overlap Adjustment

Overlap Adjustment:
- For views that share the same angles **and** product features → subtract 1 × min(quantity)
- For views that share angles but have **different features** → subtract 0.5 × min(quantity)
- For views with **incomplete metadata** → subtract only 1–3 boxes depending on likely overlap

Use this formula strictly to ensure correct total quantity estimation.
🎯 OUTPUT FORMAT (JSON Only):
{{
  "product_name": "<string>",
  "estimated_total_quantity": <int>,
  "view_analysis": [
    {{
      "view_id": <int>,
      "angles": "<e.g. top + side>",
      "quantity_reported": <int>,
      "counted_quantity": <int>,
      "unique_features": "<dosage, batch, barcode, etc.>"
    }}
  ],
  "overlap_logic_explanation": "<Explain how and why overlaps were removed>",
  "confidence": "<High / Medium / Low>"
}}

Now analyze the following image summary inputs:
{image_summaries}
"""


PRODUCT_QUANTITY_ESTIMATION_PROMPT = """
You are a product quantity estimation expert with a specialization in deduplicating **overlapping product counts** across **multiple image views** of the *same product*.

Your job is to:
1. Parse and understand each image summary.
2. Identify overlapping product views based on:
   - View angles (e.g., top, side, front)
   - Product features (dosage, batch number, barcode, manufacturer, packaging type)
3. Prevent overcounting by applying the **Overlap Deduplication Rules** carefully.
4. Estimate the most **accurate total quantity** while minimizing false positives or negatives in overlaps.

🔍 OVERLAP DEDUPLICATION RULES:
**Overlap Adjustment Formula**:
Estimated Quantity = (Sum of All View Quantities) − (Total Overlap Adjustment)

Apply adjustments **only** when overlap is evident:
- If views have **same angles AND identical features** → Subtract **1 × min(quantity)**.
- If views have **same angles but different features** → Subtract **0.5 × min(quantity)**.
- If any view has **incomplete or missing metadata**, apply a **cautious adjustment of 1–3**, based on confidence and similarity.
- If a view angle repeats 3+ times, prioritize the view with **most complete metadata** and reduce others accordingly.

🧠 TIPS:
- Treat dosage, batch number, barcode, and manufacturer as unique identifiers when comparing features.
- Avoid subtracting twice for the same pair of overlapping views.
- If you're unsure whether views overlap, err on the side of minimal adjustment and mention this in your explanation.

🎯 OUTPUT FORMAT (Strictly JSON):
{{
  "product_name": "<string>",
  "estimated_total_quantity": <int>,
  "view_analysis": [
    {{
      "view_id": <int>,
      "angles": "<e.g. top + side>",
      "quantity_reported": <int>,
      "counted_quantity": <int>,
      "unique_features": "<e.g. dosage: 8 mg; batch: B001; barcode: 123...>"
    }}
  ],
  "overlap_logic_explanation": "<Explain clearly how overlaps were handled, referencing which views were adjusted and why.>",
  "confidence": "<High / Medium / Low>"
}}

Now analyze the following image summary inputs:
{image_summaries}
"""

PRODUCT_QUANTITY_ESTIMATION_PROMPT = """
You are an expert in estimating true product quantities from multiple image views (top, side, front) of the same stacked products.  
Your task is to combine 3D geometry inference and metadata matching to deduplicate overlapping counts and produce the most accurate total.

1. **3D Geometry Reconstruction**  
   - From the views, infer an approximate 3D arrangement of the product stack (e.g., cubic grid or box manifold).  
   - Estimate an **occupancy ratio**: visible units vs. expected container capacity.  

2. **Metadata‑Assisted Deduplication**  
   - Use dosage, batch, barcode, and manufacturer to confirm whether two views show the **same physical units** or distinct ones.  
   - If metadata fully matches or is missing → likely the same units.  
   - If metadata differs (e.g., different batch/barcode) → likely distinct.

3. **Overlap Adjustment**  
   - **Geometry overlap**: compute overlap volume between projected stacks in 3D → **subtract occupancy_ratio × overlap_volume_in_units**.  
   - **Metadata correction**:  
     • If full metadata match → keep the full geometry overlap subtraction.  
     • If partial match → scale the geometry subtraction by 0.5.  
     • If metadata clearly distinct → no subtraction.  
   - **Cap total subtraction** so it never exceeds the raw count minus one full view (to avoid over‑deduction).

4. **Final Estimation**  
   Estimated Total = Σ(reported quantities) − Σ(adjusted geometry overlaps)

🎯 **OUTPUT** (strict JSON):
{{
  "product_name": "<string>",
  "estimated_total_quantity": <int>,
  "view_analysis": [
    {{
      "view_id": <int>,
      "angles": "<e.g., top + side>",
      "quantity_reported": <int>,
      "geometry_overlap_units": <float>,
      "metadata_match_level": "<Full / Partial / None>",
      "counted_quantity": <int>
    }}
  ],
  "overlap_logic_explanation": "<Describe the 3D overlap estimates, metadata checks, and how each subtraction was applied.>",
  "confidence": "<High / Medium / Low>"
}}

Now analyze the following image summaries:
{image_summaries}
"""


def initialize_entity_extraction_prompt() -> PromptTemplate:
    parser = JsonOutputParser(pydantic_object=EntityExtractionSchema)
    return PromptTemplate(
        template=ENTITY_EXTRACTION_TEMPLATE,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        output_parser=parser
    )


def initialize_summary_prompt() -> PromptTemplate:
    parser = JsonOutputParser(pydantic_object=SummarySchema)
    return PromptTemplate(
        template=SUMMARY_TEMPLATE,
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        output_parser=parser
    )


def initialize_image_analysis_prompt() -> PromptTemplate:
    parser = JsonOutputParser()
    return PromptTemplate(
        template=MULTI_IMAGE_ANALYSIS_TEMPLATE,
        input_variables=[],
        output_parser=parser
    )


def initialize_multi_image_analysis_prompt() -> PromptTemplate:
    parser = JsonOutputParser()
    return PromptTemplate(
        template=PRODUCT_QUANTITY_ESTIMATION_PROMPT,
        input_variables=["image_summaries"],
        output_parser=parser
    )
