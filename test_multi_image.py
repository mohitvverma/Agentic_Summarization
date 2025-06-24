import asyncio
from domains.workflows.routes import document_summarize_orchestrator
import json
from pathlib import Path

async def test_multi_image_analysis():
    # Define the paths to the images
    image_paths = [
        str(Path("images/1A.JPEG").absolute()),
        str(Path("images/1B.JPEG").absolute()),
        str(Path("images/1C.JPEG").absolute())
    ]
    
    print(f"Processing images: {image_paths}")
    
    # Run the orchestrator with the images
    result = await document_summarize_orchestrator(images_path=image_paths)
    
    # Print the result in a readable format
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    
    # Check if we have a consolidated analysis
    if "is_consolidated" in result.get("metadata", {}):
        print("\n=== CONSOLIDATED ANALYSIS ===")
        print(f"Product Name: {result['metadata'].get('product_name', 'N/A')}")
        print(f"Quantity Estimation: {result['metadata'].get('quantity_estimation', 'N/A')}")
        print(f"Confidence Level: {result['metadata'].get('confidence_level', 'N/A')}")
        print("\nSummary:")
        print(result.get("summary", "No summary available"))
        
        if "key_details" in result:
            print("\nKey Details:")
            for detail in result["key_details"]:
                print(f"- {detail}")
    
if __name__ == "__main__":
    asyncio.run(test_multi_image_analysis())