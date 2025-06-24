import asyncio
import pprint
from pathlib import Path
from domains.workflows.routes import document_summarize_orchestrator

async def test_document_processing():
    # Replace this with the path to a sample document
    # Example: file_path = "/Users/username/Documents/sample.pdf"
    # The path found in routes.py is: '/Users/mohitverma/Documents/multi-conversational-tool/temp/DIKSHA_R (1).pdf'
    # But you should replace it with a path to a document that exists on your system
    file_path = "/Users/mohitverma/Documents/multi-conversational-tool/temp/DIKSHA_R (1).pdf"

    # Check if the file exists
    if not Path(file_path).exists():
        print(f"Error: File {file_path} does not exist.")
        print("Please update the file_path variable with a valid path to a document on your system.")
        return

    print(f"Processing document: {file_path}")

    # Process the document with entity extraction enabled
    result = await document_summarize_orchestrator(
        file_paths=file_path,
        extract_entities=True,
        token_max=1000
    )

    # Print the result
    print("Result status:", result.get("status"))

    # Check if summary is included
    if result.get("summary"):
        print("\nSummary found:")
        print(result["summary"])
    else:
        print("\nNo summary found in the result.")

    # Check if entities are included
    if result.get("entities") and isinstance(result["entities"], dict):
        print("\nEntities found:")
        print("Entities status:", result["entities"].get("status"))

        # Check for specific entity types
        if result["entities"].get("entities"):
            print(f"\nFound {len(result['entities']['entities'])} entities")
            for entity in result["entities"]["entities"][:3]:  # Print first 3 entities
                print(f"  - {entity.get('name', 'Unknown')} ({entity.get('type', 'Unknown')})")

        if result["entities"].get("dates"):
            print(f"\nFound {len(result['entities']['dates'])} dates")
            for date in result["entities"]["dates"][:3]:  # Print first 3 dates
                print(f"  - {date.get('date', 'Unknown date')}: {date.get('context', '')}")

        if result["entities"].get("key_topics"):
            print(f"\nFound {len(result['entities']['key_topics'])} key topics")
            print("  - " + ", ".join(result["entities"]["key_topics"][:5]))  # Print first 5 topics
    else:
        print("\nNo entities found in the result.")

    # Print the full result structure
    print("\nFull result structure:")
    pprint.pprint(result)

if __name__ == "__main__":
    asyncio.run(test_document_processing())
