import datetime

import asyncio

from typing import Dict, Any, List, Union
from loguru import logger

from domains.workflows.generator import run_orchestrator_graph


async def document_summarize_orchestrator(
    file_paths: Union[str, List[str]],
    image_paths: Union[str, List[str]] = None,
    extract_entities: bool = True,
    token_max: int = 1000,
    instructions: str = None,
    use_agent: bool = True
) -> Dict[str, Any]:
    try:
        result = await run_orchestrator_graph(
            file_paths=file_paths,
            extract_entities=extract_entities,
            token_max=token_max,
            instructions=instructions,
            use_agent=use_agent
        )

        if isinstance(result, dict):
            if "metadata" in result:
                result["metadata"]["processed_at"] = datetime.datetime.now().isoformat()
            elif "results" in result and isinstance(result["results"], list):
                for item in result["results"]:
                    if isinstance(item, dict) and "metadata" in item:
                        item["metadata"]["processed_at"] = datetime.datetime.now().isoformat()

        return result

    except Exception as e:
        logger.error(f"Error in document_summarize_orchestrator: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    file_paths = ['/Users/mohitverma/Documents/multi-conversational-tool/temp/DIKSHA_R (1).pdf']
    result = asyncio.run(document_summarize_orchestrator(file_paths))
    print('result')
    print(result)
