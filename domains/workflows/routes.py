import datetime

import asyncio

from typing import Dict, Any, List, Union
from loguru import logger

from dotenv import load_dotenv
load_dotenv()
from domains.workflows.generator import run_orchestrator_graph


async def document_summarize_orchestrator(
    file_paths: Union[str, List[str]]=None,
    extract_entities: bool = True,
    token_max: int = 1000,
    images_path: Union[str, List[str]] = None
) -> Dict[str, Any]:
    try:
        result = await run_orchestrator_graph(
            file_paths=file_paths,
            images_path=images_path,
            extract_entities=extract_entities,
            token_max=token_max
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
    file_paths = ['/Users/mohitverma/Downloads/Unconfirmed 538281.crdownload/1A.jpeg',
                  '/Users/mohitverma/Downloads/Unconfirmed 538281.crdownload/1B.JPEG',
                  '/Users/mohitverma/Downloads/Unconfirmed 538281.crdownload/1C.JPEG']
    result = asyncio.run(document_summarize_orchestrator(images_path=file_paths))
    print('result')
    print(result)
