import pprint
from typing import Dict, Any, Optional, TypeVar
from langchain_core.messages import HumanMessage

T = TypeVar('T')


def create_result_dict(file_path: str, file_type: str, status: str = "success",
                      summary: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                      entities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result = {
        "file_path": file_path,
        "file_type": file_type,
        "status": status
    }

    if summary:
        result["summary"] = summary

    if metadata:
        result["metadata"] = metadata
    else:
        result["metadata"] = {}

    if entities:
        result["entities"] = entities

    return result


def extract_summary_text(summary: Any) -> str:
    if isinstance(summary, str):
        return summary
    elif hasattr(summary, "summary"):
        return summary.summary
    elif isinstance(summary, dict) and "summary" in summary:
        return summary["summary"]
    else:
        return str(summary)


def get_attribute(obj: Any, attr_name: str, default: Optional[T] = None) -> T:
    if hasattr(obj, attr_name):
        return getattr(obj, attr_name)
    elif isinstance(obj, dict) and attr_name in obj:
        return obj[attr_name]
    return default


def summary_generation_prompt(
        image_url,
        template
):
    return [
        HumanMessage(
            content=[
                {"type": "text", "text": template},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
            ]
        )
    ]
