import operator
from typing import TypedDict, Annotated


# class OverallSummaryState(TypedDict):
#     contents: Annotated[list, operator.add]
#     summaries: Annotated[list, operator.add]
#     collapsed_summaries: Annotated[list, operator.add]
#     final_summary: str


class EntityExtractionState(TypedDict):
    file_path: Annotated[str, operator.add]
    file_type: Annotated[str, operator.add]
    file_name: Annotated[str, operator.add]
    original_file_name: Annotated[str, operator.add]
    contents: Annotated[list, operator.add]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: Annotated[list, operator.add]
    final_summary: Annotated[str, operator.add]
    entities: dict
