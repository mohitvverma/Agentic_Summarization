import os
from enum import Enum
from typing import ClassVar

from pydantic_settings import BaseSettings


class LLMServiceEnum(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    AZURE_OPENAI = "azure_openai"
    GROQ = "groq"
    AWS = 'aws'


class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_CHAT_BASE_URL: str = os.environ.get(
        "OPENAI_CHAT_BASE_URL", "https://api.openai.com/v1/chat/completions"
    )
    GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    GOOGLE_CSE_ID: str = os.environ.get("GOOGLE_CSE_ID", "")

    THRESHOLD_MESSAGE_TO_SUMMARIZE: int = int(
        os.environ.get("THRESHOLD_MESSAGE_TO_SUMMARIZE", 10)
    )
    # for we search
    TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")

    # Session key
    SECRET_SESSION_RANDOM_KEY: str = os.environ.get(
        "SECRET_SESSION_RANDOM_KEY", "secret"
    )

    # aws settings
    AWS_ACCESS_KEY_ID: str = os.environ.get("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION_NAME: str = os.environ.get("AWS_REGION_NAME", "")

    # api path
    PERMISSION_API_PATH: str = os.environ.get("PERMISSION_API_PATH", "")

    # general settings
    TOTAL_DOCS_TO_RETRIEVE: int = os.environ.get("TOTAL_DOCS_TO_RETRIEVE", 15)
    CHUNK_SIZE: int = os.environ.get("CHUNK_SIZE", 1000)
    CHUNK_OVERLAP: int = os.environ.get("CHUNK_OVERLAP", 100)

    RESOLUTION: int = os.environ.get("RESOLUTION", 300)
    PDF_CHUNK_SIZE: int = os.environ.get("PDF_CHUNK_SIZE", 5)

    LLM_SERVICE_TYPE: str = os.environ.get(
        "LLM_SERVICE_TYPE", LLMServiceEnum.OPENAI.value
    )

    SUPPORTED_FILE_TYPES: list[str] = os.environ.get(
        "SUPPORTED_FILE_TYPES",
        ['jpg','png','jpeg','gif','webp', 'jpg', 'pdf']
    )

    LLMS: ClassVar[dict] = {
        "CHAT_MODEL_NAME": os.environ.get("CHAT_MODEL_NAME", "gpt-4o-mini"),
        "EMBEDDING_MODEL_NAME": os.environ.get(
            "EMBEDDING_MODEL_NAME", "text-embedding-3-small"
        ),
        "LLM_MODEL_NAME": os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini"),
        "CLASSIFICATION_MODEL_NAME": os.environ.get("CLASSIFICATION_MODEL_NAME", "gpt-4o-mini"),
        "TABLE_SUMMARIZER_MODEL": os.environ.get("TABLE_SUMMARIZER_MODEL", "gpt-4o-mini"),
        "VISION_MODEL": os.environ.get("VISION_MODEL", "gpt-4o-mini"),
        "OPTIMIZED_QUESTION_MODEL": os.environ.get("OPTIMIZED_QUESTION_MODEL", "gpt-4o-mini"),
        "LANGUAGE_DETECTION_MODEL": os.environ.get("LANGUAGE_DETECTION_MODEL", "gpt-4o-mini"),
        "CHAT_STREAMING_MODEL": os.environ.get("CHAT_STREAMING_MODEL", "gpt-4o"),
        "RAG_LLM_MODEL": os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini"),
        "REASONING_MODEL": os.environ.get("REASONING_MODEL", "gpt-o3"),
    }

    GEMINI_LLMS: ClassVar[dict] = {
        "CHAT_MODEL_NAME": os.environ.get("CHAT_MODEL_NAME", "gpt-4o-mini"),
        "EMBEDDING_MODEL_NAME": os.environ.get(
            "EMBEDDING_MODEL_NAME", "text-embedding-3-small"
        ),
        "LLM_MODEL_NAME": os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini"),
        "CLASSIFICATION_MODEL_NAME": os.environ.get("CLASSIFICATION_MODEL_NAME", "gpt-4o-mini"),
        "TABLE_SUMMARIZER_MODEL": os.environ.get("TABLE_SUMMARIZER_MODEL", "gpt-4o-mini"),
        "VISION_MODEL": os.environ.get("VISION_MODEL", "gpt-4o-mini"),
        "OPTIMIZED_QUESTION_MODEL": os.environ.get("OPTIMIZED_QUESTION_MODEL", "gpt-4o-mini"),
        "LANGUAGE_DETECTION_MODEL": os.environ.get("LANGUAGE_DETECTION_MODEL", "gpt-4o-mini"),
        "CHAT_STREAMING_MODEL": os.environ.get("CHAT_STREAMING_MODEL", "gpt-4o"),
        "RAG_LLM_MODEL": os.environ.get("RAG_LLM_MODEL", "gpt-4o-mini"),
    }

    AWS_BEDROCK_LLMS: ClassVar[dict] = {
        "CHAT_MODEL_NAME": os.environ.get(
            "CHAT_MODEL_NAME", "anthropic.claude-3-5-sonnet-20240620-v1:0"
        ),
        "EMBEDDING_MODEL_NAME": os.environ.get(
            "EMBEDDING_MODEL_NAME", "text-embedding-3-small"
        ),
        "LLM_MODEL_NAME": os.environ.get(
            "LLM_MODEL_NAME", "anthropic.claude-3-5-sonnet-20240620-v1:0"
        ),
        "CLASSIFICATION_MODEL_NAME": os.environ.get(
            "CLASSIFICATION_MODEL_NAME", "anthropic.claude-3-5-sonnet-20240620-v1:0"
        ),
        "TABLE_SUMMARIZER_MODEL": os.environ.get(
            "TABLE_SUMMARIZER_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"
        ),
        "VISION_MODEL": os.environ.get(
            "VISION_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"
        ),
        "OPTIMIZED_QUESTION_MODEL": os.environ.get(
            "OPTIMIZED_QUESTION_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"
        ),
        "LANGUAGE_DETECTION_MODEL": os.environ.get(
            "LANGUAGE_DETECTION_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"
        ),
        "CHAT_STREAMING_MODEL": os.environ.get(
            "CHAT_STREAMING_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"
        ),
        "RAG_LLM_MODEL": os.environ.get(
            "RAG_LLM_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0"
        ),
    }

    AZURE_OPENAI_SETTINGS: ClassVar[dict] = {
        "LLM_MODEL_NAME": {
            "ENDPOINT": os.environ.get("AZURE_ENDPOINT_LLM_MODEL_NAME", ""),
            "API_KEY": os.environ.get("AZURE_API_KEY_LLM_MODEL_NAME", ""),
            "DEPLOYMENT": os.environ.get("AZURE_DEPLOYMENT_LLM_MODEL_NAME", ""),
            "API_VERSION": os.environ.get("AZURE_API_VERSION_LLM_MODEL_NAME", ""),
        },
        "MODEL_NAME_GPT_4": {
            "ENDPOINT": os.environ.get("AZURE_ENDPOINT_MODEL_NAME_GPT_4", ""),
            "API_KEY": os.environ.get("AZURE_API_KEY_MODEL_NAME_GPT_4", ""),
            "DEPLOYMENT": os.environ.get("AZURE_DEPLOYMENT_MODEL_NAME_GPT_4", ""),
            "API_VERSION": os.environ.get("AZURE_API_VERSION_MODEL_NAME_GPT_4", ""),
        },
        "MODEL_NAME_VISION_GPT_4": {
            "ENDPOINT": os.environ.get("AZURE_ENDPOINT_MODEL_NAME_VISION_GPT_4", ""),
            "API_KEY": os.environ.get("AZURE_API_KEY_MODEL_NAME_VISION_GPT_4", ""),
            "DEPLOYMENT": os.environ.get(
                "AZURE_DEPLOYMENT_MODEL_NAME_VISION_GPT_4", ""
            ),
            "API_VERSION": os.environ.get(
                "AZURE_API_VERSION_MODEL_NAME_VISION_GPT_4", ""
            ),
        },
        "OPENAI_AUDIO_TRANSCRIPTION_MODEL": {
            "ENDPOINT": os.environ.get(
                "AZURE_ENDPOINT_OPENAI_AUDIO_TRANSCRIPTION_MODEL", ""
            ),
            "API_KEY": os.environ.get(
                "AZURE_API_KEY_OPENAI_AUDIO_TRANSCRIPTION_MODEL", ""
            ),
            "DEPLOYMENT": os.environ.get(
                "AZURE_DEPLOYMENT_OPENAI_AUDIO_TRANSCRIPTION_MODEL", ""
            ),
            "API_VERSION": os.environ.get(
                "AZURE_API_VERSION_OPENAI_AUDIO_TRANSCRIPTION_MODEL", ""
            ),
        },
        "OPENAI_CHAT_MODEL_OMNI": {
            "ENDPOINT": os.environ.get("AZURE_ENDPOINT_OPENAI_CHAT_MODEL_OMNI", ""),
            "API_KEY": os.environ.get("AZURE_API_KEY_OPENAI_CHAT_MODEL_OMNI", ""),
            "DEPLOYMENT": os.environ.get("AZURE_DEPLOYMENT_OPENAI_CHAT_MODEL_OMNI", ""),
            "API_VERSION": os.environ.get(
                "AZURE_API_VERSION_OPENAI_CHAT_MODEL_OMNI", ""
            ),
        },
        "OPTIMIZED_QUESTION_MODEL": {
            "ENDPOINT": os.environ.get("AZURE_ENDPOINT_OPTIMIZED_QUESTION_MODEL", ""),
            "API_KEY": os.environ.get("AZURE_API_KEY_OPTIMIZED_QUESTION_MODEL", ""),
            "DEPLOYMENT": os.environ.get(
                "AZURE_DEPLOYMENT_OPTIMIZED_QUESTION_MODEL", ""
            ),
            "API_VERSION": os.environ.get(
                "AZURE_API_VERSION_OPTIMIZED_QUESTION_MODEL", ""
            ),
        },
        "ASK_AGENT_LLM_STREAMING_MODEL": {
            "ENDPOINT": os.environ.get(
                "AZURE_ENDPOINT_ASK_AGENT_LLM_STREAMING_MODEL", ""
            ),
            "API_KEY": os.environ.get(
                "AZURE_API_KEY_ASK_AGENT_LLM_STREAMING_MODEL", ""
            ),
            "DEPLOYMENT": os.environ.get(
                "AZURE_DEPLOYMENT_ASK_AGENT_LLM_STREAMING_MODEL", ""
            ),
            "API_VERSION": os.environ.get(
                "AZURE_API_VERSION_ASK_AGENT_LLM_STREAMING_MODEL", ""
            ),
        },
        "RAG_LLM_MODEL": {
            "ENDPOINT": os.environ.get("AZURE_ENDPOINT_RAG_LLM_MODEL", ""),
            "API_KEY": os.environ.get("AZURE_API_KEY_RAG_LLM_MODEL", ""),
            "DEPLOYMENT": os.environ.get("AZURE_DEPLOYMENT_RAG_LLM_MODEL", ""),
            "API_VERSION": os.environ.get("AZURE_API_VERSION_RAG_LLM_MODEL", ""),
        },
        "POST_AUDIO_LLM_INFERENCE_MODEL": {
            "ENDPOINT": os.environ.get(
                "AZURE_ENDPOINT_POST_AUDIO_LLM_INFERENCE_MODEL", ""
            ),
            "API_KEY": os.environ.get(
                "AZURE_API_KEY_POST_AUDIO_LLM_INFERENCE_MODEL", ""
            ),
            "DEPLOYMENT": os.environ.get(
                "AZURE_DEPLOYMENT_POST_AUDIO_LLM_INFERENCE_MODEL", ""
            ),
            "API_VERSION": os.environ.get(
                "AZURE_API_VERSION_POST_AUDIO_LLM_INFERENCE_MODEL", ""
            ),
        },
        "POSITIVE_LLM_RESPONSE_MODEL": {
            "ENDPOINT": os.environ.get(
                "AZURE_ENDPOINT_POSITIVE_LLM_RESPONSE_MODEL", ""
            ),
            "API_KEY": os.environ.get("AZURE_API_KEY_POSITIVE_LLM_RESPONSE_MODEL", ""),
            "DEPLOYMENT": os.environ.get(
                "AZURE_DEPLOYMENT_POSITIVE_LLM_RESPONSE_MODEL", ""
            ),
            "API_VERSION": os.environ.get(
                "AZURE_API_VERSION_POSITIVE_LLM_RESPONSE_MODEL", ""
            ),
        },
        "SUMMARIZE_LLM_MODEL": {
            "ENDPOINT": os.environ.get("AZURE_ENDPOINT_SUMMARIZE_LLM_MODEL", ""),
            "API_KEY": os.environ.get("AZURE_API_KEY_SUMMARIZE_LLM_MODEL", ""),
            "DEPLOYMENT": os.environ.get("AZURE_DEPLOYMENT_SUMMARIZE_LLM_MODEL", ""),
            "API_VERSION": os.environ.get("AZURE_API_VERSION_SUMMARIZE_LLM_MODEL", ""),
        },
        "EMBEDDING_MODEL": {
            "ENDPOINT": os.environ.get("AZURE_ENDPOINT_EMBEDDING_MODEL", ""),
            "API_KEY": os.environ.get("AZURE_API_KEY_EMBEDDING_MODEL", ""),
            "DEPLOYMENT": os.environ.get("AZURE_DEPLOYMENT_EMBEDDING_MODEL", ""),
            "API_VERSION": os.environ.get("AZURE_API_VERSION_EMBEDDING_MODEL", ""),
        },
        "VISION_MODEL": {
            "ENDPOINT": os.environ.get("AZURE_ENDPOINT_VISION_MODEL", ""),
            "API_KEY": os.environ.get("AZURE_API_KEY_VISION_MODEL", ""),
            "DEPLOYMENT": os.environ.get("AZURE_DEPLOYMENT_VISION_MODEL", ""),
            "API_VERSION": os.environ.get("AZURE_API_VERSION_VISION_MODEL", ""),
        },
    }

config_settings = Settings()
