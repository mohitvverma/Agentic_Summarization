from typing import Union
from loguru import logger

from domains.settings import config_settings

from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_aws.chat_models import ChatBedrock
from langchain_google_genai import ChatGoogleGenerativeAI


def initialize_chat_model(
        model_key: str = "CHAT_MODEL_NAME",
        temperature: float = 0.02
) -> Union[ChatOpenAI] | None:
    try:
        if config_settings.LLM_SERVICE_TYPE == "openai":
            return ChatOpenAI(
                model=config_settings.LLMS.get(
                    model_key, ""
                ),
                temperature=temperature,
                api_key=config_settings.OPENAI_API_KEY,
            )

        elif config_settings.LLM_SERVICE_TYPE == "azure_openai":
            return AzureChatOpenAI(
                azure_endpoint=config_settings.AZURE_OPENAI_SETTINGS[model_key]["ENDPOINT"],
                azure_deployment=config_settings.AZURE_OPENAI_SETTINGS[model_key][
                    "DEPLOYMENT"
                ],
                api_key=config_settings.AZURE_OPENAI_SETTINGS[model_key]["API_KEY"],
                model=config_settings.LLMS.get(model_key, ""),
                temperature=temperature,
            )

        elif config_settings.LLM_SERVICE_TYPE == "gemini":
            return ChatGoogleGenerativeAI(
                model=config_settings.GEMINI_LLMS.get(model_key, None),
                temperature=temperature,
            )

        elif config_settings.LLM_SERVICE_TYPE == "aws":
            return ChatBedrock(
                model=config_settings.AWS_BEDROCK_LLMS.get(model_key, None),
                aws_access_key_id=config_settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config_settings.AWS_SECRET_ACCESS_KEY,
                region=config_settings.AWS_REGION_NAME,
                temperature=temperature,
            )

        return None

    except Exception as e:
        logger.error(f"Failed to initialize chat model: {str(e)}")
        raise ValueError(f"Failed to initialize chat model: {str(e)}")
