from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from domains.settings import config_settings


class FileLoaderException(Exception):
    pass


def split_text(
        text: list[Document],
        chunk_size: int = config_settings.CHUNK_SIZE,
        chunk_overlap: int = config_settings.CHUNK_OVERLAP
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(text)
    logger.info(f'Splitting documents... Now you have {len(texts)} documents'
                f' with chunk size {chunk_size} and chunk overlap {chunk_overlap}')
    return texts
