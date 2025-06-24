from typing import Union, Optional, Dict, List, Any
from pathlib import Path

import requests

from domains.doc_loader import FILE_TYPE
from typing import get_args
from domains.doc_loader.utils import FileLoaderException, split_text
from loguru import logger

from os.path import expanduser, isfile
from os import remove
from urllib.parse import urlparse
from tempfile import NamedTemporaryFile

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredCSVLoader,
)


class UnifiedLoader:
    """Handles loading files from both local paths and URLs"""
    def __init__(
            self,
            base_loader_cls,
            file_path: Union[str, Path],
            headers: Optional[Dict[str, Any]] = None,
            **unstructured_kwargs: Any,
    ):
        self._temp_file = None
        self.headers = headers or {}
        self.file_path = self._setup_file_path(file_path)
        self.loader = base_loader_cls(
            file_path=str(self.file_path),
            **unstructured_kwargs
        )

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        parsed = urlparse(url)
        return bool(parsed.scheme) and bool(parsed.netloc)

    def _setup_file_path(self, file_path: Union[str, Path]) -> Path:
        """Set up file path from URL or local path"""
        if isinstance(file_path, str) and self._is_valid_url(file_path):
            self._temp_file = NamedTemporaryFile(delete=False)
            try:
                resp = requests.get(file_path, headers=self.headers)
                resp.raise_for_status()
                self._temp_file.write(resp.content)
                self._temp_file.flush()
                return Path(self._temp_file.name)
            except Exception:
                if self._temp_file:
                    self._temp_file.close()
                    remove(self._temp_file.name)
                raise

        path = Path(file_path)
        if "~" in str(path):
            path = Path(expanduser(str(path)))
        if isfile(str(path)):
            return path

        raise ValueError(f"File path {file_path!r} is not a valid file or URL")

    def load(self) -> List[Document]:
        documents = self.loader.load()
        if not documents:
            raise ValueError(f"No documents loaded from {self.file_path}")
        return documents

    def __del__(self) -> None:
        """Clean up temporary files"""
        if hasattr(self, "_temp_file") and self._temp_file:
            self._temp_file.close()
            remove(self._temp_file.name)


class FileLoader(BaseLoader):
    """Maps file types to appropriate loaders"""
    LOADER_MAP = {
        "txt": TextLoader,
        "pdf": PyMuPDFLoader,
        "docx": UnstructuredWordDocumentLoader,
        "xlsx": UnstructuredExcelLoader,
        "csv": UnstructuredCSVLoader
    }

    def __init__(self, file_path: Union[str, Path], file_type: str = "txt", headers: Optional[Dict[str, Any]] = None):
        self.file_path = str(file_path) if hasattr(file_path, '__fspath__') else file_path
        self.file_type = file_type.lower()
        self.headers = headers or {}

        if self.file_type not in self.LOADER_MAP:
            raise ValueError(f"Unsupported file type: {self.file_type}. Supported types: {', '.join(self.LOADER_MAP.keys())}")

    def load(self) -> List[Document]:
        loader_cls = self.LOADER_MAP.get(self.file_type)
        loader = UnifiedLoader(loader_cls, file_path=self.file_path, headers=self.headers)
        return loader.load()


def file_loader(
    pre_signed_url: str,
    file_name: str,
    original_file_name: str,
    file_type: str,
) -> list[Document]:

    if file_type not in get_args(FILE_TYPE):
        raise FileLoaderException(f"{file_type} is not a supported file type")

    if file_type.lower() in get_args(FILE_TYPE):
        loader = FileLoader(
            file_path=pre_signed_url,
            file_type=file_type
        )

        documents = loader.load()
        logger.info(f"Documents loaded: {len(documents)}")

    parsed_documents = split_text(documents)

    return parsed_documents