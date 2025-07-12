from os import remove
from os.path import expanduser, isfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Union, Optional, Dict, List, Any
from urllib.parse import urlparse

import requests
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredCSVLoader,
)
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from loguru import logger

from workflows.loader.utils import FileLoaderException, split_text
from workflows.settings import config_settings


class UnifiedLoader:

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
        if isinstance(file_path, str) and self._is_valid_url(file_path):
            self._temp_file = NamedTemporaryFile(delete=False)
            try:
                from workflows.settings import config_settings
                timeout = getattr(config_settings, 'REQUEST_TIMEOUT', 30)

                resp = requests.get(
                    file_path, 
                    headers=self.headers, 
                    timeout=timeout,
                    stream=True
                )
                resp.raise_for_status()

                content_length = resp.headers.get('content-length')
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    max_size = getattr(config_settings, 'MAX_FILE_SIZE_MB', 50)
                    if size_mb > max_size:
                        raise ValueError(f"File size ({size_mb:.1f}MB) exceeds maximum allowed size ({max_size}MB)")

                total_size = 0
                max_size_bytes = getattr(config_settings, 'MAX_FILE_SIZE_MB', 50) * 1024 * 1024

                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        total_size += len(chunk)
                        if total_size > max_size_bytes:
                            raise ValueError(f"File size exceeds maximum allowed size ({max_size_bytes / (1024*1024)}MB)")
                        self._temp_file.write(chunk)

                if total_size == 0:
                    raise ValueError("Downloaded file is empty")

                self._temp_file.flush()
                return Path(self._temp_file.name)

            except requests.exceptions.Timeout:
                if self._temp_file:
                    self._temp_file.close()
                    remove(self._temp_file.name)
                raise ValueError(f"Request timeout while downloading file from {file_path}")
            except requests.exceptions.RequestException as e:
                if self._temp_file:
                    self._temp_file.close()
                    remove(self._temp_file.name)
                raise ValueError(f"Failed to download file from {file_path}: {str(e)}")
            except Exception as e:
                if self._temp_file:
                    self._temp_file.close()
                    remove(self._temp_file.name)
                raise ValueError(f"Error processing file from {file_path}: {str(e)}")

        path = Path(file_path)
        if "~" in str(path):
            path = Path(expanduser(str(path)))

        if not isfile(str(path)):
            raise ValueError(f"File path {file_path!r} is not a valid file or URL")

        file_size = path.stat().st_size
        if file_size == 0:
            raise ValueError(f"File {file_path!r} is empty")

        from workflows.settings import config_settings
        max_size_bytes = getattr(config_settings, 'MAX_FILE_SIZE_MB', 50) * 1024 * 1024
        if file_size > max_size_bytes:
            size_mb = file_size / (1024 * 1024)
            max_mb = max_size_bytes / (1024 * 1024)
            raise ValueError(f"File size ({size_mb:.1f}MB) exceeds maximum allowed size ({max_mb}MB)")

        return path

    def load(self) -> List[Document]:
        documents = self.loader.load()
        if not documents:
            raise ValueError(f"No documents loaded from {self.file_path}")
        return documents

    def __del__(self) -> None:
        if hasattr(self, "_temp_file") and self._temp_file:
            self._temp_file.close()
            remove(self._temp_file.name)


class FileLoader(BaseLoader):
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
            raise ValueError(
                f"Unsupported file type: {self.file_type}. Supported types: {', '.join(self.LOADER_MAP.keys())}")

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
    if file_type not in config_settings.FILE_TYPE:
        raise FileLoaderException(f"{file_type} is not a supported file type")

    if file_type.lower() in config_settings.FILE_TYPE:
        loader = FileLoader(
            file_path=pre_signed_url,
            file_type=file_type
        )

        documents = loader.load()
        logger.info(f"Documents loaded: {len(documents)}")

    parsed_documents = split_text(documents)

    if original_file_name or file_name is not None and original_file_name != file_name:
        for doc in parsed_documents:
            doc.metadata["original_file_name"] = original_file_name
            doc.metadata["file_name"] = file_name

    return parsed_documents
