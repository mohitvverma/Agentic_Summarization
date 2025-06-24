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

from domains.settings import config_settings


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


import base64
import pprint
import re
import io
import os

import mimetypes
from pathlib import Path
from typing import Optional, Dict, Callable, Any, Tuple, Union
from functools import partial

from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from langchain_community.document_loaders import UnstructuredImageLoader
from langchain_core.document_loaders import BaseLoader
from domains.doc_loader.utils import ImageProcessingError, FileLoaderException

# Import PIL for image resizing
try:
    from PIL import Image
except ImportError:
    logger.warning("PIL not installed. Image resizing will not be available.")
    Image = None


class ImageLoader:
    # Maximum size for Bedrock in bytes (5MB)
    MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024

    def __init__(self, file_path: str, process_type: str, image_type: Optional[str] = None):
        self.file_path = Path(file_path)
        self.process_type = process_type
        self.image_type = image_type or self._guess_mime_type()
        self.validate_file()

    def _guess_mime_type(self) -> str:
        """Guess the MIME type of the image file."""
        mime_type, _ = mimetypes.guess_type(self.file_path)
        if not mime_type and self.image_type:
            return "image/jpeg"  # default fallback

        elif self.image_type == "jpg":
            mime_type = "image/jpeg"

        elif self.image_type == "jpeg":
            mime_type = "image/jpeg"

        elif self.image_type == "png":
            mime_type = "image/png"

        elif self.image_type == "gif":
            mime_type = "image/gif"

        elif self.image_type == "webp":
            mime_type = "image/webp"

        logger.debug("Guessed MIME type: %s", mime_type)
        return mime_type

    def validate_file(self) -> None:
        """Validate if file exists and has supported extension."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        if self.file_path.suffix.lower()[1:] not in config_settings.SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type: {self.file_path.suffix}")

    def resize_image_if_needed(self) -> Union[bytes, None]:
        """
        Resize the image if it exceeds the maximum size limit.
        Returns the resized image as bytes if resizing was needed, None otherwise.
        """
        if not Image:
            logger.warning("PIL not available, skipping image resize check")
            return None

        # Check file size
        file_size = os.path.getsize(self.file_path)
        if file_size <= self.MAX_IMAGE_SIZE_BYTES:
            logger.debug(f"Image size {file_size} bytes is within limits, no resize needed")
            return None

        logger.info(f"Image size {file_size} bytes exceeds {self.MAX_IMAGE_SIZE_BYTES} bytes limit, resizing")

        try:
            # Open the image
            img = Image.open(self.file_path)

            # Start with original dimensions
            width, height = img.size
            quality = 95
            output = io.BytesIO()

            # Try progressively smaller sizes and quality until under limit
            while True:
                output.seek(0)
                output.truncate(0)

                # Save with current dimensions and quality
                img_format = self.image_type.upper() if self.image_type.upper() != 'JPG' else 'JPEG'
                img.save(output, format=img_format, quality=quality)

                # Check if size is now acceptable
                current_size = output.tell()
                if current_size <= self.MAX_IMAGE_SIZE_BYTES:
                    logger.info(f"Successfully resized image to {current_size} bytes (dimensions: {width}x{height}, quality: {quality})")
                    output.seek(0)
                    return output.getvalue()

                # Reduce quality first (down to 70)
                if quality > 70:
                    quality -= 5
                    continue

                # Then start reducing dimensions
                width = int(width * 0.9)
                height = int(height * 0.9)

                # Don't make images too small
                if width < 800 or height < 800:
                    logger.warning("Image dimensions getting too small, using minimum quality")
                    quality = 70
                    img_resized = img.resize((800, 800), Image.LANCZOS)
                    output.seek(0)
                    output.truncate(0)
                    img_resized.save(output, format=img_format, quality=quality)
                    output.seek(0)
                    return output.getvalue()

                img_resized = img.resize((width, height), Image.LANCZOS)
                img = img_resized

        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def encode_image_to_base64(self) -> str:
        """Encode image to base64 with retry mechanism."""
        try:
            # Check if image needs resizing
            resized_image = self.resize_image_if_needed()

            if resized_image:
                # Use the resized image
                logger.info(f"Using resized version of image {self.file_path}")
                return base64.b64encode(resized_image).decode('utf-8')
            else:
                # Use the original image
                with open(self.file_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {self.file_path}: {str(e)}")
            raise ImageProcessingError(f"Failed to encode image: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def load_and_encode(self) -> Dict[str, Any]:
        """Load and encode image with retry mechanism."""
        try:
            encoded_image = self.encode_image_to_base64()
            return {
                "content": encoded_image,
                "image_url": f"data:image/{self.image_type};base64,{encoded_image}",
                "metadata": {
                    "source": str(self.file_path),
                    "file_name": self.file_path.name,
                    "process_type": self.process_type,
                    "mime_type": self.image_type
                }
            }

        except Exception as e:
            logger.error(f"Error processing image {self.file_path}: {str(e)}")
            raise ImageProcessingError(f"Failed to process image: {str(e)}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def process_image(
        file_path: str,
        process_type: str,
        image_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Process image with retry mechanism and proper error handling."""
    try:
        loader = ImageLoader(file_path, process_type, image_type)

        loaders: Dict[str, Callable[[], BaseLoader]] = {
            "base64": loader.load_and_encode,
            "ocr": partial(UnstructuredImageLoader(file_path).load),
        }

        if process_type not in loaders:
            raise ValueError(f"Unsupported process type: {process_type}")

        result = loaders[process_type]()
        logger.info(f"Successfully processed image: {file_path}")
        return result

    except Exception as e:
        logger.error(f"Error in process_image for {file_path}: {str(e)}")
        raise ImageProcessingError(f"Failed to process image: {str(e)}")


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

    elif file_type.lower() in get_args(config_settings.SUPPORTED_FILE_TYPES):
        image_loader = ImageLoader(
            file_path=pre_signed_url,
            process_type="base64",
            image_type=file_type.lower()
        )
        documents = [Document(
            page_content=image_loader.load_and_encode()["content"],
            metadata={
                "source": pre_signed_url,
                "file_name": file_name,
                "original_file_name": original_file_name,
                "file_type": file_type
            }
        )]

    parsed_documents = split_text(documents)

    return parsed_documents
