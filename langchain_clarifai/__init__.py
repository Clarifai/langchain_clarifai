from importlib import metadata

from langchain_clarifai.chat_models import ChatClarifai
from langchain_clarifai.embeddings import ClarifaiEmbeddings
from langchain_clarifai.vectorstores import ClarifaiVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatClarifai",
    "ClarifaiVectorStore",
    "ClarifaiEmbeddings",
    "__version__",
]
