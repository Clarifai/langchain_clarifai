"""Test Clarifai embeddings."""

from typing import Type

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_clarifai.embeddings import ClarifaiEmbeddings


class TestClarifaiEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[ClarifaiEmbeddings]:
        return ClarifaiEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {
            "model_url": "https://clarifai.com/clarifai/main/models/BAAI-bge-base-en-v15"
        }
