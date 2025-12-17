"""Test ChatClarifai wrapper."""

import sys
import os
import pytest
from langchain_clarifai.chat_models import ChatClarifai

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult


if sys.version_info < (3, 11):
    pytest.skip("clarifai requires Python >= 3.11", allow_module_level=True)

MODEL_URL = os.environ.get("CLARIFAI_MODEL_URL", "https://clarifai.com/meta/Llama-3/models/Llama-3_2-3B-Instruct")
MODEL_NAME = MODEL_URL.split("/")[-1]


@pytest.fixture
def chat() -> ChatClarifai:
    return ChatClarifai(
        model_url=MODEL_URL, model_kwargs={"temperature": 0, "max_tokens": 20}
    )


def test_chat_clarifai(chat: ChatClarifai) -> None:
    """Test ChatClarifai wrapper."""
    message = HumanMessage(content="What is the weather in Redwood City, CA today")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_clarifai_model() -> None:
    """Test ChatClarifai wrapper handles model_name."""
    chat = ChatClarifai(model_url=MODEL_URL)
    assert chat.model_id == MODEL_NAME


def test_chat_clarifai_system_message(chat: ChatClarifai) -> None:
    """Test ChatClarifai wrapper with system message."""
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_clarifai_generate() -> None:
    """Test ChatClarifai wrapper with generate."""
    chat = ChatClarifai(model_url=MODEL_URL, model_kwargs={"max_tokens": 20})
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]], stream=False)
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


def test_chat_clarifai_multiple_completions() -> None:
    """Test ChatClarifai wrapper with multiple completions."""
    chat = ChatClarifai(model_url=MODEL_URL, model_kwargs={"max_tokens": 20})
    message = HumanMessage(content="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 1
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)


def test_chat_clarifai_llm_output_contains_model_id(chat: ChatClarifai) -> None:
    """Test llm_output contains model_id."""
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_id"] == chat.model_id


def test_clarifai_invoke(chat: ChatClarifai) -> None:
    """Tests chat completion with invoke"""
    result = chat.invoke("How is the weather in New York today?", max_tokens=100)
    assert isinstance(result.content, str)


def test_clarifai_batch(chat: ChatClarifai) -> None:
    """Test batch tokens from ChatClarifai."""
    result = chat.batch(
        [
            "What is the weather in Redwood City, CA today?",
            "What is the weather in Redwood City, CA today?",
            "What is the weather in Redwood City, CA today?",
        ],
        config={"max_concurrency": 1},
    )
    for token in result:
        assert isinstance(token.content, str)


def test_clarifai_streaming(chat: ChatClarifai) -> None:
    """Test streaming tokens from Clarifai."""

    for token in chat.stream("I'm a pickle"):
        assert isinstance(token.content, str)


