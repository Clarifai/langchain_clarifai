"""Clarifai chat wrapper."""

from __future__ import annotations

import json
import logging
import warnings
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import (
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://clarifai.com/meta/Llama-3/models/Llama-3_2-3B-Instruct"


def convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    elif role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""
        additional_kwargs: Dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        if tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = tool_calls
        if context := _dict.get("context"):
            additional_kwargs["context"] = context
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    elif role == "function":
        return FunctionMessage(content=_dict.get("content", ""), name=_dict.get("name"))  # type: ignore[arg-type]
    elif role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id"),
            additional_kwargs=additional_kwargs,
        )
    else:
        return ChatMessage(content=_dict.get("content", ""), role=role)  # type: ignore[arg-type]


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            # If tool calls only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
        if "context" in message.additional_kwargs:
            message_dict["context"] = message.additional_kwargs["context"]
            # If context only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise TypeError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = _dict["tool_calls"]

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)  # type: ignore[arg-type]
    else:
        return default_class(content=content)  # type: ignore[call-arg]


def _update_token_usage(
    overall_token_usage: Union[int, dict], new_usage: Union[int, dict]
) -> Union[int, dict]:
    # Token usage is either ints or dictionaries
    # `reasoning_tokens` is nested inside `completion_tokens_details`
    if isinstance(new_usage, int):
        if not isinstance(overall_token_usage, int):
            raise ValueError(
                f"Got different types for token usage: "
                f"{type(new_usage)} and {type(overall_token_usage)}"
            )
        return new_usage + overall_token_usage
    elif isinstance(new_usage, dict):
        if not isinstance(overall_token_usage, dict):
            raise ValueError(
                f"Got different types for token usage: "
                f"{type(new_usage)} and {type(overall_token_usage)}"
            )
        return {
            k: _update_token_usage(overall_token_usage.get(k, 0), v)
            for k, v in new_usage.items()
        }
    else:
        warnings.warn(f"Unexpected type for token usage: {type(new_usage)}")
        return new_usage


class ChatClarifai(BaseChatModel):
    """Clarifai Chat models."""

    model_url: Optional[str] = None
    """Model url to use."""
    model_id: Optional[str] = None
    """Model id to use."""
    model_version_id: Optional[str] = None
    """Model version id to use."""
    app_id: Optional[str] = None
    """Clarifai application id to use."""
    user_id: Optional[str] = None
    """Clarifai user id to use."""
    pat: Optional[str] = Field(default=None, exclude=True)  #: :meta private:
    """Clarifai personal access token to use."""
    token: Optional[str] = Field(default=None, exclude=True)  #: :meta private:
    """Clarifai session token to use."""
    api_base: str = "https://api.clarifai.com"
    """Clarifai API base url"""

    deployment_id: Optional[str] = None
    nodepool_id: Optional[str] = None
    compute_cluster_id: Optional[str] = None
    """Clarifai deployment params"""

    max_retries: int = Field(default=1)

    streaming: bool = False
    """Whether to stream the results or not."""

    model_kwargs: dict = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.95,
        }.copy()
    )

    model_name: Optional[str] = None

    model: Any = Field(default=None, exclude=True)  #: :meta private:

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"pat": "CLARIFAI_PAT"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "clarifai"]

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name and field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that we have all required info to access Clarifai
        platform and python package exists in environment."""
        try:
            from clarifai.client import Model
        except ImportError:
            raise ImportError(
                "Could not import clarifai python package. "
                "Please install it with `pip install clarifai`."
            )

        values["pat"] = get_from_dict_or_env(values, "pat", "CLARIFAI_PAT")

        user_id = values.get("user_id")
        app_id = values.get("app_id")
        model_id = values.get("model_id")
        model_version_id = values.get("model_version_id")
        model_url = values.get("model_url")
        api_base = values.get("api_base")
        pat = values.get("pat")
        token = values.get("token")

        deployment_id = values.get("deployment_id")
        nodepool_id = values.get("nodepool_id")
        compute_cluster_id = values.get("compute_cluster_id")

        model = values.get("model")
        if not model or not isinstance(model, Model):
            model = Model(
                url=model_url,
                app_id=app_id,
                user_id=user_id,
                model_version=dict(id=model_version_id),
                pat=pat,
                token=token,
                model_id=model_id,
                base_url=api_base,
                deployment_id=deployment_id,
                nodepool_id=nodepool_id,
                compute_cluster_id=compute_cluster_id,
            )

            values["model"] = model

        if not model_id:
            values["model_id"] = model.id
            values["model_name"] = model.id
        if not user_id:
            values["user_id"] = model.user_app_id.user_id
        if not model_id:
            values["app_id"] = model.user_app_id.app_id
        if not model_version_id:
            values["model_version_id"] = model.model_version.id

        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
            **self.model_kwargs,
        }

    def _create_message_dicts(self, messages: List[BaseMessage]) -> List:
        """Preprocess LC to OpenAI Chat"""
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        """Postprocess OpenAI Chat to LC chat"""

        generations = []
        if isinstance(response, BaseModel):
            response = response.model_dump()
        for res in response["choices"]:
            message = convert_dict_to_message(res["message"])
            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_id,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _combine_llm_outputs(
        self, llm_outputs: List[Union[Dict[Any, Any], None]]
    ) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage:
                        overall_token_usage[k] = _update_token_usage(
                            overall_token_usage[k], v
                        )
                    else:
                        overall_token_usage[k] = v

        combined = {"token_usage": overall_token_usage, "model_id": self.model_id}
        return combined

    # -------------- Generate ----------------- #
    def _completion(self, **kwargs: Any) -> Any:
        params = {**self.model_kwargs}
        params.update(kwargs)
        params.pop("stream", None)

        json_params = json.dumps(params)
        chat_response = self.model.openai_transport(msg=json_params)
        try:
            chat_response = json.loads(chat_response)
            assert "error" not in chat_response
            return chat_response
        except Exception as e:
            logger.error(f"Error in completion: {e} -- Output: {chat_response}")
            raise Exception(chat_response)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        This can be a call to an API, a call to a local model, or any other
        implementation that generates a response to the input prompt.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """

        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts = self._create_message_dicts(messages)
        response = self._completion(messages=message_dicts, **kwargs)
        return self._create_chat_result(response)

    # -------------- Stream ----------------- #

    def _stream_completion(self, **kwargs: Any) -> Any:
        params = {**self.model_kwargs}
        params.update(kwargs)
        params["stream"] = True
        json_params = json.dumps(params)
        chat_response_iteration = self.model.openai_stream_transport(msg=json_params)

        return chat_response_iteration

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method should be implemented if the model can generate output
        in a streaming fashion. If the model does not support streaming,
        do not implement it. In that case streaming requests will be automatically
        handled by the _generate method.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
        """

        message_dicts = self._create_message_dicts(messages)
        params = {**kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        for chunk in self._stream_completion(messages=message_dicts, **params):
            chunk = json.loads(chunk)
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            if choice["delta"] is None:
                continue
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )

            cg_chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info
            )
            if run_manager:
                run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
            yield cg_chunk

    # --------------- Model Info ------------ #
    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        strict: Optional[bool] = None,
        parallel_tool_calls: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with clarifai tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Supports any tool definition handled by
                :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function,
                "auto" to automatically determine which function to call
                with the option to not call any function, "any" to enforce that some
                function is called, or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to
                :meth:`~langchain_clarifai.chat_models.ChatClarifai.bind`
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "any", "none")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    raise ValueError(
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[dict, type[BaseModel]]] = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema:
                The output schema. Can be passed in as:
                    - an OpenAI function/tool schema,
                    - a JSON Schema,
                    - a TypedDict class (support added in 0.1.7),
                    - or a Pydantic class.
                If ``schema`` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated. See :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`
                for more on how to properly specify types and descriptions of
                schema fields when specifying a Pydantic or TypedDict class.

                .. versionchanged:: 0.1.7

                        Added support for TypedDict class.

            method: The method for steering model generation, one of:

                - "function_calling":
                - "json_schema":
                - "json_mode":

                .. versionchanged:: 0.2.8

                    Added support for ``"json_schema"``.

            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            If ``include_raw`` is False and ``schema`` is a Pydantic class, Runnable outputs
            an instance of ``schema`` (i.e., a Pydantic object).

            Otherwise, if ``include_raw`` is False then Runnable outputs a dict.

            If ``include_raw`` is True, then Runnable outputs a dict with keys:
                - ``"raw"``: BaseMessage
                - ``"parsed"``: None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
                - ``"parsing_error"``: Optional[BaseException]

        Example: schema=Pydantic class, method="function_calling", include_raw=False:
            .. code-block:: python

                from typing import Optional

                from langchain_clarifai import ChatClarifai
                from pydantic import BaseModel, Field


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    # If we provide default values and/or descriptions for fields, these will be passed
                    # to the model. This is an important part of improving a model's ability to
                    # correctly return structured outputs.
                    justification: Optional[str] = Field(
                        default=None, description="A justification for the answer."
                    )


                llm = ChatClarifai(url=...)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example: schema=Pydantic class, method="function_calling", include_raw=True:
            .. code-block:: python

                from langchain_clarifai import ChatClarifai
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatClarifai(url=...)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification, include_raw=True
                )

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: schema=TypedDict class, method="function_calling", include_raw=False:
            .. code-block:: python

                # IMPORTANT: If you are using Python <=3.8, you need to import Annotated
                # from typing_extensions, not from typing.
                from typing_extensions import Annotated, TypedDict

                from langchain_clarifai import ChatClarifai


                class AnswerWithJustification(TypedDict):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: Annotated[
                        Optional[str], None, "A justification for the answer."
                    ]


                llm = ChatClarifai(url=...)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        Example: schema=OpenAI function schema, method="function_calling", include_raw=False:
            .. code-block:: python

                from langchain_clarifai import ChatClarifai

                oai_schema = {
                    'name': 'AnswerWithJustification',
                    'description': 'An answer to the user question along with justification for the answer.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'answer': {'type': 'string'},
                            'justification': {'description': 'A justification for the answer.', 'type': 'string'}
                        },
                       'required': ['answer']
                   }
               }

                llm = ChatClarifai(url=...)
                structured_llm = llm.with_structured_output(oai_schema)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        Example: schema=Pydantic class, method="json_mode", include_raw=True:
            .. code-block::

                from langchain_clarifai import ChatClarifai
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatClarifai(url=..., temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_mode",
                    include_raw=True
                )

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'. "
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{"answer": "They are both the same weight.", "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight."}'),
                #     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'),
                #     'parsing_error': None
                # }

        Example: schema=None, method="json_mode", include_raw=True:
            .. code-block::

                structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'. "
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{"answer": "They are both the same weight.", "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight."}'),
                #     'parsed': {
                #         'answer': 'They are both the same weight.',
                #         'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'
                #     },
                #     'parsing_error': None
                # }
        """  # noqa: E501
        _ = kwargs.pop("strict", None)
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": "function_calling"},
                    "schema": formatted_tool,
                },
            )
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_schema":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'json_schema'. "
                    "Received None."
                )
            formatted_schema = convert_to_json_schema(schema)
            llm = self.bind(
                response_format={"type": "json_object", "schema": formatted_schema},
                ls_structured_output_format={
                    "kwargs": {"method": "json_schema"},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_mode":
            llm = self.bind(
                response_format={"type": "json_object"},
                ls_structured_output_format={
                    "kwargs": {"method": "json_mode"},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_mode'. Received: '{method}'"
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

    # --------------- Model Info ------------ #

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "clarifai-chat"


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


def _lc_tool_call_to_clarifai_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _lc_invalid_tool_call_to_clarifai_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }

