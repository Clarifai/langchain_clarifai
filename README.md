# langchain-clarifai

This package contains the LangChain integration with Clarifai

## Installation

```bash
pip install -U langchain-clarifai
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatClarifai` class exposes chat models from Clarifai.

```python
from langchain_clarifai import ChatClarifai

llm = ChatClarifai()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`ClarifaiEmbeddings` class exposes embeddings from Clarifai.

```python
from langchain_clarifai import ClarifaiEmbeddings

embeddings = ClarifaiEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```
