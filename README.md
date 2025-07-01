# langchain-clarifai

This package contains the LangChain integration with Clarifai

## Installation

```bash
pip install -U langchain-clarifai
```

And you should configure credentials by setting the following environment variables:

```
export CLARIFAI_PAT="Your PAT"
```

If you don't know your PAT, please get it here: https://clarifai.com/settings/security


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
