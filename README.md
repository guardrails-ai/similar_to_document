## Overview

| Developed by | Guardrails AI |
| Date of development | Feb 15, 2024 |
| Validator type | Quality |
| Blog |  |
| License | Apache 2 |
| Input/Output | Output |

## Description

The objective of this validator is to ensure that any LLM-generated text is similar (in content) to a previously known document text. This validator works comparing the LLM generated text with a known “good” document text based on cosine similarity.

### Intended use

The primary intended uses of this validator is if a “golden” output is known for a similar subject when generating an LLM output. E.g., there’s previous historical data about a support QA system, and we want to ensure that any new LLM generated text will be similar to historical “golden” QA.

### Requirements

* Dependencies:
    - `sentence-transformers`
    - guardrails-ai>=0.4.0

* Foundation model access keys:
    - Yes, if commercial embedding model used

## Installation

```bash
$guardrails hub install hub://guardrails/similar_to_document
```

## Usage Examples

### Validating string output via Python

In this example, we apply the validator to a string output generated by an LLM.

```python
# Import Guard and Validator
from guardrails import Guard
from guardrails.hub import SimilarToDocument

# Initialize The Guard with this validator
guard = Guard().use(
    SimilarToDocument,
    document="""
    Large language models (LLM) are very large deep learning models that are pre-trained on vast amounts of data. 
    The underlying transformer is a set of neural networks that consist of an encoder and a decoder with self-attention capabilities. 
    The encoder and decoder extract meanings from a sequence of text and understand the relationships between words and phrases in it.
    Transformer LLMs are capable of unsupervised training, although a more precise explanation is that transformers perform self-learning. 
    It is through this process that transformers learn to understand basic grammar, languages, and knowledge.
    """,
    threshold=0.7,
    model="all-MiniLM-L6-v2",
    on_fail="exception",
)

# Test passing response
guard.validate(
    """
    Large Language Models (LLMs) are a type of neural network that can be trained on large amounts of text
    data to generate human-like text. These models have been used in a variety of applications, including
    machine translation, text summarization, and question answering.
    """
)  # Pass

try:
    # Test failing response
    guard.validate(
        """
        Graph neural networks (GNNs) are specialized neural networks that can operate on graph data
        structures. These networks are designed to capture the relationships between nodes in a graph
        and can be used for a variety of tasks, including node classification, link prediction, and graph classification.
        """
    )  # Fail
except Exception as e:
    print(e)
```
Output:
```console
Validation failed for field with errors: 
Value:
    Graph neural networks (GNNs) are specialized neural networks that can operate on graph data
    structures. These networks are designed to capture the relationships between nodes in a graph
    and can be used for a variety of tasks, including node classification, link prediction, and graph classification.
is not similar enough to document:
    Large language models (LLM) are very large deep learning models that are pre-trained on vast amounts of data. 
    The underlying transformer is a set of neural networks that consist of an encoder and a decoder with self-attention capabilities. 
    The encoder and decoder extract meanings from a sequence of text and understand the relationships between words and phrases in it.
    Transformer LLMs are capable of unsupervised training, although a more precise explanation is that transformers perform self-learning. 
    It is through this process that transformers learn to understand basic grammar, languages, and knowledge.
```

# API Reference

**`__init__(self, document, threshold=0.7, model="all-MiniLM-L6-v2", on_fail="noop")`**
<ul>

Initializes a new instance of the Validator class.

**Parameters:**

- **`document`** _(str):_ The text of the document to use for the similarity check.
- **`threshold`** _(float):_ The minimum cosine similarity to be considered similar. Defaults to 0.7.
- **`model`** _(str):_ The embedding model to use. Defaults to `all-MiniLM-L6-v2`. Check the [sentence-transformers documentation](https://www.sbert.net/docs/pretrained_models.html) for available models.
- **`on_fail`** *(str, Callable):* The policy to enact when a validator fails. If `str`, must be one of `reask`, `fix`, `filter`, `refrain`, `noop`, `exception` or `fix_reask`. Otherwise, must be a function that is called when the validator fails.

</ul>

<br>

**`__call__(self, value, metadata={}) -> ValidationResult`**

<ul>

Validates the given `value` using the rules defined in this validator, relying on the `metadata` provided to customize the validation process. This method is automatically invoked by `guard.parse(...)`, ensuring the validation logic is applied to the input data.

Note:

1. This method should not be called directly by the user. Instead, invoke `guard.parse(...)` where this method will be called internally for each associated Validator.
2. When invoking `guard.parse(...)`, ensure to pass the appropriate `metadata` dictionary that includes keys and values required by this validator. If `guard` is associated with multiple validators, combine all necessary metadata into a single dictionary.

**Parameters:**

- **`value`** *(Any):* The input value to validate.
- **`metadata`** *(dict):* A dictionary containing metadata required for validation. No additional metadata keys are needed for this validator.

</ul>
