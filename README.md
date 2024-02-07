## Details

| Developed by | Guardrails AI |
| --- | --- |
| Date of development | Feb 15, 2024 |
| Validator type | Quality |
| Blog | - |
| License | Apache 2 |
| Input/Output | Output |

## Description

The objective of this validator is to ensure that any LLM-generated text is similar (in content) to a previously known document. This validator works comparing the LLM generated text with a known “good” document based on cosine similarity.

### Intended use

- Primary intended uses: When generating an LLM output, if a “golden” output is known for a similar subject. E.g., there’s previous historical data about a support QA system, and we want to ensure that any new LLM generated text will be similar to historical “golden” QA.
- Out-of-scope use cases:

## Example Usage Guide

## Quick Start

### Installation

```bash
$ guardrails hub install hub://guardrails/similar-to-document

```

### Quick Test

```jsx
$ guardrail validate --valdiator SimilarToDocument 'llm output' --document-filepath='./filepath'
```

## Integrating in to application

### Configuring the validator

```python
from guardrails.hub import SimilarToDocument
from guardrails import Guard

with open("/path/to/good/doc.txt", "r") as f:
    doc = f.read()

similar_to_document_val = SimilarToDocument(
    document=doc,
    threshold=0.8,
    model="text-embedding-ada-002"
)
```

### Creating a Guard with the validator and your application

```python
from guardrails import Guard

guard = Guard.from_string(
    validators=[similar_to_document_val],
    num_reasks=2,
    prompt="Generate a poem in the style of Shakespeare about Guardrails."
)
```

## API Reference

`__init__`

- `document` - The document to use for the similarity check.
- `threshold` - The minimum cosine similarity to be considered similar. Defaults to 0.7.
- `model` - The embedding model to use. Defaults to text-embedding-ada-002.

## Expected deployment metrics

|  | CPU | GPU |
| --- | --- | --- |
| Latency | 300 ms | - |
| Memory | N/A | - |
| Cost | $10^-4 / query | - |
| Expected quality | N/A | - |

## Resources required

- Dependencies: Embedding model
- Foundation model access keys: Yes, if commercial embedding model used
- Compute: No

## Validator Performance

### Evaluation Dataset

N/A

### Model Performance Measures

N/A

### Decision thresholds

N/A
