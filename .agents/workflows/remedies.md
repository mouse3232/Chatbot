---
description: Remedies Workflow
---

# Remedies Workflow

## High-Level Flow

When a query is received by the Orchestrator Agent, it first checks whether the user intent is already predefined in the incoming payload. If the intent is explicitly available, the Orchestrator skips its usual processing steps such as intent detection and directly delegates the request to the Remedies Agent. This reduces unnecessary computation and improves overall response speed.

## Remedies Agent Processing

Once the Remedies Agent receives the query, it begins by performing its own intent classification to determine which specific tables need to be accessed and to define the appropriate *k-value* for retrieval. Based on this classification, the agent identifies the most relevant datasets within the Session-specific vector database (Session_id_rm VDB), ensuring that only precise and necessary records are considered.

## Data Retrieval and Embedding

After identifying the required sources, the Embedding Agent retrieves the relevant data from the VDB and passes it back to the Remedies Agent. This step ensures that the information is contextually aligned with the query and optimized for downstream processing.

## Response Generation

The Remedies Agent then processes the retrieved data according to its internal logic and predefined instructions. It synthesizes a structured and context-aware response tailored to the user’s query and prepares it for API delivery.

## Language Handling and Optimization

If the detected language code is different from 0, the generated response is sent back to the Orchestrator Agent for translation into the user’s preferred language before being returned. This streamlined workflow minimizes redundant LLM calls, improves latency, and ensures token-efficient processing while maintaining high-quality, contextually relevant responses.
