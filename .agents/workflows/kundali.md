---
description: Kundali Workflow
---

# Kundali Workflow

## High-Level Flow

When a query is received by the Orchestrator Agent, it first evaluates whether the intent is already predefined in the payload. If the intent is explicitly specified as related to Kundali, the Orchestrator bypasses additional intent detection and directly forwards the request to the Kundali Agent. This avoids redundant processing and ensures a faster, more efficient pipeline.

## Kundali Agent Processing

Upon receiving the query, the Kundali Agent performs intent classification to determine the type of Kundali analysis required—such as planetary positions, house-based interpretations, or specific astrological insights. Based on this classification, the agent identifies which tables or datasets need to be accessed and determines the optimal *k-value* for retrieval.

## Data Retrieval and Embedding

The Kundali Agent then queries the Session-specific vector database (Session_id_r1) to fetch relevant user records. The Embedding Agent assists in retrieving contextually aligned data from the VDB, ensuring that only the most relevant astrological information tied to the user’s Kundali is selected and passed back to the Kundali Agent.

## Response Generation

Using the retrieved Kundali data, the Kundali Agent processes the information according to its domain-specific logic and interpretation rules. It synthesizes a personalized and context-aware response, ensuring that insights are accurate and aligned with the user’s astrological profile before preparing the API response.

## Language Handling and Optimization

If the language code is not equal to 0, the generated response is routed back to the Orchestrator Agent for translation into the user’s preferred language. This workflow minimizes unnecessary LLM calls, improves latency, and ensures token-efficient processing while delivering precise and personalized Kundali-based insights.
