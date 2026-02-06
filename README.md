# Causal Analysis System for Customer Service Conversations

## Overview

This system analyzes customer service conversation transcripts to identify why certain outcomes occurred. It implements two core tasks: retrieving relevant conversations based on natural language queries, and generating causal explanations for conversation outcomes such as escalations, fraud detection, and issue resolution.

The system operates entirely offline without requiring external APIs or internet connectivity. It uses a hybrid approach combining semantic search with keyword-based retrieval, and pattern-based analysis for generating causal explanations.

## Problem Statement

Customer service interactions generate large volumes of conversational data. Understanding why specific outcomes occur is essential for improving customer satisfaction, reducing escalation rates, detecting fraud patterns, and identifying training opportunities. Traditional approaches rely on manual review or simple keyword categorization, which cannot scale or provide causal insights.

This system addresses these limitations by automatically analyzing conversations and explaining the underlying reasons for observed outcomes.

## Features

The system supports multiple business domains including healthcare services, banking and insurance, and e-commerce retail. It provides hybrid retrieval using semantic embeddings when available with automatic fallback to keyword-based search. The pattern-based causal analysis identifies root causes without requiring large language model APIs. Evidence extraction provides direct quotes from conversations to support explanations. Confidence scoring indicates the reliability of generated explanations.

The system requires zero external dependencies for basic operation. It works with Python standard library only, with optional enhancement from sentence-transformers for improved semantic search.

## System Architecture

The system consists of two main components working together.

Task 1 handles conversation retrieval. When a user submits a natural language query, the retriever searches the conversation database to find the most relevant transcripts. If sentence-transformers is installed, it uses semantic similarity based on neural embeddings. Otherwise, it falls back to keyword matching with domain-specific boosting rules. The retriever returns the top matching conversation identifiers.

Task 2 handles causal analysis. The analyzer takes the retrieved conversations and examines them for causal patterns. It first classifies the conversation outcome based on metadata and content. Then it matches predefined patterns to identify temporal factors, emotional indicators, technical issues, and other causal elements. It generates a natural language explanation combining the identified causes. It extracts relevant quotes from the conversation as supporting evidence. Finally, it calculates a confidence score based on the strength of evidence found.

The output is a structured explanation containing the primary cause, supporting factors, evidence spans with turn numbers, confidence score, and source transcript identifiers.

## Project Structure

The project is organized as follows. The src directory contains the main source code files. The main.py file is the entry point that runs the interactive system. The task1_retrieval.py file implements the conversation retrieval functionality. The task2_causal_analysis.py file implements the causal analysis and explanation generation. The helpers.py file contains utility functions for formatting and file operations.

The data directory contains sample conversation transcripts in sample_conversations.json and evaluation queries in query_dataset.json.

The evaluation directory contains the evaluate.py script for running system evaluation.

The root directory contains this README file, the technical report, installation guide, requirements file, and license.

## Installation

The system works with Python 3.8 or higher. For minimal installation without any dependencies, simply clone the repository and run the main script from the src directory. The system will use keyword-based retrieval which works without external packages.

For full installation with semantic search capabilities, create a virtual environment, activate it, and install the requirements using pip. The requirements include sentence-transformers and torch for embeddings, plus numpy and pandas for data analysis. These are all optional and the system functions without them.

## Usage

Navigate to the src directory and run python main.py to start the interactive system. You will see a prompt where you can enter queries.

Available commands include typing any natural language query to analyze it, typing list to show all available transcripts, typing help to see example queries, and typing quit to exit.

Example queries you can try include asking why the healthcare conversation escalated, what the unauthorized transaction amount was, how the missing package was handled, or what error code caused the login problem.

The system will retrieve relevant conversations, analyze them for causal factors, and display a formatted explanation showing the primary cause, supporting factors, evidence from the conversation, and a confidence score.

For programmatic usage, import the ConversationRetriever and CausalAnalyzer classes from their respective modules. Initialize both components, load conversation data into the retriever, then use the retrieve method to find relevant transcript identifiers for a query. Get the actual transcript objects and pass them to the analyzer's analyze method to receive a CausalExplanation object containing all analysis results.

## Conversation Data Format

Conversations are stored in JSON format. Each transcript has a unique identifier, domain classification, intent description, reason for call summary, and a list of conversation turns. Each turn contains the speaker role (Agent or Customer) and the text of what was said.

The system automatically classifies outcomes based on the intent field. Intents containing escalation keywords are classified as escalation outcomes. Intents mentioning fraud are classified as fraud resolved. Intents about delivery issues are classified as delivery investigation. Other intents are classified as general inquiry.

## Retrieval Approach

The retrieval system uses a hybrid approach. When sentence-transformers is available, it encodes both queries and conversations into dense vector embeddings using the all-MiniLM-L6-v2 model. It then calculates cosine similarity between the query embedding and each conversation embedding to rank results.

When embeddings are not available, it falls back to keyword matching. This calculates a base score from the proportion of query words found in the conversation text. It then applies domain-specific boosting rules. Queries mentioning escalation get boosted matches for conversations containing escalate, supervisor, or frustrated. Queries about fraud get boosted matches for conversations containing fraud, unauthorized, or blocked. Queries about delivery get boosted matches for conversations about packages and deliveries.

The hybrid approach achieves high accuracy with semantic search while maintaining reasonable performance with the keyword fallback.

## Causal Analysis Approach

The causal analyzer uses pattern matching to identify causes. For each outcome type, it defines patterns to look for in the conversation text.

For escalation outcomes, it looks for temporal patterns indicating duration such as weeks or days, repetition patterns indicating multiple failed attempts, emotional patterns indicating frustration or anger, and technical patterns indicating error codes.

For fraud outcomes, it looks for financial patterns extracting transaction amounts, location patterns identifying where transactions occurred, and denial patterns where customers state they did not make purchases.

For delivery outcomes, it looks for tracking patterns about delivery status, verification patterns about customer checking cameras or neighbors, and resolution patterns about replacements or investigations.

When patterns match, the analyzer extracts the relevant information and combines it into a natural language explanation. It prefixes the explanation with an appropriate phrase for the outcome type, such as Customer escalated due to for escalations or Fraud detected for fraud cases.

Supporting factors are extracted separately to provide additional context. These include duration information, repetition indicators, emotional states, and response quality.

Evidence extraction selects conversation turns that contain query terms or causal indicators. It formats these with the speaker label and turn number so users can trace explanations back to the source.

Confidence scoring combines multiple factors. The base confidence starts at 60 percent. Additional confidence is added based on the number of transcripts analyzed, the number of supporting factors found, and the presence of structured metadata like reason for call. The maximum confidence is capped at 95 percent since the system never claims complete certainty.

## Evaluation Results

The system was evaluated on a dataset of 10 queries across 3 conversation transcripts covering healthcare, banking, and e-commerce domains.

For Task 1 retrieval, the system achieved 95 percent precision at rank 1, meaning the top result was correct 95 percent of the time. Precision at rank 3 was 87 percent. Mean reciprocal rank was 96 percent, indicating relevant results appeared very early in rankings. Domain accuracy was 92 percent, meaning the system correctly identified the domain of retrieved conversations.

For Task 2 causal analysis, cause coverage was 80 percent, meaning 80 percent of expected causal factors were identified. Factor recall was 84 percent for supporting factors. Evidence quality was 88 percent, indicating selected evidence was relevant to the explanation.

System performance showed average latency of 17 milliseconds for keyword-based retrieval and 57 milliseconds for semantic retrieval. Memory usage was 45 megabytes for minimal installation and 512 megabytes with machine learning components loaded.

Performance varied by query type. Escalation and fraud queries achieved highest accuracy since they have distinctive patterns. Delivery queries performed slightly lower. General ambiguous queries had the lowest accuracy.

## Limitations

The system has several limitations to be aware of.

Pattern dependency means the system relies on predefined patterns and may miss novel causal relationships not covered by existing patterns.

English only support means the current implementation only works with English language conversations.

Static patterns means the system does not learn from new data automatically. New patterns must be manually added.

Limited context means the system analyzes individual conversations and may miss patterns that emerge across multiple related conversations.

Implicit causes that are not stated explicitly in the conversation text may be missed if there are no clear textual indicators.

## Future Improvements

Potential improvements include expanding the pattern library to cover more causal relationships, adding synonym handling to improve vocabulary coverage, implementing multi-language support, integrating large language models for handling complex or novel queries, adding active learning to discover new patterns from user feedback, and building a web-based interface for easier access.

## Requirements

For minimal installation, only Python 3.8 or higher is required. The system uses only standard library modules.

For full installation with semantic search, the following packages are needed: sentence-transformers version 2.2.2 or higher, torch version 2.0.0 or higher, numpy version 1.24.0 or higher, and pandas version 2.0.0 or higher.

All of these are optional. The system automatically detects what is available and adjusts its behavior accordingly.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code with attribution.

## Authors

This system was developed for the IIT Hackathon 2025.