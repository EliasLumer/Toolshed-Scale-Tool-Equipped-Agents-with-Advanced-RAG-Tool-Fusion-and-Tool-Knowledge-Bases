# Toolshed Knowledge Base with Advanced RAG-Tool Fusion

This repository contains sample code snippets associated with our research paper **"Toolshed Knowledge Base with Advanced RAG-Tool Fusion"**. The code is part of a Medium post and demonstrates some of the core concepts discussed in the paper.

## Abstract

Recent advancements in tool-equipped Agents (LLMs) have enabled complex tasks like secure database interactions and multi-agent code development. However, scaling tool capacity beyond agent reasoning or model limits remains a challenge. In this paper, we address these challenges by introducing Toolshed Knowledge Bases, a tool knowledge base (vector database) designed to store enhanced tool representations and optimize tool selection for large-scale tool-equipped Agents. Additionally, we propose Advanced RAG-Tool Fusion, a novel ensemble of tool-applied advanced retrieval-augmented generation (RAG) techniques across the pre-retrieval, intra-retrieval, and post-retrieval phases, without requiring model fine-tuning. During pre-retrieval, tool documents are enhanced with key information and stored in the Toolshed Knowledge Base. Intra-retrieval focuses on query planning and transformation to increase retrieval accuracy. Post-retrieval refines the retrieved tool documents and enables self-reflection. Furthermore, by varying both the total number of tools (tool-M) an Agent has access to and the tool selection threshold (top-k), we address trade-offs between retrieval accuracy, agent performance, and token cost. Our approach achieves 46%, 56%, and 47% absolute improvements on the ToolE single-tool, ToolE multi-tool and Seal-Tools benchmark datasets, respectively (Recall@5).

## Link to Paper

You can read the full paper on arXiv [here](https://arxiv.org/abs/2410.14594).
