from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from intra_retrieval.base_artf_module import BaseARTFModules
from pre_retrieval.vector_store_indexer import BaseVectorStoreIndexer

class InitialToolRetrievalModule(BaseARTFModules):
    def __init__(self, toolshed_knowledge_base: BaseVectorStoreIndexer):
        self.toolshed_knowledge_base = toolshed_knowledge_base

    def generate(self, query: str, top_k: int) -> List[Document]:
        """Queries the index using the provided query and returns the top_k results."""
        try:
            docs = self.toolshed_knowledge_base.query(query, k=top_k)
            return docs
        except Exception as e:
            raise ValueError(f"Error retrieving tools: {str(e)}")

    async def agenerate(self, query: str, top_k: int) -> List[Document]:
        """Asynchronous query to theindex."""
        try:
            docs = await self.toolshed_knowledge_base.aquery(query, k=top_k)
            return docs
        except Exception as e:
            raise ValueError(f"Error retrieving tools: {str(e)}")
