from abc import ABC, abstractmethod
from typing import List
from langchain.schema.document import Document
import os
from langchain.vectorstores import FAISS

class BaseVectorStoreIndexer(ABC):
    @abstractmethod
    def index_documents(self, documents: List[Document]):
        """Indexes the provided documents into a vector store."""
        pass

    @abstractmethod
    def save_index(self, save_path: str):
        """Saves the index to the specified path."""
        pass

    @abstractmethod
    def load_index(self, load_path: str):
        """Loads the index from the specified path."""
        pass

    @abstractmethod
    def query(self, query: str, k: int = 5):
        """Queries the index with the given query string."""
        pass

    @abstractmethod
    async def aquery(self, query: str, k: int = 5):
        """Asynchronously queries the index with the given query string."""
        pass

class FAISSVectorStoreIndexer(BaseVectorStoreIndexer):
    def __init__(self, embedding_model: ChatOpenAI = None):
        if embedding_model is None:
            print("No embedding model provided. Using default model.")
        else:
            self.embedding_model = embedding_model
        self.index = None

    def index_documents(self, documents: List[Document]):
        """Indexes the provided documents into a FAISS vector store."""
        self.index = FAISS.from_documents(documents, self.embedding_model)

    def save_index(self, save_path: str):
        """Saves the FAISS index to the specified path."""
        if self.index is not None:
            os.makedirs(save_path, exist_ok=True)
            self.index.save_local(save_path)
        else:
            raise ValueError("Index has not been created yet. Call 'index_documents' first.")

    def load_index(self, load_path: str):
        """Loads the FAISS index from the specified path."""
        self.index = FAISS.load_local(load_path, self.embedding_model, allow_dangerous_deserialization=True)

    def query(self, query: str, k: int = 5):
        """Queries the FAISS index with the given query string."""
        if self.index is not None:
            docs = self.index.similarity_search(query, k=k)
            return docs
        else:
            raise ValueError("Index has not been created yet. Call 'index_documents' or 'load_index' first.")
    
    async def aquery(self, query: str, k: int = 5):
        """Asynchronously queries the FAISS index with the given query string."""
        if self.index is not None:
            docs = await self.index.asimilarity_search(query, k=k)
            return docs
        else:
            raise ValueError("Index has not been created yet. Call 'index_documents' or 'load_index' first.")
