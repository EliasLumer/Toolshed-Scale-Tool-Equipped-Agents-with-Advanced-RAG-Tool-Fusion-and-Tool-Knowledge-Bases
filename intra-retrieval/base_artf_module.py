from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI,OpenAIEmbeddings

class BaseARTFModules(ABC):
    def __init__(self, 
                 llm: Optional[ChatOpenAI] = None, 
                 embedder: Optional[OpenAIEmbeddings] = None):
        self.llm = llm
        self.embedder = embedder
        if self.llm:
            self.structured_llm: ChatOpenAI = self._initialize_structured_llm()
        else:
            self.structured_llm = None

    def _initialize_structured_llm(self):
        """Provide a default behavior when LLM is not required."""
        return None 

    @abstractmethod
    def generate(self, **kwargs):
        """Abstract method for generating output."""
        pass

    @abstractmethod
    async def agenerate(self, **kwargs):
        """Abstract method for generating output asynchronously."""
        pass
