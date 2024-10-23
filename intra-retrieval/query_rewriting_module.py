from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from intra_retrieval.base_artf_module import BaseARTFModules

class LLMQueryRewritingModule(BaseARTFModules):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm=llm)

    def _initialize_structured_llm(self):
        class RewrittenQuery(BaseModel):
            rewritten_query: str = Field(description="The rewritten query.")
        return self.llm.with_structured_output(RewrittenQuery)

    def _get_system_message(self) -> str:
        return """You are an intelligent assistant designed to rewrite user queries for better understanding and clarity.
Your task is to analyze the user's input, identify ambiguities, and use the previous chat history for context. Correct grammar, clarify terms, and rewrite the query concisely for better understanding.

Example 1:
-----------
Previous Chat History: ["User: status of project?", "Assistant: The project is 80% complete."]
User Input: "timeline?"
Rewritten Query: "What is the project timeline?"

Example 2:
-----------
Previous Chat History: ["User: How's the product launch going?", "Assistant: The launch is on schedule."]
User Input: "what about it?"
Rewritten Query: "Can you give me more details about the product launch?"

Example 3 (Multi-hop Query):
-----------
Previous Chat History: []
User Input: "current value of inv of 5k, yearly flows 3k for 3 yrs @ R 3.5, also IRR for another one 7k cost, 4k flows for 8 yrs, 2.75%"
Rewritten Query: "What is the NPV of an initial investment of $5,000 with yearly cash flows of $3,000 for 3 years at a 3.5% rate? Also, calculate the internal rate of return (IRR) for another investment with an initial cost of $7,000 and yearly cash flows of $4,000 for 8 years."
-----------
"""

    def _get_human_message(self, user_question: str, conversation_history: List[Any]) -> str:
        return f"""Previous Chat History: {conversation_history}
User Input: '{user_question}'
Rewritten Query:"""

    def _get_rewrite_messages(self, user_question: str, conversation_history: List[Any]):
        system_message = self._get_system_message()
        human_message = self._get_human_message(user_question, conversation_history)

        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_message),
                HumanMessagePromptTemplate.from_template(human_message)
            ]
        )

        messages = chat_template.format_messages(
            conversation_history=conversation_history,
            user_question=user_question
        )

        return messages

    def generate(self, query: str, conversation_history: Optional[List[str]] = []) -> str:
        messages = self._get_rewrite_messages(query, conversation_history)
        result = self.structured_llm.invoke(messages)
        return result.rewritten_query

    async def agenerate(self, query: str, conversation_history: Optional[List[str]] = []) -> str:
        messages = self._get_rewrite_messages(query, conversation_history)
        result = await self.structured_llm.ainvoke(messages)
        return result.rewritten_query
