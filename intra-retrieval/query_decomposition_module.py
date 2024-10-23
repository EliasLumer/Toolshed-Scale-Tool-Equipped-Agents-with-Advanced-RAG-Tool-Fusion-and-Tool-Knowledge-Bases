from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from intra_retrieval.base_artf_module import BaseARTFModules
from pydantic import BaseModel, Field

class QueryDecompositionModule(BaseARTFModules):
    def __init__(self, llm: AzureChatOpenAI):
        super().__init__(llm=llm)

    def _initialize_structured_llm(self):
        class DecomposedQuery(BaseModel):
            decomposed_steps: List[str] = Field(description="The decomposed steps of the query.")
        return self.llm.with_structured_output(DecomposedQuery)

    def _get_system_message(self) -> str:
        return """You are an expert at breaking down user questions into clearly defined step(s).
You will be given a user question that can be answered by a single action or multiple actions (multi-hop queries).
For some questions, the user may be asking for a single action, which is typically just a single topic and query, which can be broken down into one step.
For other questions, the user may be asking for multiple things (usually denoted by the use of 'and' or 'additionally'), which can be broken down into multiple steps.
Your job is to:
1. Break down the question into clearly defined steps.
2. Always be as clear as possible, including the technical details.
3. For multi-step questions, break them down into 2-4 reasoning steps, depending on the complexity of the request.

Example of single-step questions:
-----------
EX. QUESTION:
What is the NPV of my project?
EX. STEP(S):
['What is the NPV of my project?']
-----------

Example of multi-step questions (2+ steps):
-----------
EX. QUESTION:
What's the net present value (NPV) and internal rate of return (IRR) for a project with an initial investment of $5,000, projected annual cash inflows of $1,200 for 7 years, and a 6% discount rate?
EX. STEP(S):
['Calculate the net present value (NPV) for the project with an initial investment of $5,000, projected annual cash inflows of $1,200 for 7 years, and a 6% discount rate.', 'Calculate the internal rate of return (IRR) for the project with the same initial investment of $5,000 and projected annual cash inflows of $1,200 over 7 years.']
-----------
"""

    def _get_human_message(self, user_question: str) -> str:
        return f"""USER QUESTION: {user_question}
STEPS FOR THAT QUERY:"""

    def generate(self, query: str) -> List[str]:
        messages = self._get_decomposition_messages(user_question=query)
        result = self.structured_llm.invoke(messages)
        return result.decomposed_steps

    async def agenerate(self, query: str) -> List[str]:
        messages = self._get_decomposition_messages(user_question=query)
        result = await self.structured_llm.ainvoke(messages)
        return result.decomposed_steps

    def _get_decomposition_messages(self, user_question: str):
        system_message = self._get_system_message()
        human_message = self._get_human_message(user_question=user_question)

        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_message),
                HumanMessagePromptTemplate.from_template(human_message)
            ]
        )

        messages = chat_template.format_messages(
            user_question=user_question
        )

        return messages
