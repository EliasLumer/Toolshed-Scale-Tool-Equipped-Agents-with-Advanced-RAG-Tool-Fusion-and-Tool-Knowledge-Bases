from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from intra_retrieval.base_artf_module import BaseARTFModules
from pydantic import BaseModel, Field

class MultiQueryExpansionModule(BaseARTFModules):
    def __init__(self, llm: ChatOpenAI, n_items: int):
        # Call the base class constructor for llm and embedder
        self.n_items = n_items
        super().__init__(llm=llm)
        # Initialize n_items specifically for this module

    def _initialize_structured_llm(self):
        class ExpandedQueries(BaseModel):
            expanded_queries: List[str] = Field(
                description=f"{self.n_items} variations or expanded versions of the user query."
            )
        return self.llm.with_structured_output(ExpandedQueries)

    def _get_system_message(self) -> str:
        return f"""You are an expert at converting user questions into {self.n_items} sentence variations that target different keywords and nuanced approaches, with the goal of embedding these queries into a vector database to retrieve relevant financial equations.
Your goal is to craft {self.n_items} nuanced sentence variations that target different aspects of understanding or solving the query.
While keeping the underlying concept of the query, you can generate variations that focus on more abstract concept of the financial conept, or quantitative version.
You can vary the structure, some variations can be more professional and others more casual.
```
**Example 1:**
**Example user question:** "I want to see the discount rate at which the NPV is break even."
**Example 3 sentences variations:**
1. "Calculate the discount rate that results in a net present value (NPV) of zero for a project, effectively finding the internal rate of return (IRR) at which the investment breaks even."
2. "Determine the precise discount rate where the net present value (NPV) of an investment equals zero, indicating the project's break-even point in terms of profitability."
3. "For a project with an initial cost of $5,000 and expected annual cash inflows of $1,200 over five years, compute the discount rate at which the net present value (NPV) becomes zero."
------------------
**Example 2:**
**Example user question:** "Calculate the future value of an investment of $10,000 over 10 years with an annual interest rate of 5% compounded annually."
**Example 3 sentence variations:**
1. "Calculate how much my investment will be in the future: initial $10,000 over 10 years with an annual interest rate of 5% compounded annually."
2. "Determine how much an initial investment will grow over time given a specific interest rate and compounding frequency."
3. "I want to know how much my investment will be worth 7 years from now."
```
Before you start, understand this from a practical standpoint: the user question can be matched to a range of financial tools or solutions within the system, and your crafted variations should optimize for breadth and specificity.
Write out your approach and plan for tackling this, then provide the {self.n_items} sentences you would craft for the user question.
Think through your approach step by step, be intelligent, take a deep breath.
--------
"""

    def _get_human_message(self, user_question: str) -> str:
        return f"""USER QUESTION: {user_question}
YOUR APPROACH, REASONING, AND {self.n_items} SENTENCES:"""

    def _get_structured_message_after_first_response(self, ai_message: str, user_question: str):
        simplified_multi_query_expansion_prompt = """You are tasked to craft three nuanced sentence variations for user queries, each designed to target different keywords and approaches, with the aim of embedding these queries into a vector database to retrieve the most relevant tools across various industries.
USER QUESTION: `{user_question}`
YOUR APPROACH, REASONING, AND 3 SENTENCES:
"""
        ai_response_prompt = "{AI_response}"
        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(simplified_multi_query_expansion_prompt),
                AIMessagePromptTemplate.from_template(ai_response_prompt),
                HumanMessage(content=f"Extract the {self.n_items} questions and format them in the structured output.")
            ])
        return chat_template.format_messages(user_question=user_question, AI_response=ai_message)

    def generate(self, query: str) -> List[str]:
        messages = self._get_expansion_messages(user_question=query)
        ai_message = self.llm.invoke(messages)
        structured_message = self._get_structured_message_after_first_response(ai_message=ai_message, user_question=query)
        result = self.structured_llm.invoke(structured_message)
        return result.expanded_queries

    async def agenerate(self, query: str) -> List[str]:
        messages = self._get_expansion_messages(user_question=query)
        ai_message = await self.llm.ainvoke(messages)
        structured_message = self._get_structured_message_after_first_response(ai_message=ai_message, user_question=query)
        result = await self.structured_llm.ainvoke(structured_message)
        return result.expanded_queries

    def _get_expansion_messages(self, user_question: str):
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
