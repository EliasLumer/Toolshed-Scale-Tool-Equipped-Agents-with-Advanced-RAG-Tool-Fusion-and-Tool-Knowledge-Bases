from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from intra_retrieval.base_artf_module import BaseARTFModules
from pydantic import BaseModel, Field

class RerankerMultiQueryExpansionVariations(BaseARTFModules):
    def __init__(
        self,
        llm: ChatOpenAI,
        top_k: int,
        multi_query_expansion_variation_module: MultiQueryExpansionModule
    ):
        self.top_k = top_k
        self.multi_query_expansion_variation_module = multi_query_expansion_variation_module
        super().__init__(llm=llm)
        self._initialize_structured_llm()

    def _initialize_structured_llm(self):
        class FinalToolNames(BaseModel):
            tool_names: Annotated[
                List[str],
                Field(
                    description=f"The list of {self.top_k} exact final tool names after reranking.",
                    min_length=self.top_k,
                    max_length=self.top_k
                )
            ]
        self.structured_llm = self.llm.with_structured_output(FinalToolNames)

    def _format_documents(self, documents: List[Document]) -> str:
        formatted_list = []
        for doc in documents:
            tool_name = doc.metadata.get('tool_name', 'Unknown')
            tool_description = doc.page_content
            formatted_list.append(f"""-------\nTOOL NAME: {tool_name}\nTOOL DESCRIPTION & USEFUL DETAILS: {tool_description}""")
        return "\n".join(formatted_list)

    def _get_finalized_list_thoughts_messages(
        self,
        user_question: str,
        ai_response: str,
        user_question_results: List[Document],
        sentence_results: List[List[Document]]
    ):
        # Build the dynamic prompt based on the number of sentences
        sentences_section = ""
        for idx, sentence_result in enumerate(sentence_results, start=1):
            sentences_section += f"SENTENCE {idx} EMBEDDED AND RETRIEVED TOOLS:\n{self._format_documents(documents=sentence_result)}\n================\n"
        
        finalized_list_thoughts = f"""OK here are the results:
USER QUESTION EMBEDDED AND RETRIEVED TOOLS:
{self._format_documents(documents=user_question_results)}
{sentences_section}
=========
Based on these results, rank the top {self.top_k} most relevant tools to solve the user question. Just return the {self.top_k} TOOL NAMES for each relevant tool.
"""
        ai_response_prompt = "{ai_response}"

        # Get the sentence extraction prompt from the multi_query_expansion_variation_module
        sentence_extraction_messages = self.multi_query_expansion_variation_module._get_expansion_messages(user_question=user_question)

        chat_template = ChatPromptTemplate.from_messages(
            [
                *sentence_extraction_messages,
                AIMessagePromptTemplate.from_template(ai_response_prompt),
                HumanMessagePromptTemplate.from_template(finalized_list_thoughts)
            ]
        )

        messages = chat_template.format_messages(ai_response=ai_response)

        return messages

    def generate(
        self,
        user_question: str,
        ai_response: str,
        user_question_results: List[Document],
        sentence_results: List[List[Document]] # Ensure these the same order as the sentences
    ) -> List[Document]:
        # Step 1: Get the expansion messages from the multi_query_expansion_variation_module
        expansion_messages = self.multi_query_expansion_variation_module._get_expansion_messages(user_question=user_question)

        # Step 2: Prepare the final messages
        messages = self._get_finalized_list_thoughts_messages(
            user_question=user_question,
            ai_response=ai_response,
            user_question_results=user_question_results,
            sentence_results=sentence_results
        )

        # Step 3: Invoke the structured LLM to get the top_k tool names
        # Note: This is a simple way of handling errors. In production, more tests should be done to ensure the AI output tool name matches the python tool name.
        attempts = 0
        max_attempts = 3
        result = None

        while attempts < max_attempts:
            attempts += 1
            try:
                result = self.structured_llm.invoke(messages)
                if len(result.tool_names) == self.top_k:
                    break
            except Exception as e:
                pass  # Handle exceptions as needed

        if result and len(result.tool_names) == self.top_k:
            return result.tool_names
        else:
            # Fallback or error handling
            return []

    async def agenerate(
        self,
        user_question: str,
        ai_response: str,
        user_question_results: List[Document],
        sentence_results: List[List[Document]] # Ensure these the same order as the sentences
        ) -> List[Document]:
        # Step 1: Get the expansion messages from the multi_query_expansion_variation_module
        expansion_messages = self.multi_query_expansion_variation_module._get_expansion_messages(user_question=user_question)

        # Step 2: Prepare the final messages
        messages = self._get_finalized_list_thoughts_messages(
            user_question=user_question,
            ai_response=ai_response,
            user_question_results=user_question_results,
            sentence_results=sentence_results
        )

        # Step 3: Invoke the structured LLM to get the top_k tool names
        # Note: This is a simple way of handling errors. In production, more tests should be done to ensure the AI output tool name matches the python tool name.
        attempts = 0
        max_attempts = 3
        result = None

        while attempts < max_attempts:
            attempts += 1
            try:
                result = await self.structured_llm.ainvoke(messages)
                if len(result.tool_names) == self.top_k:
                    break
            except Exception as e:
                pass  # Handle exceptions as needed

        if result and len(result.tool_names) == self.top_k:
            return result.tool_names
        else:
            # Fallback or error handling
            return []
