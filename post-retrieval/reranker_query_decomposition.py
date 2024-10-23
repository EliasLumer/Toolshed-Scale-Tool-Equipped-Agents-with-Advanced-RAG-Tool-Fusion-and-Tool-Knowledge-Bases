class RerankerDecomposedQueries(BaseARTFModules):
    def __init__(self, llm: AzureChatOpenAI, final_top_k: int):
        self.final_top_k = final_top_k
        super().__init__(llm=llm)
        self._initialize_structured_llm()

    def _initialize_structured_llm(self):
        class FinalToolNames(BaseModel):
            tool_names: List[str] = Field(
                ...,
                description=f"The list of {self.final_top_k} exact final tool names after reranking.",
                min_length=self.final_top_k,
                max_length=self.final_top_k
            )
        self.structured_llm = self.llm.with_structured_output(FinalToolNames)

    def _get_final_combined_thoughts_messages(
        self,
        user_question: str,
        list_of_intents: List[str],
        list_of_list_of_tools: List[List[str]]
    ):
        num_intents = len(list_of_intents)
        top_k_per_intent = len(list_of_list_of_tools[0])
        num_tools_divisible_by_intent = max(1, self.final_top_k // num_intents)
        num_tools_divisible_by_intent_text = f"{num_tools_divisible_by_intent} tools" if num_tools_divisible_by_intent > 1 else f"{num_tools_divisible_by_intent} tool"

        final_combined_thoughts = f"""You are an expert at combining and narrowing down the top tools from each user intent to a single unique list of {self.final_top_k} tools that solve the user question.
You will be given a user query that has been broken down into {num_intents} distinct user intents.
You are also given the {top_k_per_intent} most relevant tools for each intent that can solve that particular intent, which ARE IN ORDER OF RELEVANCE!
Your task is to combine the top {top_k_per_intent} tools from each intent into a single unique list of {self.final_top_k} tools that are most relevant to the user question, which can solve the entire {num_intents}-step process.
Your first approach should be to take the top {num_tools_divisible_by_intent_text} from each intent and then add the next most relevant tool(s) from the intents until you have a unique list of {self.final_top_k} tools.
However, one important thing to note is that there may be overlap between the tools from each intent, this is because of our retrieval process.
If there are overlapping tools within each top {num_tools_divisible_by_intent_text}, first understand which top {num_tools_divisible_by_intent_text} are most relevant to their respective intents, and then go to the other intents and add the next most {num_tools_divisible_by_intent} relevant tools.

Here is an example with no overlapping tools (with 2 distinct user intents and 3 tools per intent, and a final top k of 3):
---------
USER QUESTION: 'I want to calculate the net present value (NPV) and internal rate of return (IRR) for a project with specific cash flows.'
INTENT 1: 'Calculate the net present value (NPV) for the project.'
LIST OF TOOLS FOR INTENT 1: ['get_net_present_value', 'get_present_value', 'get_future_value']
INTENT 2: 'Calculate the internal rate of return (IRR) for the project.'
LIST OF TOOLS FOR INTENT 2: ['get_internal_rate_of_return', 'get_modified_internal_rate_of_return', 'get_return_on_investment']
THE APPROACH TO TAKE:
First, identify the top 1 tool for intent 1 is 'get_net_present_value' and the top 1 tool for intent 2 is 'get_internal_rate_of_return'. Since there is no overlap, we can directly take these top 1 tool from each intent. So we can choose 1 more tool which is the most releavnt from list. To add up to 3 final tools, we select either 'get_present_value' or 'get_modified_internal_rate_of_return'. We select 'get_modified_internal_rate_of_return'. Now we have successfully built a unique list of 3 tools.
---------

Here is an example with overlapping tools (with 4 distinct user intents and 5 tools per intent, and a final top k of 10):
---------
USER QUESTION: 'I want to get the Return on Equity (ROE), Return on Assets (ROA), debt ratio, and dividend payout ratio of a company.'
INTENT 1: 'Calculate the Return on Equity (ROE) of the company.'
LIST OF TOOLS FOR INTENT 1: ['get_return_on_equity', 'get_net_profit_margin', 'get_operating_profit_margin', 'get_gross_profit_margin', 'get_return_on_investment']
INTENT 2: 'Calculate the Return on Assets (ROA) of the company.'
LIST OF TOOLS FOR INTENT 2: ['get_return_on_assets', 'get_return_on_equity', 'get_net_profit_margin', 'get_operating_profit_margin', 'get_gross_profit_margin']
INTENT 3: 'Calculate the debt ratio of the company.'
LIST OF TOOLS FOR INTENT 3: ['get_debt_ratio', 'get_debt_to_equity_ratio', 'get_total_debt', 'get_total_assets', 'get_current_ratio']
INTENT 4: 'Calculate the dividend payout ratio of the company.'
LIST OF TOOLS FOR INTENT 4: ['get_dividend_payout_ratio', 'get_retention_ratio', 'get_earnings_per_share', 'get_dividend_yield', 'get_price_earnings_ratio']
THE APPROACH TO TAKE:
First, identify the top 2 tools for Intent 1 is 'get_return_on_equity' and 'get_net_profit_margin'. The top 2 tools for Intent 2 are 'get_return_on_assets' and 'get_return_on_equity'. The top 2 tools for Intent 3 are 'get_debt_ratio' and 'get_debt_to_equity_ratio'. The top 2 tools for intent 4 are 'get_dividend_payout_ratio' and 'get_retention_ratio'. Since there is an overlap between Intent 1 and Intent 2 'get_return_on_equity' and 'get_return_on_assets', we need to understand which intent these 2 tools should be counted for. Considering Intent 1 is very much related to return on equity, we can count 'get_return_on_equity' there. Then we go to Intent 2 and add the next 1 tool which is 'get_net_profit_margin'. Since there is no more overlap, this gives us a unique list of 8 tools that are most relevant to the entire user question. So we can choose 2 more tools which is the most relevant from the list. The last two tools we will choose is 'get_total_debt' and 'get_dividend_yield', since they are relevant in the third slot to their respective intents.Now we have successfully built a unique list of 10 tools.

Final unique list of {self.final_top_k} tools: ['get_return_on_equity', 'get_return_on_assets', 'get_debt_ratio', 'get_dividend_payout_ratio', 'get_net_profit_margin'].
---------
YOUR TURN:
"""

        # Build the human prompt
        human_combiner_prompt = f"USER QUESTION: '{user_question}'\n"
        for idx, intent in enumerate(list_of_intents, start=1):
            human_combiner_prompt += f"INTENT {idx}: '{intent}'\n"
            human_combiner_prompt += f"LIST OF TOOLS FOR INTENT {idx}: {list_of_list_of_tools[idx-1]}\n"
        human_combiner_prompt += f"THE APPROACH TO TAKE AND {self.final_top_k} FINAL UNIQUE TOOLS:"

        # Create the chat template
        final_thoughts_chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(final_combined_thoughts),
                HumanMessagePromptTemplate.from_template(human_combiner_prompt)
            ]
        )

        # Format the messages
        final_thoughts_messages = final_thoughts_chat_template.format_messages()

        return final_thoughts_messages

    def _get_structured_prompt_after_first_response(
        self,
        ai_response: str
    ):
        human_structured = f"""Return ONLY the {self.final_top_k} final list of unique tools in a structured list format. As an example: ['tool_1', 'tool_2', 'tool_3', 'tool_4', 'tool_5']
Do not include any other information such as your reasoning or context.
Just the {self.final_top_k} final combined tools."""

        # Build the chat template with the AI response and the human instruction
        human_final_chat_template = ChatPromptTemplate.from_messages(
            [
                AIMessage(content=ai_response),
                HumanMessage(content=human_structured)
            ]
        )
        human_final_chat_message = human_final_chat_template.format_messages()
        return human_final_chat_message

    def generate(
        self,
        user_question: str,
        list_of_intents: List[str],
        list_of_list_of_tools: List[List[str]]
    ) -> List[str]:
        attempts = 0
        max_attempts = 3
        result = None

        while attempts < max_attempts:
            attempts += 1
            try:
                # Step 1: Prepare the initial messages
                messages = self._get_final_combined_thoughts_messages(
                    user_question=user_question,
                    list_of_intents=list_of_intents,
                    list_of_list_of_tools=list_of_list_of_tools
                )

                # Step 2: Use the regular LLM to get the AI response
                ai_response_content = (self.llm.invoke(messages)).content
                #print(ai_response_content)
                # Step 3: Prepare the structured prompt with the AI response
                structured_messages = self._get_structured_prompt_after_first_response(
                    ai_response=ai_response_content
                )

                # Step 4: Use the structured LLM to parse the AI response and get the structured output
                result = self.structured_llm.invoke(structured_messages)
                if len(list(set(result.tool_names))) == self.final_top_k:
                    break
            except Exception as e:
                continue  # Handle exceptions as needed

        if result and len(list(set(result.tool_names))) == self.final_top_k:
            return result.tool_names
        else:
            # Fallback or error handling
            return []

    async def agenerate(
        self,
        user_question: str,
        list_of_intents: List[str],
        list_of_list_of_tools: List[List[str]]
    ) -> List[str]:
        attempts = 0
        max_attempts = 3
        result = None

        while attempts < max_attempts:
            attempts += 1
            try:
                # Step 1: Prepare the initial messages
                messages = self._get_final_combined_thoughts_messages(
                    user_question=user_question,
                    list_of_intents=list_of_intents,
                    list_of_list_of_tools=list_of_list_of_tools
                )

                # Step 2: Use the regular LLM to get the AI response
                ai_response_content = (await self.llm.ainvoke(messages)).content
                #print(ai_response_content)
                # Step 3: Prepare the structured prompt with the AI response
                structured_messages = self._get_structured_prompt_after_first_response(
                    ai_response=ai_response_content
                )

                # Step 4: Use the structured LLM to parse the AI response and get the structured output
                result = await self.structured_llm.ainvoke(structured_messages)
                if len(list(set(result.tool_names))) == self.final_top_k:
                    break
            except Exception as e:
                continue  # Handle exceptions as needed

        if result and len(list(set(result.tool_names))) == self.final_top_k:
            return result.tool_names
        else:
            # Fallback or error handling
            return []
