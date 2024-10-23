from langchain_core.prompts import ChatPromptTemplate,HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage

class BaseToolDocumentEnhancementGenerator(ABC):
    def __init__(self, llm: ChatOpenAI, n_items: int):
        self.llm = llm
        self.n_items = n_items
        self.structured_llm = self._initialize_structured_llm()

    @abstractmethod
    def _initialize_structured_llm(self):
        """Define the structured LLM with the appropriate output model."""
        pass

    @abstractmethod
    def _get_system_message(self, **kwargs):
        """Construct the system message for the prompt."""
        pass

    @abstractmethod
    def _get_human_message(self, **kwargs):
        """Construct the human message for the prompt."""
        pass

    def _get_messages(self, **kwargs):
        system_message = self._get_system_message(**kwargs)
        human_message = self._get_human_message(**kwargs)
        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_message),
                HumanMessagePromptTemplate.from_template(human_message)
            ]
        )
        messages = chat_template.format_messages()
        return messages

    
    @abstractmethod
    def generate(self, **kwargs):
        """Abstract method for generating output."""
        pass

    @abstractmethod
    async def agenerate(self, **kwargs):
        """Abstract method for generating output asynchronously."""
        pass

class QuestionGenerator(BaseToolDocumentEnhancementGenerator):
    def __init__(self, llm: ChatOpenAI, n_items: int):
        super().__init__(llm, n_items)

    def _initialize_structured_llm(self):
        class GeneratedQuestions(BaseModel):
            generated_questions: List[str] = Field(
                description=f"{self.n_items} example questions that the function can answer."
            )
        return self.llm.with_structured_output(GeneratedQuestions)

    def _get_system_message(self, **kwargs):
        return f"""You are an expert at generating hypothetical questions that a given Python function can answer.
You are given the name of a function, its description, and its arguments.
Your job is to do the following:
1. Generate {self.n_items} example questions that the function can answer.

The example questions should have roughly 50% questions that contain all of the arguments and read similar to a text-book type of question (What is the present value of $20,000 to be received in 10 years if the discount rate is 7%?).
While 50% other questions are more abstract and focus on the function's utility and purpose -- and include no function arguments (ex. "I want to know the discount rate of a project with breakeven net present value").
Overall, these two types of questions should cover the diverse usage of the questions.

Here are some examples:

---

**Function Name:** get_present_value

**Description:**
Calculates the present value of a future amount using the formula: PV = FV / (1 + r)^t. This function helps determine the current worth of future cash flows, considering the time value of money. Useful for assessing investment opportunities, loans, or any financial decision involving future payments.

**Arguments:**
- future_value: float, "Future amount to be received or paid."
- discount_rate: float, "Discount rate (as a decimal)."
- periods: int, "Number of periods until payment."

**Example Questions:**
- What is the present value of $20,000 to be received in 10 years if the discount rate is 7%?
- How much should I invest today to have $50,000 in 5 years at an annual discount rate of 6%?
- I want to find out the current value of a future sum I expect to receive.
- If I need $100,000 in 15 years, how much do I need to invest now assuming an 8% annual return?
- How does the discount rate affect the present value of my future cash inflow?

---

**Function Name:** get_internal_rate_of_return

**Description:**
Calculates the Internal Rate of Return (IRR), which is the discount rate that makes the net present value (NPV) of cash flows equal to zero. This function helps evaluate the profitability of potential investments or projects. Useful when comparing multiple investment opportunities to determine which yields the highest return.

**Arguments:**
- cash_flows: List[float], "Sequence of cash flows starting with initial investment (negative value)."

**Example Questions:**
- What is the IRR for an investment with an initial outlay of $50,000 and annual returns of $15,000 for 5 years?
- How can I find the discount rate at which my project's net present value breaks even?
- I want to evaluate the profitability of my investment over several years.
- If my project has varying annual cash flows, how do I determine its internal rate of return?
- At what rate does the net present value of my cash flows become zero?

---

**Function Name:** get_future_value

**Description:**
Calculates the future value of an investment using the compound interest formula: FV = PV * (1 + r/n)^(n*t). This function helps investors determine how much their current investment will grow over time, given a specific interest rate and compounding frequency. Useful for planning long-term investments, retirement funds, or savings growth.

**Arguments:**
- present_value: float, "Initial amount invested."
- interest_rate: float, "Annual interest rate (as a decimal)."
- periods: int, "Number of periods (years)."
- compounding_frequency: int, "Times interest is compounded per period."

**Example Questions:**
- What will be the future value of a $10,000 investment after 5 years at an annual interest rate of 5% compounded quarterly?
- How much will my savings grow to in 10 years with a 3% annual interest rate compounded monthly?
- I want to know how much my investment will be worth in the future considering compound interest.
- If I invest $5,000 today, what will it amount to in 20 years at an annual rate of 7% compounded annually?
- How does changing the compounding frequency affect the future value of my investment?

---

**Function Name:** get_loan_payment

**Description:**
Calculates the periodic loan payment required to amortize a loan using the formula for an amortizing loan. This function helps borrowers understand their regular payment obligations. Useful for planning mortgages, car loans, or any installment-based debt repayment.

**Arguments:**
- principal: float, "Total loan amount borrowed."
- annual_interest_rate: float, "Annual interest rate (as a decimal)."
- periods: int, "Total number of payment periods."

**Example Questions:**
- What is the monthly payment on a $30,000 car loan with a 5% annual interest rate over 5 years?
- How much will I need to pay each month to pay off my mortgage loan?
- If I borrow $200,000 for a home at a 4% annual interest rate for 30 years, what are my monthly payments?
- I want to find out my regular payment amounts for a personal loan I'm considering.
- How does the interest rate impact the size of my loan payments?

---

Take a deep breath and think through the problem step by step."""

    def _get_human_message(self, tool_name: str, tool_description: str, tool_arguments: str):
        return f"""FUNCTION NAME: `{tool_name}`
FUNCTION DESCRIPTION: `{tool_description}`
FUNCTION ARGUMENTS: `{tool_arguments}`

{self.n_items} GENERATED QUESTIONS THAT THE FUNCTION CAN ANSWER:"""

    def generate(self, tool_name: str, tool_description: str, tool_arguments: str) -> List[str]:
        messages = self._get_messages(
            tool_name=tool_name, 
            tool_description=tool_description, 
            tool_arguments=tool_arguments
        )
        result = self.structured_llm.invoke(messages)
        return result.generated_questions

    async def agenerate(self, tool_name: str, tool_description: str, tool_arguments: str) -> List[str]:
        messages = self._get_messages(
            tool_name=tool_name, 
            tool_description=tool_description, 
            tool_arguments=tool_arguments
        )
        result = await self.structured_llm.ainvoke(messages)  # Using async invoke
        return result.generated_questions

class KeyTopicGenerator(BaseToolDocumentEnhancementGenerator):
    def __init__(self, llm: ChatOpenAI, n_items: int):
        super().__init__(llm, n_items)

    def _initialize_structured_llm(self):
        class KeyTopics(BaseModel):
            key_topics: List[str] = Field(
                description=f"{self.n_items} key topics, themes, or intents that capture the overarching theme and topic of the questions."
            )
        return self.llm.with_structured_output(KeyTopics)

    

    def _get_system_message(self, **kwargs) -> str:
        return f"""You are an expert at generating key topics, themes, or intents from a list of questions that a Python function can answer.
You are given the name of a function, its description, along with various example questions that can be answered by using the function.
Your job is to do the following:
1. Generate a list of {self.n_items} key topics, each 1-5 words long, that capture the overarching theme and topic of the following questions.

See below the following examples

Example 1:
-----------
FUNCTION NAME: `get_present_value`
FUNCTION DESCRIPTION: Calculates the present value of a future amount using the formula: PV = FV / (1 + r)^t. This function helps determine the current worth of future cash flows, considering the time value of money.
EXAMPLE QUESTIONS:
- What is the present value of $20,000 to be received in 10 years if the discount rate is 7%?
- How much should I invest today to reach a certain financial goal in the future?
- I want to find out the current value of a future sum I expect to receive.
- If I need $100,000 in 15 years, how much do I need to invest now assuming an 8% annual return?
- How does the discount rate affect the present value of my future cash inflow?
KEY TOPICS: ['Present Value Calculation', 'Time Value of Money', 'Discounted Cash Flows', 'Investment Valuation', 'Future Cash Worth']
-----------

Example 2:
-----------
FUNCTION NAME: `get_internal_rate_of_return`
FUNCTION DESCRIPTION: Calculates the Internal Rate of Return (IRR), which is the discount rate that makes the net present value (NPV) of cash flows equal to zero. This function helps evaluate the profitability of potential investments or projects.
EXAMPLE QUESTIONS:
- What is the IRR for an investment with an initial outlay of $50,000 and annual returns of $15,000 for 5 years?
- How can I find the discount rate at which my project's net present value breaks even?
- I want to evaluate the profitability of my investment over several years.
- If my project has varying annual cash flows, how do I determine its internal rate of return?
- At what rate does the net present value of my cash flows become zero?
KEY TOPICS: ['Internal Rate of Return', 'Investment Profitability', 'Discount Rate Calculation', 'NPV Break-Even Point', 'Cash Flow Analysis']
-----------

Example 3:
-----------
FUNCTION NAME: `get_future_value`
FUNCTION DESCRIPTION: Calculates the future value of an investment using the compound interest formula: FV = PV * (1 + r/n)^(n*t). This function helps investors determine how much their current investment will grow over time, given a specific interest rate and compounding frequency.
EXAMPLE QUESTIONS:
- What will be the future value of a $10,000 investment after 5 years at an annual interest rate of 5% compounded quarterly?
- How much will my savings grow to in 10 years with a 3% annual interest rate compounded monthly?
- I want to know how much my investment will be worth in the future considering compound interest.
- If I invest $5,000 today, what will it amount to in 20 years at an annual rate of 7% compounded annually?
- How does changing the compounding frequency affect the future value of my investment?
KEY TOPICS: ['Future Value Projection', 'Compound Interest', 'Investment Growth Over Time', 'Savings Accumulation', 'Compounding Frequency Impact']
-----------

Example 4:
-----------
FUNCTION NAME: `get_loan_payment`
FUNCTION DESCRIPTION: Calculates the periodic loan payment required to amortize a loan using the formula for an amortizing loan. This function helps borrowers understand their regular payment obligations.
EXAMPLE QUESTIONS:
- What is the monthly payment on a $30,000 car loan with a 5% annual interest rate over 5 years?
- How much will I need to pay each month to pay off my mortgage loan?
- If I borrow $200,000 for a home at a 4% annual interest rate for 30 years, what are my monthly payments?
- I want to find out my regular payment amounts for a personal loan I'm considering.
- How does the interest rate impact the size of my loan payments?
KEY TOPICS: ['Loan Payment Calculation', 'Amortization Schedule', 'Monthly Payment Determination', 'Debt Repayment Planning', 'Interest Rate Impact on Loans']
-----------

Take a deep breath and think through the problem step by step."""

    def _get_human_message(self, tool_name: str, tool_description: str, example_questions: List[str]):
        return f"""FUNCTION NAME: `{tool_name}`
FUNCTION DESCRIPTION: `{tool_description}`
EXAMPLE QUESTIONS: {example_questions}
{self.n_items} KEY TOPICS THAT CAPTURE THE OVERARCHING THEME AND TOPIC:"""

    def generate(self, tool_name: str, tool_description: str, example_questions: List[str]) -> List[str]:
        messages = self._get_messages(
            tool_name=tool_name,
            tool_description=tool_description,
            example_questions=example_questions
        )
        result = self.structured_llm.invoke(messages)
        return result.key_topics

    async def agenerate(self, tool_name: str, tool_description: str, example_questions: List[str]) -> List[str]:
        messages = self._get_messages(
            tool_name=tool_name,
            tool_description=tool_description,
            example_questions=example_questions
        )
        result = await self.structured_llm.ainvoke(messages)  
        return result.key_topics
