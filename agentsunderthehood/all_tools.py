from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10
# MODEL = "ollama:gwen3:1.7b"
MODEL = "gemini-2.5-flash"


@tool(description="Lookup the price of a product in the catalog")
def get_product_price(product: str) -> float:
    print(f"Getting price of {product}... ")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@tool(description="Apply a discount tier to a price and return the final price.")
def apply_discount(price: float, discount_tier: str) -> float:
    print(f"Applying discount tier {discount_tier} to price: {price}")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


@traceable(name="Langchain Agent Loop")
def run_agent(query: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    # llm = init_chat_model(model=MODEL, temperature=0, timeout=30)
    llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0, timeout=30)
    llm_with_tools = llm.bind_tools(tools)

    print(f"query: {query}")
    messages = [
        SystemMessage(content=(
            """
            You are helpful shopping assistant.
            STRICT RULES - you must follow these exactly:
            1. Never guess or assume any product price. You MUST CALL necessary agentsunderthehood to get the real price and discount
            2. Never calculate discounts yourself using math. Always use the binded agentsunderthehood.
            3. If user does not specify a discount tier, ask them which tier to use - do NOT assume one.
            """
        )),
        HumanMessage(content=query)
    ]

    for iteration in range(1, MAX_ITERATIONS):
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\n Final answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0] # could return multiple agentsunderthehood
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError("No agentsunderthehood found.")
        response = tool_to_use.invoke(tool_args)
        messages.append(ai_message)
        messages.append(ToolMessage(content=str(response), tool_call_id = tool_call_id))
        print(f"Tool Response  -> {response}")

    print("ERROR: Max iterations reached without a final answer")
    return response


if __name__ == "__main__":
    print("Hello Langchain Agent")
    result = run_agent("What is the price of a laptop after applying a gold discount?")
