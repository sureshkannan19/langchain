import ollama
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langsmith import traceable

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"
# MODEL = "gemini-2.5-flash"


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    print(f"Getting price of {product}... ")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    print(f"Applying discount tier {discount_tier} to price: {price}")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as agentsunderthehood (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


@traceable(name="Ollama Raw Agent Loop")
def run_agent(query: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }
    print(f"query: {query}")
    messages = [
        {
            "role": "system",
            "content": """"
         You are helpful shopping assistant.
         STRICT RULES - you must follow these exactly:
         1. Never guess or assume any product price. You MUST CALL necessary agentsunderthehood to get the real price and discount
         2. Never calculate discounts yourself using math. Always use the binded agentsunderthehood.
         3. If user does not specify a discount tier, ask them which tier to use - do NOT assume one.
         """
        }, {
            "role": "user",
            "content": query
        }
    ]

    for iteration in range(1, MAX_ITERATIONS):
        response = ollama_chat(messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\n Final answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]  # could return multiple agentsunderthehood
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"[Tool Selected] {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError("No agentsunderthehood found.")
        response = tool_to_use(**tool_args)

        print(f"Tool Response  -> {response}")

        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(response)})


    print("ERROR: Max iterations reached without a final answer")
    return response


if __name__ == "__main__":
    print("Hello Langchain Agent")
    result = run_agent("What is the price of a laptop after applying a gold discount?")
