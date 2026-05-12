from dotenv import load_dotenv
from google import genai
from langsmith import wrappers
import google.generativeai as genai

load_dotenv()
def main():
    # genai.Client() reads GOOGLE_API_KEY / GEMINI_API_KEY from the environment
    gemini_client = genai.Client()

    # Wrap the Gemini client to enable LangSmith tracing
    client = wrappers.wrap_gemini(
        gemini_client,
        tracing_extra={
            "tags": ["gemini", "python"],
            "metadata": {
                "integration": "google-genai",
            },
        },
    )

    # Make a traced Gemini call
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Explain quantum computing in simple terms.",
    )

    print(response.text)

def list_models():
    import os

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    for model in genai.list_models():
        if "embed" in model.name:
            print(model.name)

if __name__ == "__main__":
    # main()
    list_models()