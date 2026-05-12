import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")
MODEL = "gemini-2.5-flash"

llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0, timeout=30)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
prompt_template = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
 {context}

 Question: {question}

 Provide a detailed answer:
""")


def format_docs(documents: list[Document]):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in documents)


def retrieve_chain_without_lcel():
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    docs = retriever.invoke(question)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    print(f"Raw RAG Response: {response}")


def retrieve_chain_with_lcel():
    retrieval_chain = (
            RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs)
            | prompt_template | llm | StrOutputParser()
    )
    return retrieval_chain

def llm_without_rag():
    result = llm.invoke([HumanMessage(content=question)])
    print(f"Answer {result}")


if __name__ == "__main__":
    ## Raw LLm without RAG
    question: str = "Did vijay become CM?"
    # llm_without_rag()
    # retrieve_chain_without_lcel()
    response = retrieve_chain_with_lcel().invoke({"question": question})
    print(response)