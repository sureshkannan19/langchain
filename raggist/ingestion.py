import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader(
        "C:\\Users\\SureshBabu\\PycharmProjects\\langchain\\raggist\\vijay_cm.txt",
        encoding="utf-8")
    documents: list[Document] = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Created {len(texts)} chunks")

    # change output_dimensionality = 1536 as desired
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")

    import time

    for i, text in enumerate(texts):
        try:
            PineconeVectorStore.from_documents(
                [text],
                embeddings,
                index_name=os.environ["INDEX_NAME"]
            )
            print(f"Inserted chunk {i + 1}")
            time.sleep(0.5)  # avoid rate limiting
        except Exception as e:
            print(f"Failed chunk {i + 1}: {e}")



