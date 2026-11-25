import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pymupdf
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
import io
import numpy as np
import requests

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

# 2. Document Loading
pdf_path = os.path.join(
    os.path.dirname(__file__),
    "document_loader",
    "Industrial Visit Report.pdf",
)
if not os.path.isfile(pdf_path):
    raise FileNotFoundError(f"Report PDF not found: {pdf_path}")

print(f"Loading document from: {pdf_path}")

pdf_docs = PyPDFLoader(pdf_path).load()

# This report is image-based in places. OCR pages with little or no embedded text.
ocr_reader = RapidOCR()
docs = []
pdf = pymupdf.open(pdf_path)

for page_number, page in enumerate(pdf_docs, start=1):
    text = page.page_content.strip()

    if len(text) < 20:
        pdf_page = pdf[page_number - 1]
        pixmap = pdf_page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        result, _ = ocr_reader(np.asarray(image))
        if result:
            text = "\n".join(line[1] for line in result).strip()

    if text:
        docs.append(
            Document(
                page_content=text,
                metadata={**page.metadata, "page": page_number},
            )
        )

pdf.close()

if not docs:
    raise ValueError("No readable text was found in the report PDF.")

print(f"Loaded {len(docs)} readable document pages.")

# 3. Document Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} text chunks.")

# 4. Embeddings & In-Memory Vector Store (Qdrant + FastEmbed)
print("Creating embeddings and indexing into Qdrant vector store...")
embeddings = FastEmbedEmbeddings()

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="rag_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="rag_collection",
    embedding=embeddings,
)
vectorstore.add_documents(chunks)
print("Indexing complete!")

# 5. Initialize NVIDIA LLM
# Note: nemotron uses reasoning tokens, so max_completion_tokens should be >= 1024
llm = None
rag_chain = None
if api_key:
    llm = ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_api_key=api_key,
        temperature=0.2,
        max_completion_tokens=2500,
        timeout=30,
    )
else:
    print("NVIDIA_API_KEY is not configured. Retrieval will run in offline mode.")

# 6. Retriever & RAG Prompt
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an assistant for the Industrial Visit Report titled "
        "'Report on Food Preservation' by Neha Khatun. "
        "Answer accurately using ONLY the provided report context. "
        "For missing details, say that the information is not available in the report. "
        "Include the page number when it helps the reader verify the answer.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}"),
])

if llm is not None:
    rag_chain = prompt | llm | StrOutputParser()

# 7. Test RAG with a question
def query_rag(question: str):
    print("\n" + "="*60)
    print(f"Query: {question}")
    print("="*60)
    
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    print(f"\nRetrieved {len(retrieved_docs)} relevant context chunk(s):")
    for i, doc in enumerate(retrieved_docs, start=1):
        page_num = doc.metadata.get("page", "Unknown")
        preview = doc.page_content.strip().replace("\n", " ")[:150]
        print(f" [{i}] (Page {page_num}): {preview}...")

    # Build context string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Generate response
    if rag_chain is None:
        print("\nOffline mode: NVIDIA API is unavailable.")
        response = (
            "The NVIDIA API key is not configured, so an AI-generated answer "
            "cannot be produced. Retrieved report context:\n\n"
            f"{context}"
        )
    else:
        print("\nGenerating response from NVIDIA LLM...")
        try:
            response = rag_chain.invoke(
                {"context": context, "question": question}
            )
        except requests.exceptions.Timeout:
            response = (
                "The NVIDIA API timed out. Retrieved report context:\n\n"
                f"{context}"
            )
        except requests.exceptions.RequestException as exc:
            response = (
                f"The NVIDIA API request failed: {exc}\n\n"
                "Retrieved report context:\n\n"
                f"{context}"
            )
    print("\nAnswer:")
    print(response)
    print("="*60)
    return response

if __name__ == "__main__":
    test_question = "Who prepared the report, what is the report topic, and which course is listed?"
    query_rag(test_question)
