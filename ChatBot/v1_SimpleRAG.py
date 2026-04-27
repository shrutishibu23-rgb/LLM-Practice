import os
import pandas as pd
import httpx

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY Not found in .env")

client = httpx.Client(verify=False)

FAISS_DIR = "faiss_index"
CSV_PATH = "archive/data.csv"

def get_vectorstore():
    embeddings = OpenAIEmbeddings(
        base_url="https://genailab.tcs.in", # set openai_api_base to the LiteLLMProxy
        model = "azure/genailab-maas-text-embedding-3-large",
        openai_api_key=API_KEY,
        http_client = client
    )

    if os.path.exists(FAISS_DIR):
        print("\nLoading FAISS Index...")
        return FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization = True
        )
    
    print("\nReadins CSV...")
    documents = pd.read_csv(CSV_PATH, encoding = "utf-8")

    print(f"Loaded {len(documents)} rows from CSV")

    texts = documents.astype(str).apply(
        lambda row: " | ".join(row.values),
        axis=1
    ).tolist()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_text("\n\n".join(texts))

    print(f"Total chunks created: {len(chunks)}")

    print("Creating FAISS Index and embeddings...")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local(FAISS_DIR)

    print("FAISS index created and saved...")

    return vectorstore

def get_conversational_chain():
    llm = ChatOpenAI(
    base_url="https://genailab.tcs.in", # set openai_api_base to the LiteLLMProxy
    model = "azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key = API_KEY,
    http_client = client, 
    temperature=0.5
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are an assistant answering questions using ONLY the context below.
If the context is insufficient, say so and summarize what IS available.

Context:
{context}

Question:
{question}

Answer:
"""
    )
    
    chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm 
        | StrOutputParser()
    )

    return chain 

def answer_question(question: str):
    vectorstore=get_vectorstore()

    print("\nRunning similarity search...")
    docs = vectorstore.similarity_search(question, k=3)

    context = "\n\n".join(doc.page_content for doc in docs)

    chain = get_conversational_chain()

    print("\nGenerating final answer...\n")
    response=chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    print("\n" + "="*70)
    print("\nFinal answer: ")
    print(response)
    print("="*70)

if __name__=="__main__":
    print("RAG BOT(Print exit to quit)\n")

    while True:
        user_q = input("Question: ").strip()
        if user_q.lower()=="exit":
            print("Bye!")
            break
        if user_q:
            answer_question(user_q)