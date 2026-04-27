import os
import pandas as pd
import asyncio
import streamlit as st
import time
import uuid

from dotenv import load_dotenv

from typing import Any, Optional, Dict
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents.middleware import PIIMiddleware, AgentMiddleware, hook_config, AgentState
from langgraph.runtime import Runtime

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY Not found in .env")

client = ChatGoogleGenerativeAI(
    api_key=API_KEY, 
    model="gemini-3-flash-preview"
)

FAISS_DIR = "crop_faiss_index"
CSV_PATH = "archive/crop_yield.csv"

async def get_vectorstore():
    return await asyncio.to_thread(get_vectorstore_sync)

def get_vectorstore_sync():
    embeddings =  GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
    )

    if os.path.exists(FAISS_DIR):
        print("\nLoading FAISS Index...")
        return FAISS.load_local(
            FAISS_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    print("\nReading CSV...")
    documents = pd.read_csv(CSV_PATH, encoding="utf-8")

    print(f"Loaded {len(documents)} rows from CSV")

    texts = documents.astype(str).apply(
        lambda row: f"Crop: {row['Crop']} | Crop_Year: {row['Crop_Year']} | Season: {row['Season']} | State: {row['State']} | Area: {row['Area']} | Production: {row['Production']} | Annual_Rainfall: {row['Annual_Rainfall']} | Fertilizer: {row['Fertilizer']} | Pesticide: {row['Pesticide']} | Yield: {row['Yield']}",
        axis=1
    ).tolist()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text("\n\n".join(texts))

    print(f"Total chunks created: {len(chunks)}")

    print("Creating FAISS Index and embeddings...")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
   
    asyncio.run(save_FAISS(vectorstore))

    print("FAISS index created and saved...")

    return vectorstore

async def save_FAISS(vectorstore):
    await ayncio.to_thread(vectorstore.save_local, FAISS_DIR)

@tool(description="This function is used to search for user requested crops from the vector store embeddings.")
async def search_crops(question: str):
    vectorstore = await get_vectorstore()

    docs = await asyncio.to_thread(
        vectorstore.similarity_search,
        question,
        3
    )

    if not docs:
        return "No crops available for this query."

    return "\n\n".join(doc.page_content.strip() for doc in docs)

@tool(description="This tool is used to search for crop and agricultural fertilizers and pesticides ONLY for the crop as queried by the user. It fetches results from DuckDuckGo and processes the response.")
async def search_fertilizers_pesticides(question: str):
    
    try:
        search = DuckDuckGoSearchRun()

        result = await asyncio.to_thread(search.invoke, question)
        
        print("Fetching results from DuckDuckGo...")
        if isinstance(result, list) and result:
                return "\n".join([str(res) for res in result[:3]])  
            
        elif isinstance(result, str):
            return result

        return "Sorry, no relevant data was found for fertilizers and pesticides."
    
    except Exception as e:
        return f"An error occurred while searching: {e}"

class ClimateContextGuardrail(AgentMiddleware):
    """Adds context when inputs are outside Indian climate extremes"""

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state = AgentState, runtime = Runtime) -> Optional[Dict[str, Any]]:
        msg = state["messages"][0].content.lower()
        
        print("Running extreme trigger checker custom guardrail...")
        state_name = self.extract_state(msg)
        weather_condition = self.extract_weather(msg)

        if not state_name or not weather_condition:
            return None
        
        if self.is_weather_condition_realistic(state_name, weather_condition):
            return None
        else:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": f"The input conditions like '{weather_condition}' are not typically observed in {state_name}. Please provide a more realistic weather condition."
                }],
                "jump_to": "end"
            }
        
    def extract_state(self, msg: str) -> str:
        """Extract the state from the user's message."""
        state_keywords = ["delhi", "punjab", "kerala", "kashmir", "rajasthan", "maharashtra", "tamil nadu", "uttar pradesh", "bihar"]
        for state in state_keywords:
            if state in msg:
                return state
        return None

    def extract_weather(self, msg: str) -> str:
        """Extract the weather condition (snow, heatwave, etc.) from the message."""
        weather_keywords = ["snow", "heatwave", "flood", "drought", "rain", "storm"]
        for weather in weather_keywords:
            if weather in msg:
                return weather
        return None

    async def is_weather_condition_realistic(self, state_name: str, weather_condition: str) -> bool:
        """
        Use the agent's LLM to validate if the weather condition is realistic for the state.
        """
        try:
            prompt = f"Is it realistic for the state of {state_name} in India to have the weather condition {weather_condition}?"
            
            response = await llm.ainvoke({
                "messages": [
                    SystemMessage(content=prompt),
                    HumanMessage(content="Is this weather condition realistic for the state?")
                ]
            })

            answer = response["messages"][-1]["content"].strip().lower()
            if "yes" in answer or "realistic" in answer:
                return True
            else:
                return False
        except Exception as e:
            print(f"Error in weather validation: {e}")
            return False
    
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=API_KEY,
    temperature=0.2
)

agent = create_agent(
    model=llm,
    tools=[search_crops, search_fertilizers_pesticides],
    middleware=[ClimateContextGuardrail()],
    checkpointer=InMemorySaver()
)

st.set_page_config(
    page_title="AgroBot",
    page_icon="🌱",
    layout="centered"
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

st.title("🌾 AgroBot 🌾")
st.caption("Crop Recommendation Assistant")
st.markdown(
    f"<div style='font-size:10px; opacity:0.5; text-align:right;'>Thread ID: {st.session_state.thread_id}</div>",
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = agent
    
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
SYS_MSG = """
You are a crop recommendation assistant with access to a specific dataset of agricultural data (contained in a CSV file).
You cannot fetch information from the web or external databases.
You must **only** respond using the data in the provided CSV file and return recommendations based on the `search_crops` tool.
If a query doesn't match any relevant data, say: "No data is available in the database."
Do not generate new information or make assumptions outside of the dataset.
You can ONLY search online for information about fertilizers and pesticides using the search_fertilizers_pesticides tool.
Use only the information in the dataset to respond.
        """
        
async def run_agent(user_input: str):
    start = time.perf_counter()

    response = await st.session_state.agent.ainvoke(
        {
            "messages": [
                HumanMessage(user_input),
                SystemMessage(SYS_MSG)
            ]
        },
        config={"configurable": {"thread_id": st.session_state.thread_id}}
    )

    elapsed = time.perf_counter() - start
    return response["messages"][-1].content, elapsed

if prompt := st.chat_input("Ask about crops, yield, fertilizers…"):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        with st.spinner("Thinking 🌱..."):
            time.sleep(0.8)
            reply, latency = asyncio.run(run_agent(prompt))

        typed = ""
        for ch in reply:
            typed += ch
            placeholder.markdown(typed)
            time.sleep(0.01)

        st.caption(f"⏱ Response generated in {latency:.2f}s")

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )