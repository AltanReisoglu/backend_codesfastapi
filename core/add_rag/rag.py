import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pymongo import MongoClient

# Senin LLM importun
from langchain.chat_models import init_chat_model
SESSION_ID="abc123"
OKUL_ADI="YILDIZ TEKNİK ÜNİVERSİTESİ"
# ================================
# ENV Yükleme
# ================================
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")  # atlas ya da local bağlantı
DB_NAME = os.getenv("DB_NAME", "rag")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ytu")
INDEX_NAME = os.getenv("INDEX_NAME", "ytu_search")

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# ================================
# MongoDB Yükleme
# ================================
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

"""vectorstore = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding_model,
    index_name=INDEX_NAME,
)"""

#retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def get_vectorstore_for_school(school_name: str):
    """
    Her okul için ayrı koleksiyon/index seçilebilir.
    """
    collection = client[DB_NAME][COLLECTION_NAME]  # İstersen school_name'e göre ayrı collection seç
    index_name = f"{school_name.replace(' ', '_').lower()}_index"  # örn: bogazici_universitesi_index

    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_model,
        index_name="ytu_search"
    )

# ================================
# TOOL (dinamik okul seçimi)
# ================================
@tool
def retriever_tool(query: str,history=None) -> str:
    """
    Kullanıcı hangi okul hakkında bilgi istiyorsa,
    o okulun vektör indeksinden döküman çeker.
    """
    
    vectorstore = get_vectorstore_for_school(school_name=OKUL_ADI)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    docs = retriever.invoke(query)
    if not docs:
        return f"No relevant information found for {OKUL_ADI}."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"[{OKUL_ADI}] Document {i+1}:\n{doc.page_content}")
    return "\n\n".join(results)
# ================================
# TOOL HEPSİ AYNI COLLECTION ALTINDA İSE BU.İKİNCİ SENARYO OKULARI ARYI COLLECTIONLARDA TUTUP GET_VECTOR_STREU' ÖZELLEŞTİREBİLİRİZ.
# ================================
"""@tool
def retriever_tool(query: str) -> str:
    '''
    This tool retrieves relevant documents about school information from vectorstore based on the user's query , if you don't know answer.    
    '''
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    return "\n\n".join(results)"""


tools = [retriever_tool]
tools_dict = {t.name: t for t in tools}


# ================================
# AGENT SETUP
# ================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


system_prompt = f""" Sen bir üniversite öğrencilerine yönelik akıllı asistansın.

- Öğrencilerle sohbet edebilir, onların ders, ödev, kariyer, kampüs yaşamı ve kişisel gelişimle ilgili sorularına yanıt verirsin.
- Cevaplarında arkadaşça, anlaşılır ve motive edici bir dil kullanırsın.
- Konu hakkında yeterli bilgin varsa kendi bilginle yanıt verirsin.
- Eğer bilgilerin güncel değilse veya kesinlik gerektiriyorsa, **retriever_tool** kullanarak dış kaynaktan araştırma yapar ve en doğru cevabı verirsin.
- **retriever_tool**, yalnızca **{OKUL_ADI}** ile ilgili **akademik** konularında kullanılmalıdır.
- Gerektiğinde öğrenciyi düşünmeye teşvik eder, alternatif bakış açıları sunar ve pratik öneriler verirsin ama konuşmalarını kısa tutmaya çalışırsın.
- Yanıtların kısa, öz ama açıklayıcı olur; öğrenciyi boğmadan bilgi sağlarsın.

"""


def call_llm(state: AgentState) -> AgentState:
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = model.invoke(messages)
    return {"messages": [message]}


def take_action(state: AgentState) -> AgentState:
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        if t["name"] not in tools_dict:
            result = "Incorrect Tool Name"
        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
        results.append(
            ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
        )
    return {"messages": results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
memory = MemorySaver()
rag_agent = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}


def get_response_model(query,history=None):
    user_msg = HumanMessage(content=query) 
    result = rag_agent.invoke({"messages": [user_msg]}, config) 
    as1=result['messages'][-1] 
    if hasattr(as1,"content"): 
        return as1.content 
    else: 
        return as1

