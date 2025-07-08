from typing import TypedDict, Sequence, Annotated, Dict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage,ToolMessage
from langchain_community.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langgraph.graph import add_messages
from langgraph.prebuilt import ToolNode
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from vertexai.preview.language_models import TextEmbeddingModel
import vertexai
from langchain_huggingface import HuggingFaceEmbeddings
import os
import shutil

load_dotenv(find_dotenv())

# llm = ChatGoogleGenerativeAI(
#     model="models/gemini-1.5-flash-latest",
#     temperature=0.5
# )
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.6,
    # max_tokens=64,
    # api_key=groq_api_key
)

embeddings = HuggingFaceEmbeddings(
    model = "BAAI/bge-small-en-v1.5",
)


pdf_path = "C:\\Users\\aman singh\\Desktop\\langGrpah\\Deepseek-R1.pdf"


# Safety measure I have put for debugging purposes
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF File not foundL {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

# Checks if the PDF is there
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise


# Chunking process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)

# persist_directory =  r"C:/Users/aman singh/OneDrive/Documents/Prodigal ai(Gen AI)/Deepseek-R1.pdf"
persist_directory =  r"C:\Users\aman singh\Desktop\langGrpah\Deepseek_R1_chroma_db"
collection_name = "Deepseek-R1"

# If our collection does not exist in the directory, we create using the os command
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


try:
    # Here we actually create the chroma database using our embeddings model
    if os.path.exists(persist_directory):
      shutil.rmtree(persist_directory)
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name = collection_name

    )
    
    print(f"Created ChromaDB vector store!")

except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

# Now we create our retriever
retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":5} # k is the amount of chunks to return
)

@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the information form the provided document."""

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in provided document. "
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}: \n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt ="""
You are an intelligent and reliable AI assistant, tasked with helping users explore and understand the content of an uploaded document. 
Your primary responsibility is to provide accurate, structured, and context-aware responses based strictly on the document's content.

Responsibilities:
- Clearly explain concepts, definitions, and ideas presented in the document using simple language and analogies when helpful.
- Provide relevant examples, code snippets (especially in Python), or step-by-step explanations when the document covers technical or practical material.
- Do not generate or assume any information that is not explicitly present in the document — avoid speculation and hallucination.
- If a user question is not answered or supported by the document, clearly state that and offer to assist with what is available.
- Prioritize clarity, factual accuracy, and helpfulness over creativity or guesswork.

Your goal is to serve as a grounded guide. Always tie your responses directly to the document’s content and cite the source material or page when appropriate. 
Be humble in uncertainty and always maintain the user's trust through transparency and reliability.
"""

# Creating a dictionary of our tools
tools_dict = {our_tool.name: our_tool for our_tool in tools}

# LLm Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    messages = llm.invoke(messages)
    return {"messages": [messages]}


# Retriever Agent 
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response."""
    
    tool_calls = state["messages"][-1].tool_calls
    results =[]
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if not t['name'] in tools_dict: # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query',''))
            print(f"Result length: {len(str(result))}")

        
        # Appends the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content = str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {"messages": results}


graph = StateGraph(AgentState)

graph.add_node("llm", call_llm)
graph.add_node("retriever_agent",take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        True:"retriever_agent", False:END
    }

)
graph.add_edge("retriever_agent","llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

# this function is only for printing
def running_agent():
    print("\n ==== RAG ====")

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit','quit']:
            break
        
        messages = [HumanMessage(content=user_input)] # Convert back to a HumanMessage type

        result = rag_agent.invoke({"messages": messages})

        print("\n ==== ANSWER ====")
        print(result['messages'][-1].content)


running_agent()