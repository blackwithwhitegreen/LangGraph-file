import gradio as gr
from typing import TypedDict, Sequence, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_community.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langgraph.graph import add_messages
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os, shutil, tempfile

load_dotenv(find_dotenv())

# LLM Setup
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.6,
)

# Embeddings
embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")

retriever = None  # Global retriever object

# Tool (dynamic)
@tool
def retriever_tool(query: str) -> str:
    """Searches and returns the information from the uploaded document."""
    if retriever is None:
        return "Please upload a document first."
    docs = retriever.invoke(query)
    if not docs:
        return "I found no relevant information in the document."
    return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

# Tool binding
tools = [retriever_tool]
tools_dict = {tool_.name: tool_ for tool_ in tools}
llm = llm.bind_tools(tools)

# Agent Setup
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
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

def call_llm(state: AgentState) -> AgentState:
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(messages)
    return {"messages": [response]}

def take_action(state: AgentState) -> AgentState:
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        if t['name'] not in tools_dict:
            result = "Incorrect Tool Name. Please retry."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    return {"messages": results}

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()

# PDF Processing

def process_pdf(file):
    global retriever
    if not file:
        return "No PDF uploaded."

    loader = PyPDFLoader(file.name)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages_split = splitter.split_documents(pages)

    persist_directory = tempfile.mkdtemp()

    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="dynamic_doc"
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return f"PDF loaded successfully. {len(pages_split)} chunks indexed."

# Gradio Chat Logic
def gradio_chat(user_input, history):
    messages = [HumanMessage(content=user_input)]
    result = rag_agent.invoke({"messages": messages})
    reply = result['messages'][-1].content
    history.append((user_input, reply))
    return history, history

# Gradio UI
with gr.Blocks(css=".scroll-area { overflow-y: auto; max-height: 85vh; }") as demo:
    gr.Markdown("# RAG Chat Assistant (Upload Your PDF)")

    with gr.Row():
        with gr.Column(scale=1, elem_classes="scroll-area"):
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_status = gr.Textbox(label="Status", interactive=False)
            upload_button = gr.Button("Process PDF")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot()
            user_msg = gr.Textbox(label="Ask a question", placeholder="What is reward modeling?")
            submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")

    state = gr.State([])

    upload_button.click(fn=process_pdf, inputs=[file_input], outputs=[upload_status])
    submit_btn.click(fn=gradio_chat, inputs=[user_msg, state], outputs=[chatbot, state])
    clear_btn.click(lambda: ([], []), outputs=[chatbot, state])

if __name__ == "__main__":
    demo.launch()