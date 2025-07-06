from typing import TypedDict, List, Dict, Sequence, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage,BaseMessage
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.tools import tool
from dotenv import load_dotenv, find_dotenv
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages # Without any over-writeing i appends the messages and for this it uses reducer function.


load_dotenv(find_dotenv())

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """This fucntion adds 2 numbers"""
    return a + b

tools = [add]

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.5
)

model = llm.bind_tools(tools)

# Creating a node which acutally access the agent in the graph

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
                                   "You are my AI assistant, please answer my query to the best of your abiliy."
                                   )

    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Conditional edge
def should_continue(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
# Defineing the graph
graph = StateGraph(AgentState)
graph.add_node("our_agent",model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

# Conditional edge
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

# Add edge which back to go tools to our_agent
graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user"," Add 10 + 10, Add 4527587845244747 + 89")]}
print_stream(app.stream(inputs, stream_mode="values"))
