from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv(find_dotenv())

class AgentState(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]

# If the Google Gemini model is not working due to out of couta, we can use Groq model.
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=1
)

def process(state: AgentState) -> AgentState:
    """This node is used for the provideing solutions as per the inputs."""

    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))

    print(f"\nAI: {response.content}")
    print("CURRENT STATE:", state["messages"])
    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})
    # print(result["messages"])
    conversation_history = result["messages"]

    user_input = input("Enter: ")

with open("logging.txt", "w")as file:
    file.write("Your Conversation Log:\n")

    for message in  conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")



# This type of function can be made for checking and trim the chat history
# # Trim old HumanMessages if more than 10
#     human_msgs = [m for m in conversation_history if isinstance(m, HumanMessage)]
#     if len(human_msgs) > 10:
#         for i, m in enumerate(conversation_history):
#             if isinstance(m, HumanMessage):
#                 del conversation_history[i]
#                 break
