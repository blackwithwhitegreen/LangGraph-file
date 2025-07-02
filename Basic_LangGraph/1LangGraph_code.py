from typing import TypedDict
from langgraph.graph import StateGraph
from IPython.display import Image, display

class AgentState(TypedDict): # Our State Schema
    message : str
    name : str


def greating_node(state: AgentState) -> AgentState:
    """
    Simple node that adds a greatting message to the state
    """
    # state["message"] = "Hey " + state["message"] + "How is your Day going?"
    state['message'] = "Hey " + f"{state['name']},you are doing great in LnagGraph" # Create personlized complement agent

    return state

graph = StateGraph(AgentState)

graph.add_node("greeter",greating_node)

# Defining the Starting point and End point.
graph.set_entry_point("greeter")
graph.set_finish_point("greeter")

app = graph.compile() 
result = app.invoke({"name": "Bob", "message": ""})


print(result["message"])
print(app.get_graph().draw_ascii())

