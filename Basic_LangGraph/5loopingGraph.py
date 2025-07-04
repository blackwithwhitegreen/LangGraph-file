from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, START, END
import random
from IPython.display import display
import graphviz


class AgentState(TypedDict):
    name: str
    number: List[int]
    counter: int


def greeting_node(state: AgentState) -> AgentState:
    """Greeting Node which says hi to the person"""
    state["name"] = f"Hi there, {state['name']}"
    state["counter"] = 0
    return state


def random_node(state: AgentState) -> AgentState:
    """This generate a random number from 0 to 10"""
    state["number"].append(random.randint(0, 10))
    state["counter"] += 1
    return state


def should_contine(state: AgentState) -> AgentState:
    """Functionto decide what to do next"""

    if state['counter'] < 5:
        print("ENTERING LOOP", state["counter"])
        return "loop"  # Continue looping
    else:
        return "exit"  # Exit the loop


# Creating Grpah
graph = StateGraph(AgentState)

# Creating Nodes
graph.add_node("greeting", greeting_node)
graph.add_node("random", random_node)

# Creating edge betwee greeting node and random node
graph.add_edge("greeting", "random")

# Creating Conditional edges
graph.add_conditional_edges(
    "random",  # Source node
    should_contine,  # Action
    {
        "loop": "random",  # Self-loop back to same node
        "exit": END  # End the graph
    }
)

# Entry point
graph.set_entry_point("greeting")
app = graph.compile()

result = app.invoke({"name": "rohit","number":[], "counter":-1})

print(result)