from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name : str
    age : str
    final : str
    skills : str

def first_node(state: AgentState) -> AgentState:
    """
    This is the first node of our Sequence
    """

    state['final'] = f"Hi {state['name']}"
    return state

def second_node(state: AgentState) -> AgentState:
    """
    This is the second node of our Sequence
    """

    state['final'] = state['final'] +  f" you are {state['age']} years old."
    return state

def third_node(state: AgentState) -> AgentState:
    """
    This is the third node of our Sequence
    """
    state['final'] = state['final'] + f" You have skills in {state['skills']}!"
    return state

graph = StateGraph(AgentState)

graph.add_node("first_node", first_node)
graph.add_node("second_node",second_node)
graph.add_node("third_node",third_node)

graph.set_entry_point("first_node")

# For connecting First node and second node we use edge.
graph.add_edge("first_node","second_node")
graph.set_finish_point("second_node")


# graph.set_entry_point("second_node")

# For connecting second node and third node we use edge.
graph.add_edge("second_node","third_node")
graph.set_finish_point("third_node")


app = graph.compile()

result = app.invoke({"name":"rohit", "age": 30, "skills":"Machine Learning and LangGraph"})
print(result)

print(app.get_graph().draw_ascii())

 