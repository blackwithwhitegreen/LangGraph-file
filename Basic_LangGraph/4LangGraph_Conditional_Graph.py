from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    number1: int
    operation: str
    number2: int
    finalNumber: int

# Creating first node function
def adder(state: AgentState) -> AgentState:
    """This node adds the 2 numbers"""
    state['finalNumber'] = state['number1'] + state['number2']

    return state

def subtractor(state: AgentState) -> AgentState:
    """This node Subtract the 2 numbers"""
    state['finalNumber'] = state['number1'] - state['number2']
    return state

def decide_next_node(state: AgentState) -> AgentState:
    """This node will select the next node of the graph"""
    if state['operation'] == "+":
        return "addition_operation"
    
    elif state['operation'] == "-":
        return "subtraction_operation"
    


# creating graph
graph = StateGraph(AgentState)

# Adding nodes
graph.add_node("add_node",adder)
graph.add_node("subtract_node", subtractor)
graph.add_node("router", lambda state: state) # passthrough Function


graph.add_edge(START,"router")

# Conditonal Edge
graph.add_conditional_edges(
    "router", # Source 
    decide_next_node, # Path
    # Path map is initilizing in the dictionary
    {
        # Edge : Node
        "addition_operation" : "add_node",
        "subtraction_operation": "subtract_node"
        # Starting point of this node is router
    }
)

# Edges for END
graph.add_edge("add_node", END)
graph.add_edge("subtract_node", END)

app = graph.compile()

result = AgentState(number1= 3, operation= "-", number2= 5)

print(app.invoke(result))
print(app.get_graph().draw_ascii())
