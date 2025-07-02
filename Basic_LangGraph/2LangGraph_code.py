from typing import TypedDict, List
from langgraph.graph import StateGraph

# Defineing State Schema first
class AgentState(TypedDict):
    values : List[int]
    name : str
    result: str
    operation : str

def process_value(state: AgentState) -> AgentState:
    """
    This function process handles multiple different inputs
    """
    # this if condition is used only for the new feature, but if we want to do the simple we can use the simple state 
    if state['operation'] == "*":
        result = 1
        for v in state['values']:
            result *= v
    elif state['operation'] == "+":
        result = sum(state['values'])
    else:
        result =0
    state['result']  = f"Hi {state['name']}! your result is {result}" 
    # state['result'] = f"Hi there {state['name']}! Your sum {sum(state['values'])}"
    return state                                                         


# Create a Graph
graph = StateGraph(AgentState)

# Adding Node
graph.add_node('processor',process_value )

# Defining Starting point and ending point
graph.set_entry_point("processor") # Set the starting node.
graph.set_finish_point("processor") # Set the ending node.


app = graph.compile() #Compiling the graph

answer = app.invoke({"values": [1,2,3,4], 'name':"Root", "operation": "*"})
print(answer)

print(app.get_graph().draw_ascii())