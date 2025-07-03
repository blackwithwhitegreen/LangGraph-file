from langgraph.graph import StateGraph, START,END
from typing import TypedDict

class AgentState(TypedDict):
    number1: int
    operation1: str
    number2: int
    final_number1 : int
    number3: int
    operation2: str
    number4: int
    final_number2: int

# Creating Node
def adder1(state: AgentState) -> AgentState:
    """This node add 2 numbers"""
    state['final_number1'] = state['number1'] + state['number2']
    return state

def adder2(state: AgentState) -> AgentState:
    """This node add 2 numbers"""
    state['final_number2'] = state['number3'] + state['number4']
    return state

def subtract1(state: AgentState) -> AgentState:
    """This node subtract 2 number"""
    state['final_number1'] = state['number1'] - state['number2']
    return state

def subtract2(state: AgentState) -> AgentState:
    """This node subtract 2 number"""
    state['final_number2'] = state['number3'] - state['number4']
    return state

# def decide_next_node(state: AgentState) -> AgentState:
#     if state['operation1'] == "+":
#         return "addition_operation"
#     elif state['operation1'] == "-":
#         return "subtract_operation"
    
# def decide_next_node2(state: AgentState) -> AgentState:
#     if state['operation2'] == "+":
#         return "addition_operation"
#     elif state['operation2'] == "-":
#         return "subtract_operation"

def decide_next_node(state: AgentState) -> AgentState:
    if state['operation1'] == "+" or state['operation2'] == "+":
        return "addition_operation"
    elif state['operation1'] == "-" or state['operation2'] == "-":
        return "subtract_operation"

graph = StateGraph(AgentState)

# Create Node
graph.add_node("add_node",adder1)
graph.add_node("subtract_node", subtract1)
graph.add_node("router", lambda state: state)# Passthrough function

graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "addition_operation": "add_node",
        "subtract_operation": "subtract_node"   
    }
)


graph.add_node("add_node1",adder2)
graph.add_node("subtract_node1", subtract2)
graph.add_node("router2", lambda state: state) #Passthrough function

graph.add_conditional_edges(
    "router2",
    decide_next_node,
    {
        "addition_operation": "add_node1",
        "subtract_operation": "subtract_node1"
    }
)

# linking with router2
graph.add_edge("add_node","router2")
graph.add_edge("subtract_node","router2")

graph.add_edge("add_node1",END)
graph.add_edge("subtract_node1",END)

app = graph.compile()

result = AgentState(number1=10, operation1= "-", number2=5, number3=7,  number4=2,operation2="+")

print(app.invoke(result))

print(app.get_graph().draw_ascii())