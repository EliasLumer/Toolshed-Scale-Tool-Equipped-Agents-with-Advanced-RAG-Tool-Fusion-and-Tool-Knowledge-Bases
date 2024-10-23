from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from end_to_end.advanced_rag_tool_fusion_langgraph import advanced_rag_tool_fusion

toolshed_dict = {tool.name: {"tool_object":tool} for tool in tool_list}

class AdvancedRAGToolFusionAgent(TypedDict):
    user_query: str
    conversation_history: List[str]
    retrieved_tool_name_from_toolshed: List[str]
    messages: Annotated[list, add_messages]

def retrieve_tools_from_toolshed(state: AdvancedRAGToolFusionAgent):
    result=advanced_rag_tool_fusion.invoke({"user_query": state["user_query"], "conversation_history": state.get("conversation_history", [])})
    return {"retrieved_tool_name_from_toolshed": result["final_top_k_tools"]}

def agent_node(state: AdvancedRAGToolFusionAgent):
    selected_tools = [toolshed_dict[tool_name]['tool_object'] for tool_name in state["retrieved_tool_name_from_toolshed"]]
    # Bind the selected tools to the LLM for the current interaction.
    llm_with_tools = llm.bind_tools(selected_tools)
    # Invoke the LLM with the current messages and return the updated message list.
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools=tool_list)

builder = StateGraph(AdvancedRAGToolFusionAgent)
builder.add_node("retrieve_tools_from_toolshed", retrieve_tools_from_toolshed)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_conditional_edges("agent", tools_condition, path_map=["tools", END])
builder.add_edge(START, "retrieve_tools_from_toolshed")
builder.add_edge("retrieve_tools_from_toolshed", "agent")

builder.add_edge("tools", "agent")

advanced_rag_tool_fusion_with_agent = builder.compile()
