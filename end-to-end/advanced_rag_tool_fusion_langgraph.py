from typing import TypedDict, Literal, Any, Optional, Annotated
import operator 
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

# define all the modules
# rewrite query
query_rewriter = LLMQueryRewritingModule(llm=llm)
# decompose query
query_decomposition_module = QueryDecompositionModule(llm=llm)
# multi query expansion or variation
multi_query_expansion_variation_module = MultiQueryExpansionModule(llm=llm, n_items=2)
# retrieve initial tools
initial_tool_retrieval_module = InitialToolRetrievalModule(embedder=embedder, toolshed_knowledge_base=faiss_indexer)
# rerank multi query expansion variations
individual_top_k = 5
reranker_multi_query_expansion_variations = RerankerMultiQueryExpansionVariations(llm=llm, top_k=individual_top_k, multi_query_expansion_variation_module=multi_query_expansion_variation_module)
# rerank decomposed queries
final_top_k = 5
reranker_query_decomposition = RerankerDecomposedQueries(llm=llm, final_top_k=final_top_k)

# Define state types
class ToolshedState(TypedDict):
    user_query: str
    conversation_history: List[str]
    rewritten_query: str
    decomposed_queries: List[str]
    list_of_intents: List[str]
    retrieved_tools: Annotated[List[List[str]], operator.add]
    decomposed_query_dicts: Annotated[List[Dict[str,Any]], operator.add]
    final_top_k_tools: List[str]

class DecomposedQueryState(TypedDict):
    decomposed_query: str
    expanded_queries: List[str]
    decomposed_query_tools: List[Any]
    expanded_query_dicts: Annotated[List[Dict[str, Any]], operator.add]
    decomposed_query_dicts: Annotated[List[Dict[str,Any]], operator.add]

class ExpandedQueryState(TypedDict):
    expanded_query: str
    retrieved_tools: List[List[str]]

# Define nodes
def rewrite_user_query(state: ToolshedState):
    rewritten_query = query_rewriter.generate(query=state["user_query"], conversation_history=state.get("conversation_history", []))
    return {"rewritten_query": rewritten_query}

def decompose_user_query(state: ToolshedState):
    decomposed_queries = query_decomposition_module.generate(query=state["rewritten_query"])
    return {"decomposed_queries": decomposed_queries}

def continue_to_process_decomposed_queries(state: ToolshedState):
    return [Send("process_decomposed_query", {"decomposed_query": dq}) for dq in state["decomposed_queries"]]

# Subgraph for processing each decomposed query
def process_decomposed_query_subgraph():
    subgraph = StateGraph(DecomposedQueryState)

    def expand_query(state: DecomposedQueryState):
        expanded_queries = multi_query_expansion_variation_module.generate(query=state["decomposed_query"])
        print(expanded_queries)
        return {"expanded_queries": expanded_queries}

    def continue_to_retrieve_tools_for_each_expanded_queries(state: DecomposedQueryState):
        print('gets to this state???')
        return [Send("retrieve_tools_for_each_expanded_query", {"expanded_query": eq}) for eq in state["expanded_queries"]]
    
    def retrieve_tools_for_query(state: DecomposedQueryState):
        retrieved_tools = initial_tool_retrieval_module.generate(query=state["decomposed_query"], top_k=individual_top_k)
        return {"decomposed_query_tools": retrieved_tools}

    def retrieve_tools_for_expanded_query(state: ExpandedQueryState):
        retrieved_tools = initial_tool_retrieval_module.generate(query=state["expanded_query"], top_k=individual_top_k)
        print('retrieves tools')
        return {"expanded_query_dicts": [{"expanded_query": state["expanded_query"], "retrieved_tools": retrieved_tools}]}
    
    def rerank_expanded_queries(state: DecomposedQueryState):
        print('rerank expanded queries')
        expanded_query_dicts = state["expanded_query_dicts"]
        expanded_queries_list = [eq["expanded_query"] for eq in expanded_query_dicts]
        expanded_queries_tools_list = [eq["retrieved_tools"] for eq in expanded_query_dicts]
        top_tools = reranker_multi_query_expansion_variations.generate(
            user_question=state["decomposed_query"],
            ai_response=expanded_queries_list,
            user_question_results=state["decomposed_query_tools"],
            sentence_results=expanded_queries_tools_list
        )
        return {"decomposed_query_dicts": [{"decomposed_query": state["decomposed_query"],"expanded_query_dicts": expanded_query_dicts,"decomposed_query_tools": state["decomposed_query_tools"], "final_top_k_tools": top_tools}]}

    # Build the subgraph
    subgraph.add_node("expand_query", expand_query)
    subgraph.add_node('retrieve_tools_for_decomposed_query', retrieve_tools_for_query)
    subgraph.add_node("retrieve_tools_for_each_expanded_query", retrieve_tools_for_expanded_query)
    subgraph.add_node("rerank_expanded_queries", rerank_expanded_queries)
    subgraph.add_edge(START, "expand_query")
    subgraph.add_edge("expand_query", "retrieve_tools_for_decomposed_query")
    subgraph.add_edge("retrieve_tools_for_decomposed_query", "rerank_expanded_queries")
    subgraph.add_conditional_edges("expand_query", continue_to_retrieve_tools_for_each_expanded_queries, ["retrieve_tools_for_each_expanded_query"])
    subgraph.add_edge("retrieve_tools_for_each_expanded_query", "rerank_expanded_queries")
    subgraph.add_edge("rerank_expanded_queries", END)
    return subgraph.compile()

def rerank_decomposed_queries(state: ToolshedState):
    decomposed_query_dicts = state["decomposed_query_dicts"]
    decomposed_queries_list = [dq["decomposed_query"] for dq in decomposed_query_dicts]
    # check if more than 1 decomposed query, if theres only 1, return the tools
    if len(decomposed_queries_list) == 1:
        return {"final_top_k_tools": decomposed_query_dicts[0]["final_top_k_tools"]}
    else:
        decomposed_queries_final_tools_list = [dq["final_top_k_tools"] for dq in decomposed_query_dicts]
        top_tools = reranker_query_decomposition.generate(
            user_question=state["rewritten_query"],
            list_of_intents=decomposed_queries_list,
            list_of_list_of_tools=decomposed_queries_final_tools_list
        )
        return {"final_top_k_tools": top_tools}

# Define the main workflow
workflow = StateGraph(ToolshedState)

# Add nodes
workflow.add_node("rewrite_user_query", rewrite_user_query)
workflow.add_node("decompose_user_query", decompose_user_query)
workflow.add_node("process_decomposed_query", process_decomposed_query_subgraph())
workflow.add_node("rerank_decomposed_queries", rerank_decomposed_queries)

# Define edges
workflow.add_edge(START, "rewrite_user_query")
workflow.add_edge("rewrite_user_query", "decompose_user_query")
workflow.add_conditional_edges("decompose_user_query", continue_to_process_decomposed_queries, ["process_decomposed_query"])
workflow.add_edge("process_decomposed_query", "rerank_decomposed_queries")
workflow.add_edge("rerank_decomposed_queries", END)

advanced_rag_tool_fusion = workflow.compile()
