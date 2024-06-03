from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(temperature=0.8,
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-70b-8192"
        )


from langchain.agents import AgentType, Tool, initialize_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt.tool_node import ToolNode
from langchain_community.utilities import GoogleSerperAPIWrapper


# def should_continue(messages):
#     last_message = messages[-1]
#     # If there is no function call, then we finish
#     if not last_message.tool_calls:
#         return END
#     else:
#         return "action"


# # Define a new graph
# workflow = MessageGraph()
# search_serper_api_wrapper = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))
# tools =[Tool(name="Search", func=search_serper_api_wrapper.run, description="google search")]
# model=model.bind_tools(tools)
# workflow.add_node("agent", model)
# workflow.add_node("action", ToolNode(tools))

# workflow.set_entry_point("agent")

# # Conditional agent -> action OR agent -> END
# workflow.add_conditional_edges(
#     "agent",
#     should_continue,
# )

# # Always transition `action` -> `agent`
# workflow.add_edge("action", "agent")

# memory = SqliteSaver.from_conn_string(":memory:") # Here we only save in-memory

# # Setting the interrupt means that any time an action is called, the machine will stop
# app = workflow.compile(checkpointer=memory, interrupt_before=["action"])

# thread = {"configurable": {"thread_id": "4"}}

# a=app.invoke("台灣有什麼好吃的", thread)
# print(a)

search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

self_ask_with_search = initialize_agent(
    tools, model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
a=self_ask_with_search.invoke(
    """台北松山區有什麼富含蔬菜美食?請用JSON格式回答。
    請依需求判斷是否需要使用搜尋功能，若需要搜尋功能，請推估要使用的關鍵字,
    例如:要找尋富含蔬菜的美食就使用沙拉來搜尋,
    店名：阿喜魯肉飯 地點:google地圖連結 評價:google，就回傳{"title":"阿喜魯肉飯",map:"google map link",rating:google上的評分}"""
)
print(a)