import streamlit as st
import asyncio
import nest_asyncio
# nest_asyncio.apply()

from assistant.graph import researcher, Analyst
from langchain_core.messages import HumanMessage

# Streamlit page settings
st.set_page_config(page_title="LangGraph Live Viewer", layout="wide")
st.title(" Assistant Research Graph")

Teacher = Analyst(
    name='Teacher',
    expertise='A top level expert in this field who specialises in research and the theory of the topic'
)
Practitioner = Analyst(
    name='Practitioner',
    expertise='A practical expert in this field who has hands on experience and can provide insights based on real world applications'
)

user_input = st.text_input("Topic you want to research!")

# async def run_graph(output_area):
#     config = {"configurable": {"thread_id": "1"}}
#     events_log = ""

#     async for event in researcher.astream_events(
#         {
#             "topic": user_input,
#             "Analysts": [Teacher, Practitioner]
#         },
#         config,
#         version="v2"
#     ):
#         node = event['metadata'].get('langgraph_node', 'Unknown')
#         event_type = event['event']
#         name = event.get('name', '')
#         events_log += f"**Node:** {node} | **Type:** {event_type} | **Name:** {name}\n\n"
#         output_area.markdown(events_log)  # Update UI live

# Button to run graph
if st.button("Run Graph"):
    output_area = st.empty()
    response = researcher.invoke(
        {
            'topic': user_input,
            'Analysts': [Teacher, Practitioner]
        }
    )
    report = response.get('report', 'No report generated.')
    output_area.markdown(f"### Generated Report:\n\n{report}")
    # asyncio.get_event_loop().run_until_complete(run_graph(output_area))