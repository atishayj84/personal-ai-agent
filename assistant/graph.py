from os import getenv
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
      openai_api_key=getenv("OPENROUTER_API_KEY"),
      openai_api_base=getenv("OPENROUTER_BASE_URL"),
      model="microsoft/mai-ds-r1:free", temperature=0)  

from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import operator
from typing import  Annotated
from langgraph.graph import MessagesState

from langchain_community.tools.tavily_search import TavilySearchResults


from langchain_community.document_loaders import WikipediaLoader
from langgraph.graph import MessagesState

from langgraph.types import Send

class Analyst(BaseModel):
    """Analyst persona for the graph"""
    name: str = Field(default="Analyst", description="Name of the analyst")
    expertise: str

class ResearchState(MessagesState):
    topic: str
    Analysts: List[Analyst]
    summary: Annotated[List[str], operator.add]
    report: str

class SearchState(MessagesState):
    """State of the graph"""
    context: Annotated[list, operator.add]
    persona: Analyst
    query: str
    summary: List[str]
    search_results: str

class SearchQuery(BaseModel):
    search_query: str

#####

generate_query_prompt = '''
You are an analyst tasked with generating a report on the topic specified by the user based on your persona:
{persona}
Generate a search query that will help you find relevant information for your report.
Make sure to consider the areas of expertise and tools available to you.
'''

tavily_search = TavilySearchResults(max_results=3)
def generate_query(state:SearchState) -> str:
    prompt = generate_query_prompt.format(
        persona=state['persona'])
    structured_llm = llm.with_structured_output(SearchQuery)
    response = structured_llm.invoke([SystemMessage(content=prompt)]+state['messages'])
    return {'query': response.search_query}

def web_search(state: SearchState):
    """Perform a web search based on the generated query."""
    query = state['query']
    # Simulate a web search result
    search_result = tavily_search.invoke(query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_result
        ]
    )
    return {'context': [formatted_search_docs]}

def wikipedia_search(state: SearchState):
    query = state['query']
    search_docs = WikipediaLoader(query=query, 
                                  load_max_docs=2).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {'context': [formatted_search_docs]}


summarize_search_prompt = '''
You are an analyst tasked with generating a report on the topic based on your persona:
{persona}
You have access to the following search results:
{context}
Generate a long form report summarizing the search results.
Do not generate new information, but rather synthesize the summaries provided.
Make sure to include key insights, trends, and any relevant data points.
Cite all the sources you used in your report.
'''
def summarize_search_results(state: SearchState):
    prompt = summarize_search_prompt.format(
        persona=state['persona'],
        context=state['context']
    )
    response = llm.invoke([SystemMessage(content=prompt)]+state['context'])
    return {'summary': [response.content]}



search_graph = StateGraph(SearchState)
search_graph.add_node('generate_query', generate_query)
search_graph.add_node('web_search', web_search)
search_graph.add_node('wikipedia_search', wikipedia_search)    
search_graph.add_node('summarize_search_results', summarize_search_results)
search_graph.add_edge(START, 'generate_query')
search_graph.add_edge('generate_query', 'web_search')
search_graph.add_edge('generate_query', 'wikipedia_search')   
search_graph.add_edge('web_search', 'summarize_search_results')
search_graph.add_edge('wikipedia_search', 'summarize_search_results')
search_graph.add_edge('summarize_search_results', END)





def initiate_search(state: ResearchState):
    topic = state['topic']
    return [Send('search_graph', {'messages': [HumanMessage(content=f"The topic to research for is {topic}")], 
                                  'persona': state['Analysts'][i]
                                  }) for i in range(len(state['Analysts']))]

generate_report_prompt = '''
You are a helpful agent that generates a comprehensive answer based on the summaries provided by the two analysts experts in- 
{analysts}
Summaries:
{summary}
Generate a comprehensive report that includes key insights from both the insights, and any relevant data points.
Do not generate new information, but rather synthesize the summaries provided.
Make sure to cite all the sources used in the report.
'''
def generate_report(state: ResearchState):
    """Generate a report based on the summary."""
    summary = "\n\n".join(state['summary'])
    analysts = state['Analysts']

    prompt = generate_report_prompt.format(
        analysts=", ".join([analyst.expertise for analyst in analysts]),
        summary=summary
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    return {'report': response.content}

research_graph = StateGraph(ResearchState)
research_graph.add_node('generate_report', generate_report)
research_graph.add_node('search_graph', search_graph.compile())

research_graph.add_conditional_edges(START, initiate_search, ['search_graph'])
research_graph.add_edge('search_graph', 'generate_report')
research_graph.add_edge('generate_report', END)

researcher = research_graph.compile()

# Teacher = Analyst(name='Teacher', expertise = 'A top level expert in this field who specialises in research and the theory of the topic')
# Practitioner = Analyst(name='Practitioner', expertise = 'A practical expert in this field who has hands on experience and can provide insights based on real world applications')

# topic = input("Enter the topic for research: ")
# response = researcher.invoke({'topic': topic, 'Analysts': [Teacher, Practitioner]})

# print(response.get('report', 'No report generated.'))