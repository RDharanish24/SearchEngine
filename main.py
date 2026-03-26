import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

# --- Environment Setup ---
load_dotenv()
BRIGHTDATA_API_KEY = os.getenv('BRIGHTDATA_API_KEY')
BRIGHTDATA_SERP_ZONE = os.getenv('BRIGHTDATA_SERP_ZONE')
BRIGHTDATA_GPT_DATASET_ID = os.getenv('BRIGHTDATA_GPT_DATASET_ID')
BRIGHTDATA_PERPLEXITY_DATASET_ID = os.getenv('BRIGHTDATA_PERPLEXITY_DATASET_ID')

HEADERS = {
    'Authorization': f'Bearer {BRIGHTDATA_API_KEY}',
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

# --- Tool Definitions ---
@tool(description="search using google")
def google_search(query):
    print(f"\n[Tool] Google search for: {query}")
    payload = {
        'zone': BRIGHTDATA_SERP_ZONE,
        'url': f'https://google.com/search?q={requests.utils.quote(query)}&brd_json=1',
        'format': 'raw',
        'country': 'US'
    }
    data = requests.post('https://api.brightdata.com/requests?async=true', headers=HEADERS, json=payload).json()
    results = [f"Title: {item.get('title')}\nLink: {item.get('link')}\nSnippet: {item.get('description', '')}" for item in data.get('organic', [])]
    return '\n\n'.join(results)[:10000]

@tool(description="search using bing")
def bing_search(query):
    print(f"\n[Tool] Bing search for: {query}")
    payload = {
        'zone': BRIGHTDATA_SERP_ZONE,
        'url': f'https://bing.com/search?q={requests.utils.quote(query)}&brd_json=1',
        'format': 'raw',
        'country': 'US'
    }
    data = requests.post('https://api.brightdata.com/requests?async=true', headers=HEADERS, json=payload).json()
    results = [f"Title: {item.get('title')}\nLink: {item.get('link')}\nSnippet: {item.get('description', '')}" for item in data.get('organic', [])]
    return '\n\n'.join(results)[:10000]

@tool(description="search using reddit")
def reddit_search(query):
    print(f"\n[Tool] Reddit search for: {query}")
    payload = {
        'zone': BRIGHTDATA_SERP_ZONE,
        'url': f'https://google.com/search?q={requests.utils.quote("site:reddit.com " + query)}&brd_json=1',
        'format': 'raw',
        'country': 'US'
    }
    data = requests.post('https://api.brightdata.com/requests?async=true', headers=HEADERS, json=payload).json()
    results = [f"Title: {item.get('title')}\nLink: {item.get('link')}\nSnippet: {item.get('description', '')}" for item in data.get('organic', [])]
    return '\n\n'.join(results)[:10000]

@tool(description="search using x")
def x_search(query):
    print(f"\n[Tool] X search for: {query}")
    payload = {
        'zone': BRIGHTDATA_SERP_ZONE,
        'url': f'https://google.com/search?q={requests.utils.quote("site:x.com " + query)}&brd_json=1',
        'format': 'raw',
        'country': 'US'
    }
    data = requests.post('https://api.brightdata.com/requests?async=true', headers=HEADERS, json=payload).json()
    results = [f"Title: {item.get('title')}\nLink: {item.get('link')}\nSnippet: {item.get('description', '')}" for item in data.get('organic', [])]
    return '\n\n'.join(results)[:10000]

@tool(description="use chatgpt to answer a question")
def gpt_prompt(query):
    print(f"\n[Tool] ChatGPT prompt: {query}")
    payload = [{"url": "https://chatgpt.com", "prompt": query}]
    url = f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={BRIGHTDATA_GPT_DATASET_ID}&format=json&custom_output_fields=answer_text_markdown"
    response = requests.post(url, headers=HEADERS, json=payload)
    snapshot_id = response.json().get('snapshot_id')

    if not snapshot_id:
        return "Failed to trigger ChatGPT dataset."

    while requests.get(f'https://api.brightdata.com/datasets/v3/progress/{snapshot_id}', headers=HEADERS).json()['status'] != 'ready':
        time.sleep(5)

    data = requests.get(f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json", headers=HEADERS).json()[0]
    return data.get('answer_text_markdown', 'No markdown output found.')

@tool(description="use perplexity to answer a question")
def perplexity_prompt(query):
    print(f"\n[Tool] Perplexity prompt: {query}")
    payload = [{"url": "https://www.perplexity.ai", "prompt": query}]
    url = f"https://api.brightdata.com/datasets/v3/trigger?dataset_id={BRIGHTDATA_PERPLEXITY_DATASET_ID}&format=json&custom_output_fields=answer_text_markdown|sources"
    response = requests.post(url, headers=HEADERS, json=payload)
    snapshot_id = response.json().get('snapshot_id')

    if not snapshot_id:
        return "Failed to trigger Perplexity dataset."

    while requests.get(f'https://api.brightdata.com/datasets/v3/progress/{snapshot_id}', headers=HEADERS).json()['status'] != 'ready':
        time.sleep(5)

    data = requests.get(f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json", headers=HEADERS).json()[0]
    answer = data.get('answer_text_markdown', 'No markdown output found.')
    sources = data.get('sources', [])
    return f"{answer}\n\nSources: {sources}"

# --- Agent Initialization (Cached) ---
@st.cache_resource
def get_agent():
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
    system_prompt = """
    You are a highly capable AI research assistant. 
    You have access to a variety of tools to gather information from search engines, social media (Reddit/X), and other AI models (ChatGPT/Perplexity).
    When asked a question, determine the best tool to find the information and synthesize a comprehensive response. 
    Always cite your sources if possible. Format your response cleanly using markdown.
    """
    tools = [google_search, bing_search, gpt_prompt, perplexity_prompt, reddit_search, x_search]
    return create_react_agent(model=llm, tools=tools, state_modifier=system_prompt)

agent = get_agent()

# --- Streamlit UI ---
st.set_page_config(page_title="Research Agent", page_icon="🕵️‍♂️", layout="centered")

st.title("🕵️‍♂️ Omni-Search Research Agent")
st.markdown("Powered by LangGraph, OpenAI, and Bright Data.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to research today?"):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # We use st.status to show a loading state while the agent runs its tools
        with st.status("Agent is researching...", expanded=True) as status:
            st.write("Invoking LangGraph ReAct agent...")
            
            # Prepare the input for the agent (passing the full conversation history)
            agent_input = {"messages": [("user", m["content"]) if m["role"] == "user" else ("assistant", m["content"]) for m in st.session_state.messages]}
            
            # Invoke the agent
            response = agent.invoke(agent_input)
            
            # The final AI message is the last one in the returned list
            final_message = response["messages"][-1].content
            
            status.update(label="Research complete!", state="complete", expanded=False)
            
        # Display the final synthesized answer
        st.markdown(final_message)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_message})