import os
import time
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph,END

load_dotenv()
BRIGHTDATA_API_KEY=os.getenv('BRIGHTDATA_API_KEY')
BRIGHTDATA_SERP_ZONE=os.getenv('BRIGHTDATA_SERP_ZONE')
BRIGHTDATA_GPT_DATASET_ID=os.getenv('BRIGHTDATA_GPT_DATASET_ID')
BRIGHTDATA_PERPLEXITY_DATASET_ID=os.getenv('BRIGHTDATA_PERPLEXITY_DATASET_ID')


HEADERS={
    'Authorization':f'Bearer {BRIGHTDATA_API_KEY}',
    'Content-Type':'application/json',
    'Accept':'application/json'
}

@tool(description="search using google")
def google_search(query):
    print("Google tool is being used")

    payload={
        'zone':BRIGHTDATA_SERP_ZONE,
        'url':f'https://google.com/search?q={requests.utils.quote(query)}&brd_json=1',
        'format':'raw',
        'country':'US'
    }

    data=requests.post('https://api.brightdata.com/requests?async=true',headers=HEADERS,json=payload).json
    results=[]

    for item in data.get('organic'):
        results.append(f"Title:{item['title']}\nLink:{item['link']}\nSnippet:{item.get('description','')}")

    return '\n\n'.join(results)[:10000]

@tool(description="search using bing")
def bing_search(query):
    print("bing tool is being used")

    payload={
        'zone':BRIGHTDATA_SERP_ZONE,
        'url':f'https://bing.com/search?q={requests.utils.quote(query)}&brd_json=1',
        'format':'raw',
        'country':'US'
    }

    data=requests.post('https://api.brightdata.com/requests?async=true',headers=HEADERS,json=payload).json
    results=[]

    for item in data.get('organic'):
        results.append(f"Title:{item['title']}\nLink:{item['link']}\nSnippet:{item.get('description','')}")

    return '\n\n'.join(results)[:10000]

@tool(description="search using reddit")
def reddit_search(query):
    print("reddit tool is being used")

    payload={
        'zone':BRIGHTDATA_SERP_ZONE,
        'url':f'https://google.com/search?q={requests.utils.quote('site:reddit.com' + query)}&brd_json=1',
        'format':'raw',
        'country':'US'
    }

    data=requests.post('https://api.brightdata.com/requests?async=true',headers=HEADERS,json=payload).json
    results=[]

    for item in data.get('organic'):
        results.append(f"Title:{item['title']}\nLink:{item['link']}\nSnippet:{item.get('description','')}")

    return '\n\n'.join(results)[:10000]

@tool(description="search using x")
def x_search(query):
    print("x tool is being used")

    payload={
        'zone':BRIGHTDATA_SERP_ZONE,
        'url':f'https://google.com/search?q={requests.utils.quote('site:x.com' + query)}&brd_json=1',
        'format':'raw',
        'country':'US'
    }

    data=requests.post('https://api.brightdata.com/requests?async=true',headers=HEADERS,json=payload).json
    results=[]

    for item in data.get('organic'):
        results.append(f"Title:{item['title']}\nLink:{item['link']}\nSnippet:{item.get('description','')}")

    return '\n\n'.join(results)[:10000]

@tool(description="use chatgpt to answer a question")
def gpt_prompt(query):
    print("gpt tool is being used")
    payload = [
         {
             "url":"https://chatgpt.com"
             "prompt":query
         }
    ]
    url=f"https://api.Brightdata.com/datasets/v3/trigger?dataset_id={BRIGHTDATA_GPT_DATASET_ID}&format=json&custom_output_fields=answer_text_markdown"
    response=requests.post(url,headers=HEADERS,json=payload)
    snapshot_id=response.json()['snapshot_id']

    while requests.get('https://api.brightdata.com/datasets/v3/progress/{snapshot_id}',headers=HEADERS).json()['status']!='ready':
         time.sleep(5)

    data=requests.get(f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json",headers=HEADERS).json()[0]

    return data['answer_text_markdown']


@tool(description="use perplexity to answer a question")
def perplexity_prompt(query):
    print("perplexity  tool is being used")
    payload = [
         {
             "url":"https://www.perplexity.ai"
             "prompt":query
         }
    ]
    url=f"https://api.Brightdata.com/datasets/v3/trigger?dataset_id={BRIGHTDATA_PERPLEXITY_DATASET_ID}&format=json&custom_output_fields=answer_text_markdown|sources"
    response=requests.post(url,headers=HEADERS,json=payload)
    snapshot_id=response.json()['snapshot_id']

    while requests.get('https://api.brightdata.com/datasets/v3/progress/{snapshot_id}',headers=HEADERS).json()['status']!='ready':
         time.sleep(5)

    data=requests.get(f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}?format=json",headers=HEADERS).json()[0]

    return data['answer_text_markdown']+'\n\n' + str(data.get('sources',[]))

llm=ChatOpenAI(model_name='gpt-4o',temperature=0)
agent=create_react_agent(
    model=llm,
    tools=[google_search,bing_search,gpt_prompt,perplexity_prompt,reddit_search,x_search],
    debug=False,
    prompt=

)

