import chainlit as cl
from chainlit.input_widget import Select

import os
from dataclasses import dataclass
from typing import Any

import logfire
from elastic_transport import ObjectApiResponse
from elasticsearch import Elasticsearch
from jira import JIRA
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, BinaryContent
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
import pymsteams

load_dotenv()
logfire.configure()

############## Base model initialization ##############
client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
)
model = OpenAIModel(os.getenv('AZURE_OPENAI_MODEL_NAME'), openai_client=client)


############## Dependency initialization ##############
@dataclass
class JiraDeps:
    client: JIRA
    project_key: str
    images = None

@dataclass
class MSTeamsDeps:
    url: str

@dataclass
class ElasticDeps:
    client: Elasticsearch
    index_name: str

@dataclass
class MainDependencies:
    es_deps: ElasticDeps
    jira_deps: JiraDeps
    ms_teams_deps: MSTeamsDeps

############## Jira Agent ##############

system_prompt = """
    You are a atlassian jira expert with access to Jira to help the user manage the story,task and issue creation and get information from it.
    
    Wherever we response the jira item and it can be can be shown with hyperlink for easy navigation,base_url(https://trackspace.lhsystems.com/browse/) + issue_key.

    Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

    Don't ask the user before taking an action, just do it. Always make sure you look at the index with the provided tools before answering the user's question unless you have already.

    When answering a question about the query, always start your answer with the full details in brackets and then give your answer on a newline. Like:

    [Using [query from the user]]

    Your answer here...
    
    After the answer, you can also provide the suggestions to the user if you have any.
    [Suggestions: [suggestion goes here]]
    """


class JiraResponse(BaseModel):
    issue_key: str
    summary: str = None
    description: str = None
    comments: list[str] = None
    suggestions: list[str] = None


jira_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=JiraDeps,
    retries=0
)

@jira_agent.tool
async def get_jira_issues(ctx: RunContext[MainDependencies], search_keyword: str) -> list[JiraResponse]:
    """
    Search for Jira issues with the specified keyword and summarize each issue with max 10 sentence description.
    Every issue should contain the issue key with hyperlink and description.
    :param ctx: RunContext object containing the dependencies.
    :param search_keyword:  Keyword to search for in the Jira issues.
    :return: List of JiraResponse objects containing the issue key and description.
    """
    print(f"Searching for Jira issues with keyword: {search_keyword}")
    logfire.info(f"Input: Searching for Jira issues with keyword: {search_keyword}")
    if search_keyword.__contains__(','):
        search_keyword = search_keyword.split(',')
        search_keyword = ' OR '.join([f'text ~"{keyword}"' for keyword in search_keyword])
    logfire.info(f"Searching for Jira issues with keyword: {search_keyword}")
    response = ctx.deps.jira_deps.client.search_issues(
        f"project={ctx.deps.jira_deps.project_key} AND text ~'{search_keyword}' ORDER BY updatedDate DESC, createdDate ASC",
        maxResults=10)

    # Print the retrieved issues
    jira_issues = []
    for issue in response:
        jira_issues.append(JiraResponse(
            issue_key=issue.key,
            summary=issue.fields.summary,
            description=issue.fields.description,
            comments=[comment.body for comment in issue.fields.comment.comments]
        ))
    return jira_issues


@jira_agent.tool
async def get_jira_issue(ctx: RunContext[MainDependencies], key: str) -> list[JiraResponse]:
    """
    Search for Jira issue with the specified key and the response should contain the below format.
    - Issue key: Issue key with the hyperlink.
    - Summary: Summary of the issue with maximum 10 sentence.
    - Description: Detailed description including key points.
    - External system involved: External system involved in the issue.
    - Recommended action: Recommended action to be taken.
    - Similar issues: Similar issues found in the past.
    :param ctx: RunContext object containing the dependencies.
    :param key: Key to search for in the Jira issues.
    :return: List of JiraResponse objects containing the issue key and description.
    """
    logfire.info(f"Searching for Jira issue with key: {key}")
    response = ctx.deps.jira_deps.client.search_issues(f"project={ctx.deps.jira_deps.project_key} AND key={key}", maxResults=10)

    # Print the retrieved issues
    jira_issues = []
    for issue in response:
        jira_issues.append(JiraResponse(
            issue_key=issue.key,
            summary=issue.fields.summary,
            description=issue.fields.description,
            comments=[comment.body for comment in issue.fields.comment.comments]
        ))
    return jira_issues


@jira_agent.tool
async def create_jira_issue(ctx: RunContext[MainDependencies], summary: str, description: str,issue_type:str,conversation_id:str,json:str) -> JiraResponse:
    """
    Create a new Jira issue.
    :param ctx: RunContext object containing the dependencies.
    :param summary: Summary of the issue to be created.
    :param description: Description should contain the below list of points and each point should be described with 3 to 5 bullet points.
    Each point should follow the Jira format , and the description should follow the jira style with proper format.
     * Issue description: Issue details with some important points.
     * Error observed: Error details.
     * Technical details: Possible technical details observed from the logs.
     * Steps to reproduce: Possible steps to reproduce the issue.
     * Recommended action: Json data of the issue to be created.
    :param issue_type: Type of the issue to be created.It should be either Bug,Task,Story. If there is an issue and create a Bug type issue.
    :param conversation_id: Conversation ID of the issue to be created.
    :param json: Json data of the issue to be created.Json data should be always formatted properly.
    :return: JiraResponse object containing the issue key and description.
    """

    image_name = os.path.basename(ctx.deps.jira_deps.images[0].path) if ctx.deps.jira_deps.images else None

    issue_description = f"""
    
    {description}
    * conversationId: {conversation_id}

    {{code}}
    {json}
    {{code}}
    
    {'* Image: !'+str(image_name)+'!' if image_name else ''}

    """

    # Define the search query
    issue_dict = {
        'project': {'key': 'AIPOC'},
        'summary': summary,
        'description': issue_description,
        'issuetype': {'name': issue_type},
    }

    # Perform the search query on the specified index
    # response = ctx.deps.client.create_issue(fields=issue_dict)
    ctx.deps.jira_deps.client = JIRA(server='https://manoharant.atlassian.net',
                           basic_auth=("manoharant@gmail.com","ATATT3xFfGF0z2tETfqzBNomWPdxMACIXRMlkMd4RjEKUtOdeypHun-ch4m28CgucDI5LJrdX3jpJg66nRpXx10lG8PCNYgQVM4kU63po59ZoRFsfVgazqZc0C3LR1bB_jnrepQOrKHt7LmIxpu6sov6fh15D8MVQLwOG4KAecJrwS1B6jHRN_g=88BE301E"))
    response = ctx.deps.jira_deps.client.create_issue(fields=issue_dict)

    if ctx.deps.jira_deps.images:
        ctx.deps.jira_deps.client.add_attachment(issue=response.key, attachment=ctx.deps.jira_deps.images[0].path)

    # Extract the search results
    results = JiraResponse(
        issue_key=response.key,
        description=response.fields.description
    )

    return results


@jira_agent.tool
async def update_jira_issue(ctx: RunContext[MainDependencies],key:str, summary: str, description: str) -> None:
    """
    Create a new Jira issue.
    :param ctx: RunContext object containing the dependencies.
    :param key: Key of the issue to be updated.
    :param summary: Summary of the issue to be created.
    :param description: Description of the issue to be created. if the description contains json data,then format it properly.
    :return: JiraResponse object containing the issue key and description.
    """
    # Define the search query
    issue_dict = {
        'summary': summary,
        'description': description,
    }

    # Perform the search query on the specified index
    # response = ctx.deps.client.create_issue(fields=issue_dict)
    ctx.deps.client = JIRA(server='https://manoharant.atlassian.net',
                           basic_auth=("manoharant@gmail.com","ATATT3xFfGF0z2tETfqzBNomWPdxMACIXRMlkMd4RjEKUtOdeypHun-ch4m28CgucDI5LJrdX3jpJg66nRpXx10lG8PCNYgQVM4kU63po59ZoRFsfVgazqZc0C3LR1bB_jnrepQOrKHt7LmIxpu6sov6fh15D8MVQLwOG4KAecJrwS1B6jHRN_g=88BE301E"))
    issue = ctx.deps.jira_deps.client.issue(key)
    issue.update(fields=issue_dict)


############## Teams Agent ##############

system_prompt = """
    You are a microsoft teams expert with access to microsoft Teams to help the user to publish the message in teams channels.

    Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

    Don't ask the user before taking an action, just do it. Always make sure you look at the index with the provided tools before answering the user's question unless you have already.

    When answering a question about the query, always start your answer with the full details in brackets and then give your answer on a newline. Like:

    [Using [query from the user]]

    Your answer here...
    
    After the answer, you can also provide the suggestions to the user if you have any.
    
    [Suggestions: [suggestion goes here]]
    """




msteams_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=MSTeamsDeps,
    retries=0
)


@msteams_agent.tool
async def publish_message_in_teams_channel(ctx: RunContext[MainDependencies],issue_key:str,summary:str) -> str:
    """
    Publish a message in the Microsoft Teams channel.
    This message should contain the below details of the issue created in Jira.
    - Issue key : Issue key of the created issue with the hyperlink.
    - Summary: Summary of the issue created.
    :param ctx: RunContext object containing the dependencies.
    :param issue_key: Issue key of the created issue.It should be shown with hyperlink for easy navigation,base_url(https://trackspace.lhsystems.com/browse/) + issue_key.
    :param summary: Summary of the issue created.
    :return: Message published in the Microsoft Teams channel.
    """
    # Initialize the connector card with your webhook URL
    myTeamsMessage = pymsteams.connectorcard(ctx.deps.ms_teams_deps.url)

    # Set the message color
    myTeamsMessage.color("#F8C471")

    message = (f"New Jira issue created with the following details:\n\n"
               f"- Issue key: [{issue_key}](https://trackspace.lhsystems.com/browse/{issue_key})\n"
               f"- Summary: {summary}\n")

    print(f"Teams Message: {message}")

    # Add your message text
    myTeamsMessage.text(message)

    # Send the message
    return myTeamsMessage.send()


#################### Main Agent ####################

system_prompt = """
    You are a Elastic agent and you are an elastic expert with access to Elasticsearch to help the user manage the log index and get information from it.

    You also manage 2 sub agents, Jira agent to help the user manage the story,task and issue creation and get information from it. Microsoft Teams agent to post messages whenever Jira agent creates any item in Jira.

    Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

    Don't ask the user before taking an action, just do it. Always make sure you look at the index with the provided tools before answering the user's question unless you have already.

    When answering a question about the query, always start your answer with the full details in brackets and then give your answer on a newline. Like:

    [Using [query from the user]]

    Your answer here...
    """


@dataclass
class ESResponse:
    results: list[dict]


@dataclass
class Result:
    conversationid: str
    message: str
    payload: str




main_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=ElasticDeps,
    retries=0
)


@main_agent.tool
async def delegate_to_jira_agent(ctx: RunContext[MainDependencies], details: str) -> str:
    print(f"Delegating to jira agent with details: {details}")
    result = await jira_agent.run(f"Can you solve {details}", deps=ctx.deps)
    return result.data

@main_agent.tool
async def delegate_to_ms_teams_agent(ctx: RunContext[MainDependencies], details: str) -> str:
    print(f"Delegating to jira agent with details: {details}")
    result = await msteams_agent.run(f"Can you publish {details}", deps=ctx.deps)
    return result.data

@jira_agent.tool
async def delegate_to_elastic_agent(ctx: RunContext[MainDependencies], details: str) -> str:
    print(f"Delegating to Elastic agent with details: {details}")
    result = await main_agent.run(f"Can you solve {details}", deps=ctx.deps)
    return result.data

@main_agent.tool
async def get_result_conversation_id(ctx: RunContext[MainDependencies], conversationid: str) -> ObjectApiResponse[Any]:
    """
    Get the result of an Elasticsearch query by conversation ID.

    :param ctx: agent context
    :param conversationid: conversation ID to extract the result
    :return: ESResponse containing the search results
    """
    # Define the search query
    print(f"Delegating to Elasticsearch agent with conversationid details: {conversationid}")
    search_query = {
        "query": {
            "match": {
                "conversationid": conversationid
            }
        },
        "size": 500,
        "_source": ["conversationid", "message", "payload"]
    }

    # Perform the search query on the specified index
    response = ctx.deps.es_deps.client.search(index=ctx.deps.es_deps.index_name, body=search_query)

    # Convert the search results to Result objects
    results = []
    for hit in response['hits']['hits']:
        source = hit['_source']
        result = Result(
            conversationid=source.get('conversationid', ''),
            message=source.get('message', ''),
            payload=source.get('payload', '')
        )
        results.append(result)

    return results


#################### Chat with agents ####################

elastic_host = os.getenv('ELASTICSEARCH_HOST')
elastic_port = os.getenv('ELASTICSEARCH_PORT')
elastic_index = os.getenv('ELASTICSEARCH_INDEX')
elastic_scheme = os.getenv('ELASTICSEARCH_SCHEME')
es_deps = ElasticDeps(
    client=Elasticsearch([{'host': elastic_host, 'port': int(elastic_port), 'scheme': elastic_scheme}]),
    index_name=elastic_index)

jira_url = os.getenv('JIRA_URL')
username = os.getenv('JIRA_USERNAME')
api_token = os.getenv('JIRA_API_TOKEN')
jira_deps = JiraDeps(client=JIRA(server=jira_url, token_auth=api_token),
                     project_key=os.getenv('JIRA_PROJECT_KEY'))

ms_teams_url = os.getenv('MICROSOFT_TEAMS_WEBHOOK_URL')
ms_teams_deps = MSTeamsDeps(url=ms_teams_url)

main_dependencies = MainDependencies(
    es_deps=es_deps,
    jira_deps=jira_deps,
    ms_teams_deps=ms_teams_deps,
)

######## UI logic goes here ########

@cl.on_chat_start
async def start():
    cl.user_session.memory = []
    cl.user_session.settings = None
    main_dependencies.jira_deps.images = None
    settings = await cl.ChatSettings(
        [
            Select(
                id="selected_option",
                label="Select Option",
                values=["Search for logs", "Search for Jira incidents", "Search for Jira stories"],
                initial_index=0,
            ),
        ]
    ).send()
    cl.user_session.settings = settings


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.settings = settings
    logfire.info(f"on_settings_update {settings}")

@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    # Send a response back to the user
    print(f"Selected option: {cl.user_session.settings["selected_option"]}")
    if cl.user_session.settings["selected_option"] == "Search for logs":
        # Processing images exclusively
        images = [file for file in message.elements if "image" in file.mime]
        if images:
            main_dependencies.jira_deps.images = images
            with open(images[0].path, 'rb') as file:
                image_data = file.read()
            response = await main_agent.run(
                [
                    'Summarize the details from the image. If you find any error or conversation id, then search with elastic and summarize with the detailed information.',
                    BinaryContent(data=image_data, media_type='image/jpeg'),
                ], message_history=cl.user_session.memory, deps=main_dependencies
            )
        else:
            response = await main_agent.run(message.content, message_history=cl.user_session.memory, deps=main_dependencies)
        cl.user_session.memory = response.all_messages()
        await cl.Message(content=response.data).send()
    elif cl.user_session.settings["selected_option"] == "Search for Jira incidents":
        main_dependencies.jira_deps = JiraDeps(client=JIRA(server=jira_url, token_auth=api_token),
                             project_key=os.getenv('JIRA_INCIDENT_PROJECT_KEY'))
        response = await jira_agent.run(message.content, message_history=cl.user_session.memory, deps=main_dependencies)
        cl.user_session.memory = response.all_messages()
        await cl.Message(content=response.data).send()
    elif cl.user_session.settings["selected_option"] == "Search for Jira stories":
        main_dependencies.jira_deps = JiraDeps(client=JIRA(server=jira_url, token_auth=api_token),
                             project_key=os.getenv('JIRA_PROJECT_KEY'))
        response = await jira_agent.run(message.content, message_history=cl.user_session.memory, deps=main_dependencies)
        cl.user_session.memory = response.all_messages()
        await cl.Message(content=response.data).send()
