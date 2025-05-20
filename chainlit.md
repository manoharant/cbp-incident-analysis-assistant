# Welcome to DIAA - Digital Incident Analysis Assistant! 🚀🤖

Hi there, 👋 We're excited to have you on board. Incident Analysis tool designed to help you to analyze incidents and create issues.

# Setup
Follow the below steps to setup the project.

## Environment Setup

- Create a `.env` file in the root directory using the `.env.example` file
- Add the following environment variables
  - `AZURE_OPENAI_ENDPOINT` - Azure endpoint
  - `AZURE_OPENAI_API_KEY` - Azure subscription key
  - `AZURE_OPENAI_API_VERSION` - Azure API version
  - `AZURE_OPENAI_MODEL_NAME` - Azure model name
- Add the following environment variables to search elastic index
  - `ELASTICSEARCH_HOST` - Elastic host
  - `ELASTICSEARCH_PORT` - Elastic port
  - `ELASTICSEARCH_INDEX` - Elastic index
  - `ELASTICSEARCH_SCHEME` - Elastic scheme
- Jira credentials
  - `JIRA_URL` - Jira url
  - `JIRA_API_TOKEN` - Jira API token
  - `JIRA_PROJECT_KEY` - Jira project key(e.g. `NEWBE`)
  - `JIRA_INCIDENT_PROJECT_KEY` - Jira incident project key(e.g. `LCAGIM`)
- Local Jira settings
  - `LOCAL_JIRA_URL` - Jira url (e.g. `http://localhost:8080`)
  - `LOCAL_JIRA_API_TOKEN` - Jira API token
  - `LOCAL_JIRA_PROJECT_KEY` - Jira project key(e.g. `AIPOC`)

- Teams credentials
  - `MICROSOFT_TEAMS_WEBHOOK_URL` - Teams webhook url

- LOGFIRE settings
  - `LOGFIRE_CONSOLE` - Logfire console disable(default: `false`)

## Prerequisites
- Python 3.10 or above
- create virtual environment
- install the required packages
  - `pip install -r requirements.txt`
- run the project
  - `chainlit run app.py`

## Project Structure

- `app.py` - Streamlit app
    - this has 3 AI agents for interacting with Jira and Elastic
        - `JiraAgent` - to fetch the data from Jira
        - `ElasticAgent` - to fetch the data from Elastic
        - `TeamsAgent` - to send the data to Teams
