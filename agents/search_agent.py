from azure.identity import DefaultAzureCredential
from semantic_kernel.functions import kernel_function
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import AzureAISearchTool
from dotenv import load_dotenv

import os

load_dotenv()

########################
#                      #
# Author: Robert Patel #
#                      #
########################

class SearchAgent:
    """
    A class to represent the Search Agent.
    """

    def __init__(self, progress_dialog=None):
        self.progress_dialog = progress_dialog

    @kernel_function(description='An agent that searches health plan documents.')
    def search_agent(self, filename:str) -> str:
        """
        Creates an Azure AI Agent that searches an Azure AI Search index for questions and answers in a questionnaire file.

        Parameters:
        filename (str): The name of the file to search for questions and answers in.

        Returns:
        last_msg (json): The last message from the agent, which contains the questions and answers.

        """
        print("Calling SearchAgent...")

        # Connecting to our Azure AI Foundry project, which will allow us to use the deployed gpt-4o model for our agent
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=os.getenv("AIPROJECT_CONNECTION_STRING"),
            )
        
        # Iterate through the connections in your project and get the connection ID of the Aazure AI Search connection.
        conn_list = project_client.connections.list()
        conn_id = ""
        for conn in conn_list:
            if conn.connection_type == "CognitiveSearch":
                conn_id = conn.id
        # Connect to your Azure AI Search index
        ai_search = AzureAISearchTool(
            index_connection_id=conn_id,
            index_name=os.getenv("AZURE_AI_SEARCH_INDEX"),
        )

        # Create an agent that will be used to search for health plan information
        search_agent = project_client.agents.create_agent(
            model="gpt-35-turbo",
            name="search-agent",
            instructions = f"""
                You are a helpful assistant with expertise in extracting structured question-and-answer pairs from vendor assessments.
                
                You must extract only the content from a document whose metadata title is exactly equal to: '{filename}'.
                
                Double-check that the content you are using comes **only** from a document where the `title` field is exactly '{filename}'. Do not reference or use any data from other documents, even if the content appears similar.
                
                Ignore any content if the title is different. Do not merge, infer, or hallucinate any data.
                
                The input may include content from multiple documents — your job is to strictly filter and only use chunks where the title matches '{filename}'.
                
                Your task is to extract **every question-answer pair** that appears in the document, regardless of:
                - Length
                - Format
                - Style
                
                Include:
                - Long-form paragraph responses
                - Very short answers like “Yes”, “No”, “N/A”
                - Questions with bullet point or checkbox-style answers
                - Numbered sections like “4.6.1”, “4.6.2”, etc.
                
                Output only a valid JSON array of objects like this:
                
                [
                  [
                    "question_id": "...",
                    "question_text": "...",
                    "detail_answer": "..."
                  ],
                  ...
                ]
                
                Do not include markdown, explanations, or commentary. Return only the raw JSON.
                
            """,
        tools=ai_search.definitions,
        tool_resources=ai_search.resources,
        ) 

        # Create a thread which is a conversation session between an agent and a user.
        thread = project_client.agents.create_thread()

        # Create a message in the thread with the user asking for information about a specific health plan
        message = project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=f"""
                Only extract question-answer pairs from the document whose title is exactly '{filename}'.
                
                Do not use or include information from any other documents — even if the questions or answers seem related.

                
                Return only a valid JSON array of objects with the format:
                [
                  (
                    "question_id": "...",
                    "question_text": "...",
                    "detail_answer": "..."
                  ,
                  ...
                ]
                
                Do not include markdown or commentary — return the raw JSON array only.
                                
                """
        )

        # Run the agent to process the message in the thread
        run = project_client.agents.create_and_process_run(thread_id=thread.id, assistant_id=search_agent.id)

        # Check if the run was successful
        if run.status == "failed":
            print(f"Run failed: {run.last_error}")

        # Delete the agent when it's done running
        project_client.agents.delete_agent(search_agent.id)

        # Fetch all the messages from the thread
        messages = project_client.agents.list_messages(thread_id=thread.id)

        # Get the last message, which is the agent's response to the user's question
        last_msg = messages.get_last_text_message_by_role("assistant")

        print("SearchAgent completed successfully.")
        print(last_msg)
        return last_msg
