import os
import json
from azure.identity import DefaultAzureCredential
from semantic_kernel.functions import kernel_function
from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv

load_dotenv()

########################
#                      #
# Author: Robert Patel #
#                      #
########################

class RiskAssessmentAgent:
    """
    A class to represent the Risk Assessment Agent.
    """

    def __init__(self, progress_dialog=None):
        self.progress_dialog = progress_dialog

    @kernel_function(description='An agent that evaluates third-party risk based on questionnaire responses.')
    def assess_risk(self, qa_json: str) -> str:
        """
        Creates an Azure AI Agent that reviews answers to a vendor questionnaire and evaluates risk.

        Parameters:
        qa_json (str): JSON string of questions and answers extracted from the vendor assessment.

        Returns:
        last_msg (json): The last message from the agent, which contains the risk assessment results.
        """
        print("Calling RiskAssessmentAgent...")
        print(f"\nRetrieved the questions & answers JSON from Search Agent!\n{qa_json}\n")
        # Connect to Azure AI Foundry project
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=os.getenv("AIPROJECT_CONNECTION_STRING"),
        )

        # Create the risk assessment agent
        risk_agent = project_client.agents.create_agent(
            model="gpt-35-turbo",
            name="risk-assessment-agent",
            instructions="""
            You are an expert Enterprise Risk Evaluator and Architect with 15+ years of experience in security, compliance, and SaaS architecture. 
            You are reviewing third-party risk assessments and vendor questionnaires.

                - Assign a risk rating (Low, Medium, High) per item.
                - If answers suggest poor security, missing capabilities, or vague responses, rate the risk higher.
                - At the end, summarize if the vendor overall should Pass or Fail based on cumulative risk.

                Return only a structured JSON like:
                [
                {
                    "question_id": "...",
                    "risk_level": "Low|Medium|High",
                    "rationale": "..."
                },
                ...
                {
                    "verdict": "Pass|Fail",
                    "justification": "..."
                }
                ]
                """
)

        # Start a new thread for this assessment session
        thread = project_client.agents.create_thread()

        # Provide the JSON Q&A as user input
        message = project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=f"Evaluate the following questionnaire responses for risk: {qa_json}"

        )

        # Run the assessment
        run = project_client.agents.create_and_process_run(thread_id=thread.id, assistant_id=risk_agent.id)

        if run.status == "failed":
            print(f"Run failed: {run.last_error}")

        # Clean up
        project_client.agents.delete_agent(risk_agent.id)

        # Get the final message from the assistant
        messages = project_client.agents.list_messages(thread_id=thread.id)
        last_msg = messages.get_last_text_message_by_role("assistant")

        print("RiskAssessmentAgent completed successfully.")
        return last_msg