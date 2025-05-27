from azure.identity import DefaultAzureCredential
from semantic_kernel.functions import kernel_function
from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv

import os

load_dotenv()

########################
#                      #
# Author: Robert Patel #
#                      #
########################

class ReportAgent:
    """
    Generates an executive‑ready narrative report from the structured JSON
    emitted by RiskAssessmentAgent.
    """

    def __init__(self, progress_dialog=None):
        self.progress_dialog = progress_dialog

    @kernel_function(description="Create a narrative report from a risk‑analysis JSON string.")
    def write_report(self, risk_json: str) -> str:
        """
        Parameters
        ----------
        risk_json : str
        The JSON string returned by RiskAssessmentAgent.

        Returns
        -------
        str
        A plain‑text report suitable for executives and stakeholders.
        """
        print("Calling ReportAgent...")
        print(f"\nRetrieved information from risk_assessment_agent!\n{risk_json}\n")
        # Connect to Azure AI Foundry project
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=os.getenv("AIPROJECT_CONNECTION_STRING"),
        )

        # Create a lightweight agent focused on report writing
        report_agent = project_client.agents.create_agent(
            model="gpt-35-turbo",
            name="risk-report-agent",
            instructions="""
                Create a clean risk report with this exact structure:

                Risk Analysis:
                - [Question ID]: [Risk Level]
                Rationale: [One sentence]

                Executive Summary:
                [10-12 sentences and rationale behind why it is a Pass or Fail]

                Rules:
                - Use only plaintext
                - Never include JSON or markdown
                - Maintain consistent spacing
                - Keep rationales concise
                """
        )

        # Start an isolated thread
        thread = project_client.agents.create_thread()

        project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content=f"""Use the structured risk-analysis JSON below to generate a report using the required format:

            {risk_json}

            Only return the report in plaintext, following the format exactly."""
        )


        # Execute the run
        run = project_client.agents.create_and_process_run(
            thread_id=thread.id, assistant_id=report_agent.id
        )
        if run.status == "failed":
            raise RuntimeError(f"ReportAgent run failed: {run.last_error}")

        # Retrieve the finished message
        messages = project_client.agents.list_messages(thread_id=thread.id)
        last_msg = messages.get_last_text_message_by_role("assistant")

        # Optional cleanup (comment out if you want to reuse the agent)
        project_client.agents.delete_agent(report_agent.id)

        # Updates the progress bar.
        if self.progress_dialog:
            self.progress_dialog.update_progress(100, "Report completed")

        print("ReportAgent completed successfully.")

        return last_msg