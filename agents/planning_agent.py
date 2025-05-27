import json, asyncio, re, os
from dotenv import load_dotenv
from semantic_kernel import Kernel

from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments
from agents.search_agent import SearchAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.report_agent import ReportAgent

load_dotenv()

########################
#                      #
# Author: Robert Patel #
#                      #
########################

class PlannerAgent:
    
    def __init__(self, progress_dialog=None):
        self.progress_dialog = progress_dialog

    async def planner_main(self, user_input: str) -> str:
        # The envionrment variables needed to connect to the gpt-4o model in Azure AI Foundry
        deployment_name = "gpt-35-turbo"
        endpoint = os.getenv("AZURE_OPEN_AI_ENDPOINT")
        api_key = os.getenv("AZURE_OPEN_AI_KEY")
        # The Kernel is the main entry point for the Semantic Kernel. It will be used to add services and plugins to the Kernel.
        kernel = Kernel()
    
        # Add the necessary services and plugins to the Kernel
        # Adding the ReportAgent and SearchAgent plugins will allow the OrchestratorAgent to call the functions in these plugins
        service_id = "planner_agent"
        kernel.add_service(AzureChatCompletion(service_id=service_id, deployment_name=deployment_name, endpoint=endpoint, api_key=api_key))
        kernel.add_plugin(SearchAgent(progress_dialog=self.progress_dialog), plugin_name="SearchAgent")
        kernel.add_plugin(RiskAssessmentAgent(progress_dialog=self.progress_dialog), plugin_name="RiskAssessmentAgent")
        kernel.add_plugin(ReportAgent(progress_dialog=self.progress_dialog), plugin_name="ReportAgent")

    
        settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
        # Configure the function choice behavior to automatically invoke kernel functions
        settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
        settings.parallel_tool_calls = None        # ← turn it off
    
        # Create the Orchestrator Agent that will call the Search and Report agents to create the report
        agent = ChatCompletionAgent(
            kernel=kernel,
            # Creates the planner agent and provides instructions on how to operate each agent.
            name="PlannerAgent",
            instructions=f"""
                You are an agent designed to generate executive-level risk assessment reports on third-party vendors. The user will provide the name of a file containing responses to a security questionnaire. Your job is to orchestrate the generation of the report by calling the appropriate functions. 

                Do not write the report yourself. Your role is to be an orchestrator that delegates work to the specialized agents available to you. Each plugin available to you is an agent with a specific task. Here are descriptions of the plugins you have access to:

                - SearchAgent: An agent that reads and extracts question/answer pairs from the provided document. It returns a JSON array with the fields: "question_id", "question_text", and "detail_answer".
                - RiskAssessmentAgent: An agent that evaluates each question/answer pair and assigns a risk level and rationale. It returns a JSON array of risk assessments and a final object with the structure: {{"verdict": "Pass" or "Fail", "justification": "..."}}.
                - ReportAgent: An agent that turns the output from RiskAssessmentAgent into a human-readable executive report.

                You must follow these steps exactly in the order they are given to you below:

                1. Call SearchAgent to extract the question/answer data from the provided file.
                2. Call RiskAssessmentAgent using the full output from SearchAgent.
                3. Call ReportAgent using the full output from RiskAssessmentAgent.

                Once the ReportAgent returns the final report, you must return a single valid JSON object to the user with two attributes:

                - report_was_generated: A boolean value that indicates whether the report was successfully created. Use true if the report was created, or false if something went wrong.
                - content: A string containing the executive report if successful, or an explanation of the error if unsuccessful.

                Here’s an example of a valid JSON response:
                {{"report_was_generated": false, "content": "The report for enterprise-test-file-10.pdf could not be generated because the extracted data was incomplete."}}

                Your response must contain only a single valid JSON object. Do not include any extra text, comments, or markdown. Use lowercase true/false values and double quotes for all keys and string values. The output will be parsed using json.loads() — if it is not a valid JSON object, it will result in a runtime error.
                """
,
        )
    
        # Now use the argument passed into the function
        chat_history = ChatHistory()
        # Adds user input to chat history for agent to utilize
        chat_history.add_user_message(f"Please use the file named '{user_input}' for SearchAgent.search_agent().")

        args = KernelArguments()
        args["chat_history"] = chat_history

        #
        async for response in agent.invoke(arguments=args):
            fixed = response.content.content.replace("True", "true").replace("False", "false")

            # Extract all JSON blocks
            json_matches = re.findall(r'\{.*?\}', fixed, re.DOTALL)

            if not json_matches:
                raise ValueError("No valid JSON objects found in LLM output.")

            # oad the last JSON block
            try:
                result = json.loads(json_matches[-1])
            except json.JSONDecodeError as e:
                print("SON decode error:", e)
                print("Raw matched content:\n", json_matches[-1])
                raise

            # Extract and return report content
            report = result.get("content", "")
            required_sections = ["Executive Summary:", "Risk Analysis:"]
            if all(section in report for section in required_sections):
                return report