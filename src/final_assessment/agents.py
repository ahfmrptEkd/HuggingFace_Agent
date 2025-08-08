from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from tools import google_grounding_search, execute_python, process_image, download_files_from_api, process_code_file, process_csv, process_pdf, process_excel, process_archive, read_text_file, process_audio

load_dotenv()

# System prompt for GAIA benchmark
SYSTEM_PROMPT = """You are an expert AI assistant designed to solve GAIA benchmark questions. Your primary goal is to provide accurate, concise, and precisely formatted answers.
ANSWER FORMAT RULES:
- Provide ONLY the final answer. Do NOT include any prefixes like "FINAL ANSWER:", "Answer:", or "The answer is:".
- If the answer is a number: Do NOT include commas or units (unless the question explicitly asks for units). Provide only the numerical value.
- If the answer is a string: Do NOT include articles (a/an/the) or abbreviations. Digits should be in plain text (e.g., "one", "two").
- If the answer is a list: Provide a comma-separated list. Each element in the list must adhere to the above rules for numbers or strings.
PROBLEM-SOLVING APPROACH:
1. Analyze the question carefully to understand the core problem and required output format.
2. Devise a comprehensive plan to solve the problem, considering all necessary steps and aiming to minimize tool calls for efficiency.
3. Determine which tools are necessary to gather information, perform calculations, or process data.
4. Execute tools step-by-step, verifying intermediate results.
5. Synthesize information from tool outputs to formulate the final answer.
6. Ensure the final answer strictly adheres to the ANSWER FORMAT RULES.
TOOLS AVAILABLE:
- google_grounding_search(query: str): Use this for general web searches, current events, or information not available in your training data.
- execute_python(code: str): Use this for complex calculations, data manipulation, or running Python scripts.
- process_image(image_path: str): Use this to analyze local image files, extract text, or get visual descriptions.
- download_files_from_api(task_id: str, file_extension: str = None): Use this ONLY when the question explicitly mentions files, attachments, or uploaded content associated with a task ID.
- process_code_file(code_file_path: str): Use this to read and execute local code files (currently supports Python).
- process_csv(csv_path: str, operation: str = "summary", params: dict = None): Use this to analyze and extract data from local CSV files.
- process_pdf(pdf_path: str): Use this to extract text content from local PDF files.
- process_excel(excel_path: str, operation: str = "summary", params: dict = None): Use this to analyze and extract data from local Excel files.
- process_archive(archive_path: str, operation: str = "list", extract_to: str = None): Use this to list or extract contents of local .zip archive files.
- read_text_file(file_path: str): Use this to read the content of any local text-based file (e.g., .txt, .md, .json).
- process_audio(audio_path: str): Use this to transcribe and analyze local audio files.
- process_youtube_video(url: str, question: str): Use this ONLY when a YouTube URL is provided in the question to analyze video content.
Be precise and methodical in your approach. Your answer will be compared for exact match against the benchmark solution."""

class GaiaAgent:
    def __init__(self):
        """Initialize the GAIA agent with Gemini and tools"""
        
        # Get API key - works both locally (.env) and on HF Spaces (secrets)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Initialize chat model
        self.chat = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,  # Lower temperature for consistent answers
        )
        
        # Set up tools
        self.tools = [
            google_grounding_search, 
            execute_python, 
            process_image, 
            download_files_from_api, 
            process_code_file, 
            process_csv, 
            process_pdf,
            process_excel,
            process_archive,
            read_text_file,
            process_audio
        ]
        self.chat_with_tools = self.chat.bind_tools(self.tools)
        
        # Build the LangGraph workflow
        self.agent = self._build_agent()
    
    def _build_agent(self):
        """Build the LangGraph agent workflow"""
        
        # Define agent state
        class AgentState(TypedDict):
            messages: Annotated[list[AnyMessage], add_messages]
        
        def assistant(state: AgentState):
            """Main assistant node"""
            return {
                "messages": [self.chat_with_tools.invoke(state["messages"])],
            }
        
        # Build the graph
        builder = StateGraph(AgentState)
        
        # Define nodes
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(self.tools))
        
        # Define edges
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,  # If tools needed, go to tools; otherwise end
        )
        builder.add_edge("tools", "assistant")
        
        return builder.compile()
    
    def __call__(self, question: str) -> str:
        """Main interface for app.py - solve a question and return clean answer"""
        return self.solve_question(question)
    

    def solve_question(self, question: str) -> str:
        """
        Solve a GAIA question and return the final answer
        
        Args:
            question (str): The GAIA question to solve (may include TASK_ID metadata)
            
        Returns:
            str: Clean final answer for exact match scoring
        """
        try:
            # Extract task_id if present in the question format
            task_id = None
            actual_question = question
            
            if question.startswith("TASK_ID:"):
                lines = question.split("\n", 2)
                if len(lines) >= 3 and lines[1] == "" and lines[2].startswith("QUESTION:"):
                    task_id = lines[0].replace("TASK_ID:", "").strip()
                    actual_question = lines[2].replace("QUESTION:", "").strip()
                    print(f"Extracted task_id: {task_id}")
            
            # Create enhanced system prompt with task_id context if available
            system_prompt = SYSTEM_PROMPT
            if task_id:
                system_prompt += f"\n\nIMPORTANT: This question has task_id '{task_id}'. ONLY use the download_files_from_api tool if the question explicitly references files, attachments, or uploaded content (e.g., 'in the image', 'attached file', 'spreadsheet', 'document', 'audio file'). Do not attempt to download files for general knowledge questions."
            
            # Create initial message with system prompt and actual question
            messages = [
                HumanMessage(content=f"{system_prompt}\n\nQuestion: {actual_question}")
            ]
            
            # Run the agent
            response = self.agent.invoke({"messages": messages})
            
            # Extract the final answer from the last message
            final_message = response['messages'][-1]
            final_answer = final_message.content.strip()
            
            # Clean up the answer - remove any potential prefixes
            prefixes_to_remove = [
                "FINAL ANSWER:",
                "Final Answer:",
                "Answer:",
                "The answer is:",
                "The final answer is:",
                "Result:",
            ]
            
            for prefix in prefixes_to_remove:
                if final_answer.startswith(prefix):
                    final_answer = final_answer[len(prefix):].strip()
            
            return final_answer
            
        except Exception as e:
            print(f"Error solving question: {e}")
            return f"Error: Unable to solve question - {str(e)}"

# For backward compatibility and testing
def create_agent():
    """Factory function to create a GAIA agent"""
    return GaiaAgent()

# For direct testing (remove this section before deployment if desired)
if __name__ == "__main__":
    try:
        agent = GaiaAgent()
        test_question = "search the web for 42nd president and their wifes name"
        result = agent.solve_question(test_question)
        print(f"Test result: {result}")
    except Exception as e:
        print(f"Test failed: {e}")