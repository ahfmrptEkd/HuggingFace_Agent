from langchain_core.tools import tool
import os
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import base64
import tempfile
import pypdf
import pandas
import zipfile
from pathlib import Path
import mimetypes
from typing import Optional
import whisper
import torch
import yt_dlp
import google.generativeai as genai
import time

load_dotenv()

vision_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
# Create the underlying REPL tool
#_python_repl = PythonREPLTool()


@tool
def google_grounding_search(query: str) -> str:
    """
    Search for current information using Google's grounded search.
    
    Use this tool when you need:
    - Latest/current information (news, events, prices, etc.)
    - Real-time data that might not be in your training
    - Recent developments or updates
    - Current facts to supplement your knowledge
    
    Args:
        query: Search query (be specific and focused)
        
    Returns:
        Current information from Google search with citations
        
    Example usage:
    - google_grounding_search("latest AI news January 2025")
    - google_grounding_search("current Tesla stock price")
    - google_grounding_search("Manchester United new signings 2025")
    """
    try:
        # Import the newer Google genai library
        from google import genai
        from google.genai import types
        import os
        
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY not found in environment variables"
        
        # Initialize client and grounding tool
        client = genai.Client(api_key=api_key)
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        
        # Configure for grounding
        grounding_config = types.GenerateContentConfig(
            tools=[grounding_tool]
        )
        
        #print(f"🔎 Performing grounded search for: {query}")
        
        # Make grounded search request
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Search for and provide current information about: {query}",
            config=grounding_config
        )
        
        result = response.text.strip()
        
        if not result:
            return "No results found from grounded search"
            
        return f"Current Information (via Google Search):\n{result}"
        
    except ImportError as e:
        return f"Error: google-genai library not available. Import error: {str(e)}"
    except Exception as e:
        return f"Error performing grounded search: {str(e)}"
    
@tool
def execute_python(code: str) -> str:
    """Execute Python code for mathematical calculations, data analysis, and general computation.
    
    Args:
        code: Valid Python code to execute
        
    Returns:
        The output/result of the executed code
    """
    try:
        # For simple calculations, use eval
        if all(char in "0123456789+-*/.() " for char in code.strip()):
            result = eval(code)
            return str(result)
        
        # For more complex code, use exec with captured output
        import io
        import sys
        from contextlib import redirect_stdout
        
        # Capture stdout
        captured_output = io.StringIO()
        local_vars = {}
        
        with redirect_stdout(captured_output):
            exec(code, {"__builtins__": __builtins__}, local_vars)
        
        output = captured_output.getvalue().strip()
        
        # If no output was printed, try to return the last variable value
        if not output and local_vars:
            # Get the last defined variable
            last_var = list(local_vars.values())[-1] if local_vars else None
            if last_var is not None:
                return str(last_var)
        
        return output if output else "Code executed successfully (no output)"
        
    except Exception as e:
        return f"Error executing code: {str(e)}"
    
@tool
def download_files_from_api(task_id: str, file_extension: str = None) -> str:
    """Downloads a file (image, PDF, CSV, code, audio, Excel, etc.) associated with a task ID from the API.
    The file is saved to a temporary location, and its local path is returned.
    
    Args:
        task_id: The task ID for which to download the file.
        file_extension: Optional. The expected file extension (e.g., ".py", ".csv", ".pdf").
                        If provided, this will be used for the temporary file.
                        Otherwise, the extension will be inferred from the Content-Type header.
        
    Returns:
        The absolute path to the downloaded file, or an error message.
    """
    try:
        api_url = "https://agents-course-unit4-scoring.hf.space"
        response = requests.get(f"{api_url}/files/{task_id}", timeout=30)
        response.raise_for_status()
        
        ext = file_extension
        if not ext:
            # Determine file extension from headers or default to .bin
            content_type = response.headers.get('Content-Type', '')
            if 'image/jpeg' in content_type:
                ext = '.jpg'
            elif 'image/png' in content_type:
                ext = '.png'
            elif 'application/pdf' in content_type:
                ext = '.pdf'
            elif 'text/csv' in content_type:
                ext = '.csv'
            elif 'text/x-python' in content_type or 'application/x-python-code' in content_type:
                ext = '.py'
            elif 'audio/mpeg' in content_type:
                ext = '.mp3'
            elif 'audio/wav' in content_type:
                ext = '.wav'
            elif 'application/vnd.ms-excel' in content_type:
                ext = '.xls'
            elif 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in content_type:
                ext = '.xlsx'
            elif 'application/zip' in content_type:
                ext = '.zip'
            elif 'text/plain' in content_type:
                ext = '.txt'
            else:
                ext = '.bin' # Default for unknown types
            
        # Create a temporary file to save the content
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(response.content)
            file_path = temp_file.name
            
        print(f"Downloaded file for task {task_id} to: {file_path}")
        return file_path
        
    except requests.exceptions.RequestException as e:
        return f"Error downloading file from API: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def process_image(image_path: str) -> str:
    """Analyze an image file from a local path - extract any text present and provide visual description.
    This tool can handle various image formats like PNG, JPEG, GIF, etc.
    
    Args:
        image_path: The absolute path to the local image file.
        
    Returns:
        Extracted text (if any) and visual description of the image.
    """
    try:
        # Dynamically determine the MIME type
        import mimetypes
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            # Default to a common type if detection fails
            mime_type = "application/octet-stream"

        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # First call: Extract text
        text_message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Extract all the text from this image. "
                            "Return only the extracted text, no explanations. "
                            "If no text is found, respond with 'No text found'."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        },
                    },
                ]
            )
        ]
        
        text_response = vision_llm.invoke(text_message)
        extracted_text = text_response.content.strip()
        
        # Second call: Get description
        description_message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Describe what you see in this image in detail. "
                            "Be specific about objects, positions, colors, text, numbers, "
                            "and any other relevant visual information."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_base64}"
                        },
                    },
                ]
            )
        ]
        
        description_response = vision_llm.invoke(description_message)
        description = description_response.content.strip()
        
        # Format the combined result
        result = f"TEXT EXTRACTED:\n{extracted_text}\n\nVISUAL DESCRIPTION:\n{description}"
        
        return result
        
    except FileNotFoundError:
        return f"Error: Image file not found at {image_path}"
    except Exception as e:
        return f"Error processing image: {str(e)}"

@tool
def process_pdf(pdf_path: str) -> str:
    """Extracts all text content from a PDF file.
    Args:
        pdf_path: The absolute path to the local PDF file.
    Returns:
        A string containing all extracted text from the PDF, or an error message.
    """
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text if text else "No text found in PDF."
    except FileNotFoundError:
        return f"Error: PDF file not found at {pdf_path}"
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

@tool
def process_csv(csv_path: str, operation: str = "summary", params: dict = None) -> str:
    """Processes a CSV file based on the specified operation.
    Args:
        csv_path: The absolute path to the local CSV file.
        operation: The operation to perform. Supported operations:
                   "summary": Returns a summary of the CSV (head, columns, dtypes, shape).
                   "get_column": Returns the content of a specific column. Requires 'column_name' in params.
                   "filter": Filters rows based on a condition. Requires 'column', 'operator', 'value' in params.
                             Supported operators: "==", "!=", ">", "<", ">=", "<=".
                   "aggregate": Performs aggregation on a column. Requires 'agg_column', 'agg_function' in params.
                                Optional: 'group_by_column'. Supported functions: "sum", "mean", "count", "min", "max".
                   "describe": Returns descriptive statistics for numerical columns.
        params: A dictionary of parameters for the chosen operation.
    Returns:
        A string containing the result of the operation, or an error message.
    """
    if params is None:
        params = {}

    try:
        df = pandas.read_csv(csv_path)

        if operation == "summary":
            summary = f"Shape: {df.shape}\n"
            summary += f"Columns:\n{df.columns.tolist()}\n"
            summary += f"Data Types:\n{df.dtypes}\n"
            summary += f"First 5 rows:\n{df.head().to_string()}"
            return summary

        elif operation == "get_column":
            column_name = params.get("column_name")
            if column_name not in df.columns:
                return f"Error: Column '{column_name}' not found."
            return df[column_name].to_string()

        elif operation == "filter":
            column = params.get("column")
            operator = params.get("operator")
            value = params.get("value")

            if not all([column, operator, value is not None]):
                return "Error: 'column', 'operator', and 'value' are required for filter operation."
            if column not in df.columns:
                return f"Error: Column '{column}' not found."

            if operator == "==":
                filtered_df = df[df[column] == value]
            elif operator == "!=":
                filtered_df = df[df[column] != value]
            elif operator == ">":
                filtered_df = df[df[column] > value]
            elif operator == "<":
                filtered_df = df[df[column] < value]
            elif operator == ">=":
                filtered_df = df[df[column] >= value]
            elif operator == "<=":
                filtered_df = df[df[column] <= value]
            else:
                return f"Error: Unsupported operator '{operator}'."
            return filtered_df.to_string()

        elif operation == "aggregate":
            agg_column = params.get("agg_column")
            agg_function = params.get("agg_function")
            group_by_column = params.get("group_by_column")

            if not all([agg_column, agg_function]):
                return "Error: 'agg_column' and 'agg_function' are required for aggregate operation."
            if agg_column not in df.columns:
                return f"Error: Column '{agg_column}' not found."
            if group_by_column and group_by_column not in df.columns:
                return f"Error: Group by column '{group_by_column}' not found."

            if agg_function not in ["sum", "mean", "count", "min", "max"]:
                return f"Error: Unsupported aggregation function '{agg_function}'."

            if group_by_column:
                result = df.groupby(group_by_column)[agg_column].agg(agg_function)
            else:
                result = df[agg_column].agg(agg_function)
            return str(result)

        elif operation == "describe":
            return df.describe().to_string()

        else:
            return f"Error: Unsupported operation '{operation}'."

    except FileNotFoundError:
        return f"Error: CSV file not found at {csv_path}"
    except Exception as e:
        return f"Error processing CSV: {str(e)}"

@tool
def process_code_file(code_file_path: str) -> str:
    """Reads and executes a code file, returning its output along with the full code.
    Args:
        code_file_path: The absolute path to the local code file.
    Returns:
        A string containing the full code and the output of the executed code, or an error message.
    """
    try:
        with open(code_file_path, "r") as f:
            code_content = f.read()

        if code_file_path.endswith(".py"):
            execution_output = execute_python(code_content) 
            return f"--- FULL CODE ---\n{code_content}--- EXECUTION OUTPUT ---\n{execution_output}"
        else:
            return f"Error: Only Python (.py) files are supported for execution. Found: {code_file_path}"

    except FileNotFoundError:
        return f"Error: Code file not found at {code_file_path}"
    except Exception as e:
        return f"Error processing code file: {str(e)}"

@tool
def process_excel(excel_path: str, operation: str = "summary", params: dict = None) -> str:
    """Processes an Excel file based on the specified operation.
    Args:
        excel_path: The absolute path to the local Excel file.
        operation: The operation to perform. Supported operations:
                   "summary": Returns a summary of the Excel file (sheet names, columns, etc.).
                   "get_sheet": Returns the content of a specific sheet. Requires 'sheet_name' in params.
    
    Returns:
        A string containing the result of the operation, or an error message.
    """
    if params is None:
        params = {}

    try:
        xls = pandas.ExcelFile(excel_path)

        if operation == "summary":
            sheet_names = xls.sheet_names
            summary = f"Sheets: {sheet_names}\n"
            for sheet in sheet_names:
                df = pandas.read_excel(xls, sheet_name=sheet)
                summary += f"\n--- Sheet: {sheet} ---\n"
                summary += f"Shape: {df.shape}\n"
                summary += f"Columns: {df.columns.tolist()}\n"
                summary += f"First 5 rows:\n{df.head().to_string()}\n"
            return summary

        elif operation == "get_sheet":
            sheet_name = params.get("sheet_name")
            if sheet_name not in xls.sheet_names:
                return f"Error: Sheet '{sheet_name}' not found."
            df = pandas.read_excel(xls, sheet_name=sheet_name)
            return df.to_string()

        else:
            return f"Error: Unsupported operation '{operation}'."

    except FileNotFoundError:
        return f"Error: Excel file not found at {excel_path}"
    except Exception as e:
        return f"Error processing Excel file: {str(e)}"

@tool
def process_archive(archive_path: str, operation: str = "list", extract_to: str = None) -> str:
        """Processes a .zip archive file.
 
        Args:
            archive_path: The absolute path to the local .zip file.
            operation: The operation to perform. Supported operations:
                       "list": Lists the contents of the archive.
                       "extract": Extracts the entire archive. Requires 'extract_to' parameter.
            extract_to: Optional. The directory to extract the files to.
                     If not provided, it will create a directory with the same name as the archive.
    
        Returns:
            A string containing the result of the operation, or an error message.
        """
        try:
            if not zipfile.is_zipfile(archive_path):
                return f"Error: File at {archive_path} is not a valid .zip file."
    
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                if operation == "list":
                    file_list = zip_ref.namelist()
                    return f"Files in archive: {file_list}"
    
                elif operation == "extract":
                    if extract_to is None:
                        # Create a directory named after the zip file (without extension)
                        extract_to, _ = os.path.splitext(archive_path)
    
                    os.makedirs(extract_to, exist_ok=True)
                    zip_ref.extractall(extract_to)
                    return f"Archive extracted successfully to: {extract_to}"
    
                else:
                    return f"Error: Unsupported operation '{operation}'."
    
        except FileNotFoundError:
            return f"Error: Archive file not found at {archive_path}"
        except Exception as e:
            return f"Error processing archive: {str(e)}"

@tool
def read_text_file(file_path: str) -> str:
    """Reads the entire content of a text file.
    Args:
        file_path: The absolute path to the local text file (.txt, .md, .json, etc.).
    Returns:
        A string containing the full content of the file, or an error message.
    """
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error reading text file: {str(e)}"
    

# Global model cache to avoid reloading
_whisper_model = None

@tool
def process_audio(audio_path: str) -> str:
    """Analyzes an audio file using local Whisper model for transcription.
    
    Args:
        audio_path: The absolute path to the local audio file
    
    Returns:
        A transcription and basic analysis of the audio content
    """
    global _whisper_model
    
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            return f"Error: Audio file not found at {audio_path}"
        
        # Check file size
        file_size = os.path.getsize(audio_path)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return f"Error: Audio file too large ({file_size / (1024*1024):.1f}MB)"
        
        # Load model once and cache it
        if _whisper_model is None:
            try:
                _whisper_model = whisper.load_model("base")
                print("Whisper model loaded")
            except Exception as e:
                return f"Error loading Whisper model: {str(e)}\nTry: pip install openai-whisper"
        
        # Transcribe audio
        result = _whisper_model.transcribe(audio_path)
        transcription = result["text"].strip()
        detected_language = result.get("language", "unknown")
        
        # Basic info
        word_count = len(transcription.split())
        
        return f"""AUDIO TRANSCRIPTION:
        File: {Path(audio_path).name}
        Size: {file_size / (1024*1024):.1f}MB
        Language: {detected_language}
        Words: {word_count}
        TRANSCRIPT:
        {transcription}
        """
        
    except Exception as e:
        return f"Error processing audio: {str(e)}"
    
@tool
def process_youtube_video(url: str, question: str) -> str:
    """
    REQUIRED for YouTube video analysis. Downloads and analyzes YouTube videos 
    to answer questions about visual content, count objects, identify details.
    
    Use this tool WHENEVER you see a YouTube URL in the question.
    This is the ONLY way to analyze YouTube video content accurately.
    
    Args:
        url: YouTube video URL (any youtube.com or youtu.be link)
        question: The specific question about the video content
    
    Returns:
        Detailed analysis of the actual video content
    """
    try:
        # Import and configure the direct Google AI library
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        
        # Create temporary directory for video
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'best[height<=720]',  # Limit quality to save quota
                'outtmpl': str(temp_path / '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
            }
            
            print(f"Downloading video from: {url}")
            
            # Download video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                
                # Find downloaded file
                video_files = list(temp_path.glob('*'))
                if not video_files:
                    return "Error: Failed to download video file"
                
                video_file = video_files[0]
                file_size = video_file.stat().st_size / (1024 * 1024)  # MB
                
                print(f"Video downloaded: {video_title} ({duration}s, {file_size:.1f}MB)")
                
                # Check file size limit
                if file_size > 100:  # 100MB limit for Gemini
                    return f"Error: Video too large ({file_size:.1f}MB). Maximum size is 100MB."
                
                # Upload and process with Gemini
                try:
                    # Upload video file
                    print("Uploading video to Gemini...")
                    video_file_obj = genai.upload_file(str(video_file))
                    
                    # Wait for processing
                    while video_file_obj.state.name == "PROCESSING":
                        print("Processing video...")
                        time.sleep(2)
                        video_file_obj = genai.get_file(video_file_obj.name)
                    
                    if video_file_obj.state.name == "FAILED":
                        return "Error: Video processing failed"
                    
                    # Create analysis prompt
                    analysis_prompt = f"""Analyze this video carefully to answer the following question: {question}
Please examine the video content thoroughly and provide a detailed, accurate answer. Pay attention to visual details, timing, and any relevant information that helps answer the question.
Video title: {video_title}
Duration: {duration} seconds
Question: {question}"""
                    
                    # Generate analysis with Gemini 2.0 Flash
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    response = model.generate_content([analysis_prompt, video_file_obj])
                    
                    # Clean up uploaded file
                    try:
                        genai.delete_file(video_file_obj.name)
                    except:
                        pass
                    
                    return f"""VIDEO ANALYSIS:
Title: {video_title}
URL: {url}  
Duration: {duration} seconds
Size: {file_size:.1f}MB
QUESTION: {question}
ANSWER: {response.text}"""

                except Exception as processing_error:
                    return f"Error processing video with Gemini: {str(processing_error)}"
                    
    except ImportError:
        return "Error: google-generativeai library not installed. Run: pip install google-generativeai"
    except Exception as e:
        return f"Error downloading or processing video: {str(e)}"