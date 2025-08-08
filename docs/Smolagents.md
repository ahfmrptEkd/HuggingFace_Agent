# Unit 2.1 - The smolagents

---

---

# **What is smolagents ?**

`smolagents` is a simple yet powerful framework for building AI agents. It provides LLMs with the *agency* to interact with the real world, such as searching or generating images.

---

# Conent

### **2️⃣ CodeAgents**

`CodeAgents` are the primary type of agent in `smolagents`. Instead of generating JSON or text, these agents produce Python code to perform actions. This module explores their purpose, functionality, and how they work, along with hands-on examples to showcase their capabilities.

### **3️⃣ ToolCallingAgents**

`ToolCallingAgents` are the second type of agent supported by `smolagents`. Unlike `CodeAgents`, which generate Python code, these agents rely on JSON/text blobs that the system must parse and interpret to execute actions. This module covers their functionality, their key differences from `CodeAgents`, and it provides an example to illustrate their usage.

### **4️⃣ Tools**

As we saw in Unit 1, tools are functions that an LLM can use within an agentic system, and they act as the essential building blocks for agent behavior. This module covers how to create tools, their structure, and different implementation methods using the `Tool` class or the `@tool` decorator. You’ll also learn about the default toolbox, how to share tools with the community, and how to load community-contributed tools for use in your agents.

### **5️⃣ Retrieval Agents**

Retrieval agents allow models access to knowledge bases, making it possible to search, synthesize, and retrieve information from multiple sources. They leverage vector stores for efficient retrieval and implement **Retrieval-Augmented Generation (RAG)** patterns. These agents are particularly useful for integrating web search with custom knowledge bases while maintaining conversation context through memory systems. This module explores implementation strategies, including fallback mechanisms for robust information retrieval.

### **6️⃣ Multi-Agent Systems**

Orchestrating multiple agents effectively is crucial for building powerful, multi-agent systems. By combining agents with different capabilities—such as a web search agent with a code execution agent—you can create more sophisticated solutions. This module focuses on designing, implementing, and managing multi-agent systems to maximize efficiency and reliability.

### **7️⃣ Vision and Browser agents**

Vision agents extend traditional agent capabilities by incorporating **Vision-Language Models (VLMs)**, enabling them to process and interpret visual information. This module explores how to design and integrate VLM-powered agents, unlocking advanced functionalities like image-based reasoning, visual data analysis, and multimodal interactions. We will also use vision agents to build a browser agent that can browse the web and extract information from it.

---

---

# CodeAgent

## **Key Advantages of smolagents**

- **Simplicity:** Minimal code complexity and abstractions, to make the framework easy to understand, adopt and extend
- **Flexible LLM Support:** Works with any LLM through integration with Hugging Face tools and external APIs
- **Code-First Approach:** First-class support for Code Agents that write their actions directly in code, removing the need for parsing and simplifying tool calling
- **HF Hub Integration:** Seamless integration with the Hugging Face Hub, allowing the use of Gradio Spaces as tools

Unlike other frameworks where agents write actions in JSON, `smolagents` **focuses on tool calls in code**, simplifying the execution process. This is because there’s no need to parse the JSON in order to build code that calls the tools: the output can be executed directly.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png)

![image.png](./imgs/image%2017.png)

The diagram above illustrates how `CodeAgent.run()` operates, following the ReAct framework we mentioned in Unit 1. The main abstraction for agents in `smolagents` is a `MultiStepAgent`, which serves as the core building block. `CodeAgent` is a special kind of `MultiStepAgent`, as we will see in an example below.

---

---

# ToolcallingAgent

Tool Calling Agents are the second type of agent available in `smolagents`. Unlike Code Agents that use Python snippets, these agents **use the built-in tool-calling capabilities of LLM providers** to generate tool calls as **JSON structures**. 

![image.png](./imgs/image%2018.png)

When you examine the agent’s trace, instead of seeing `Executing parsed code:`, you’ll see something like:

```python
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Calling tool: 'web_search' with arguments: {'query': "best music recommendations for a party at Wayne's         │
│ mansion"}                                                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

---

---

# Tools

tools are treated as **functions that an LLM can call within an agent system**.

To interact with a tool, the LLM needs an **interface description** with these key components:

- **Name**: What the tool is called
- **Tool description**: What the tool does
- **Input types and descriptions**: What arguments the tool accepts
- **Output type**: What the tool returns

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif)

- 2ways to create tools
    1. **Using the `@tool` decorator** for simple function-based tools
        1. The `@tool` decorator is the **recommended way to define simple tools**.  (method function)
        - example
            
            ```python
            from smolagents import CodeAgent, InferenceClientModel, tool
            
            # Let's pretend we have a function that fetches the highest-rated catering services.
            @tool
            def catering_service_tool(query: str) -> str:
                """
                This tool returns the highest-rated catering service in Gotham City.
            
                Args:
                    query: A search term for finding catering services.
                """
                # Example list of catering services and their ratings
                services = {
                    "Gotham Catering Co.": 4.9,
                    "Wayne Manor Catering": 4.8,
                    "Gotham City Events": 4.7,
                }
            
                # Find the highest rated catering service (simulating search query filtering)
                best_service = max(services, key=services.get)
            
                return best_service
            
            agent = CodeAgent(tools=[catering_service_tool], model=InferenceClientModel())
            
            # Run the agent to find the best catering service
            result = agent.run(
                "Can you give me the name of the highest-rated catering service in Gotham City?"
            )
            
            print(result)   # Output: Gotham Catering Co.
            ```
            
    2. **Creating a subclass of `Tool`** for more complex functionality
        1. This approach involves creating a subclass of [**`Tool`**](https://huggingface.co/docs/smolagents/v1.8.1/en/reference/tools#smolagents.Tool). For complex tools, we can implement a class instead of a Python function. (more like class)
        2. MUST INCLUDE belows:
            - `name`: The tool’s name.
            - `description`: A description used to populate the agent’s system prompt.
            - `inputs`: A dictionary with keys `type` and `description`, providing information to help the Python interpreter process inputs.
            - `output_type`: Specifies the expected output type.
            - `forward`: The method containing the inference logic to execute.
        - example
            
            ```python
            from smolagents import Tool, CodeAgent, InferenceClientModel
            
            class SuperheroPartyThemeTool(Tool):
                name = "superhero_party_theme_generator"
                description = """
                This tool suggests creative superhero-themed party ideas based on a category.
                It returns a unique party theme idea."""
            
                inputs = {
                    "category": {
                        "type": "string",
                        "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
                    }
                }
            
                output_type = "string"
            
                def forward(self, category: str):
                    themes = {
                        "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
                        "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
                        "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
                    }
            
                    return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")
            
            # Instantiate the tool
            party_theme_tool = SuperheroPartyThemeTool()
            agent = CodeAgent(tools=[party_theme_tool], model=InferenceClientModel())
            
            # Run the agent to generate a party theme idea
            result = agent.run(
                "What would be a good superhero party idea for a 'villain masquerade' theme?"
            )
            
            print(result)  # Output: "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains."
            ```
            

- smolagent already has a pre-built in toolbox:
    - **PythonInterpreterTool**
    - **FinalAnswerTool**
    - **UserInputTool**
    - **DuckDuckGoSearchTool**
    - **GoogleSearchTool**
    - **VisitWebpageTool**

pre-built tools used case for example:

Alfred could use various tools to ensure a flawless party at Wayne Manor:

- First, he could use the `DuckDuckGoSearchTool` to find creative superhero-themed party ideas.
- For catering, he’d rely on the `GoogleSearchTool` to find the highest-rated services in Gotham.
- To manage seating arrangements, Alfred could run calculations with the `PythonInterpreterTool`.
- Once everything is gathered, he’d compile the plan using the `FinalAnswerTool`.

---

## sharing and importing tools

### **Sharing a Tool to the Hub**

```python
party_theme_tool.push_to_hub("{your_username}/party_theme_tool", token="<YOUR_HUGGINGFACEHUB_API_TOKEN>")
```

### **Importing a Tool from the Hub**

by other users using the `load_tool()` function.

```python
from smolagents import load_tool, CodeAgent, InferenceClientModel

image_generation_tool = load_tool(
    "m-ric/text-to-image",
    trust_remote_code=True
)

agent = CodeAgent(
    tools=[image_generation_tool],
    model=InferenceClientModel()
)

agent.run("Generate an image of a luxurious superhero-themed party at Wayne Manor with made-up superheros.")
```

### **Importing a Hugging Face Space as a Tool**

You can also import a HF Space as a tool using `Tool.from_space()`. This opens up possibilities for integrating with thousands of spaces from the community for tasks from image generation to data analysis.

For example, For the party, Alfred can use an existing HF Space for the generation of the AI-generated image to be used in the announcement (instead of the pre-built tool we mentioned before). Let’s build it!

```python
from smolagents import CodeAgent, InferenceClientModel, Tool

image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct")

agent = CodeAgent(tools=[image_generation_tool], model=model)

agent.run(
    "Improve this prompt, then generate an image of it.",
    additional_args={'user_prompt': 'A grand superhero-themed party at Wayne Manor, with Alfred overseeing a luxurious gala'}
)
```

### **Importing a LangChain Tool**

 we can reuse LangChain tools in your smolagents workflow!

You can easily load LangChain tools using the `Tool.from_langchain()` method.

```python
from langchain.agents import load_tools
from smolagents import CodeAgent, InferenceClientModel, Tool

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = CodeAgent(tools=[search_tool], model=model)

agent.run("Search for luxury entertainment ideas for a superhero-themed event, such as live performances and interactive experiences.")
```

### **Importing a tool collection from any MCP server**

`smolagents` also allows importing tools from the hundreds of MCP servers available on [**glama.ai**](https://glama.ai/mcp/servers) or [**smithery.ai**](https://smithery.ai/). 

1. install mcp client

```python
pip install "smolagents[mcp]"
```

```python
import os
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters
from smolagents import InferenceClientModel

model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct")

server_parameters = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], model=model, add_base_tools=True)
    agent.run("Please find a remedy for hangover.")
```

---

---

# Retrieval Agents

Agentic RAG (Retrieval-Augmented Generation) extends traditional RAG systems by **combining autonomous agents with dynamic knowledge retrieval**.

While traditional RAG systems use an LLM to answer queries based on retrieved data, agentic RAG **enables intelligent control of both retrieval and generation processes**, improving efficiency and accuracy.

## **Custom Knowledge Base Tool**

Using semantic search, the agent can find the most relevant information for Alfred’s needs.

In this example, we’ll create a tool that retrieves party planning ideas from a custom knowledge base. We’ll use a BM25 retriever to search the knowledge base and return the top results, and `RecursiveCharacterTextSplitter` to split the documents into smaller chunks for more efficient search.

```python
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, InferenceClientModel

class PartyPlanningRetrieverTool(Tool):
    name = "party_planning_retriever"
    description = "Uses semantic search to retrieve relevant party planning ideas for Alfred’s superhero-themed party at Wayne Manor."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be a query related to party planning or superhero themes.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=5  # Retrieve the top 5 documents
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved ideas:\n" + "".join(
            [
                f"\n\n===== Idea {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

# Simulate a knowledge base about party planning
party_ideas = [
    {"text": "A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.", "source": "Party Ideas 1"},
    {"text": "Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.", "source": "Entertainment Ideas"},
    {"text": "For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'", "source": "Catering Ideas"},
    {"text": "Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.", "source": "Decoration Ideas"},
    {"text": "Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.", "source": "Entertainment Ideas"}
]

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in party_ideas
]

# Split the documents into smaller chunks for more efficient search
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)

# Create the retriever tool
party_planning_retriever = PartyPlanningRetrieverTool(docs_processed)

# Initialize the agent
agent = CodeAgent(tools=[party_planning_retriever], model=InferenceClientModel())

# Example usage
response = agent.run(
    "Find ideas for a luxury superhero-themed party, including entertainment, catering, and decoration options."
)

print(response)
```

1. First check the documentation for relevant information
2. Combine insights from the knowledge base
3. Maintain conversation context in memory

## **Enhanced Retrieval Capabilities**

When building agentic RAG systems, the agent can employ sophisticated strategies like:

1. **Query Reformulation:** Instead of using the raw user query, the agent can craft optimized search terms that better match the target documents
2. **Query Decomposition:** Instead of using the user query directly, **if it contains multiple pieces of information to query, it can be decomposed to multiple queries**
3. **Query Expansion:** Somehow similar to Query Reformulation but done multiple times to put the query in multiple wordings to query them all
4. **Reranking:** **Using Cross-Encoders** to assign more comprehensive and semantic **relevance scores between retrieved documents and search query**
5. **Multi-Step Retrieval:** The agent can perform multiple searches, using initial results to inform subsequent queries
6. **Source Integration:** Information can be combined from multiple sources like web search and local documentation
7. **Result Validation:** Retrieved content can be analyzed for relevance and accuracy before being included in responses

Effective agentic RAG systems require careful consideration of several key aspects. The agent **should select between available tools based on the query type and context**.

1. Memory systems help maintain conversation history and avoid repetitive retrievals. 
2. Having fallback strategies ensures the system can still provide value even when primary retrieval methods fail. 
3. Additionally, implementing validation steps helps ensure the accuracy and relevance of retrieved information.

---

---

# Multi-Agent system

Multi-agent systems enable **specialized agents to collaborate on complex tasks**, improving modularity, scalability, and robustness.

- A **Manager Agent** for task delegation
- A **Code Interpreter Agent** for code execution
- A **Web Search Agent** for information retrieval

![https://mermaid.ink/img/pako:eNp1kc1qhTAQRl9FUiQb8wIpdNO76eKubrmFks1oRg3VSYgjpYjv3lFL_2hnMWQOJwn5sqgmelRWleUSKLAtFs09jqhtoWuYUFfFAa6QA9QDTnpzamheuhxn8pt40-6l13UtS0ddhtQXj6dbR4XUGQg6zEYasTF393KjeSDGnDJKNxzj8I_7hLW5IOSmP9CH9hv_NL-d94d4DVNg84p1EnK4qlIj5hGClySWbadT-6OdsrL02MI8sFOOVkciw8zx8kaNspxnrJQE0fXKtjBMMs3JA-MpgOQwftIE9Bzj14w-cMznI_39E9Z3p0uFoA?type=png](https://mermaid.ink/img/pako:eNp1kc1qhTAQRl9FUiQb8wIpdNO76eKubrmFks1oRg3VSYgjpYjv3lFL_2hnMWQOJwn5sqgmelRWleUSKLAtFs09jqhtoWuYUFfFAa6QA9QDTnpzamheuhxn8pt40-6l13UtS0ddhtQXj6dbR4XUGQg6zEYasTF393KjeSDGnDJKNxzj8I_7hLW5IOSmP9CH9hv_NL-d94d4DVNg84p1EnK4qlIj5hGClySWbadT-6OdsrL02MI8sFOOVkciw8zx8kaNspxnrJQE0fXKtjBMMs3JA-MpgOQwftIE9Bzj14w-cMznI_39E9Z3p0uFoA?type=png)