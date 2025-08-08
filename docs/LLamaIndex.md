# Unit 2.2 - LLamaIndex

---

---

# Intro

LlamaIndex is **a complete toolkit for creating LLM-powered agents over your data using indexes and workflows**. 

three main parts that help build agents in LlamaIndex: 

1. **Components**
2. **Agents and Tools** 
3. **Workflows**

- **Components**: Are the basic building blocks you use in LlamaIndex. These include things like prompts, models, and databases.
- **Tools**: Tools are components that provide specific capabilities like searching, calculating, or accessing external services. They are the building blocks that enable agents to perform tasks.
- **Agents**: Agents are autonomous components that can use tools and make decisions. They coordinate tool usage to accomplish complex goals.
- **Workflows**: Are step-by-step processes that process logic together. Workflows or agentic workflows are a way to structure agentic behaviour without the explicit use of agents.

## What makes LlamaIndex special?

- **Clear Workflow System**: Workflows help break down how agents should make decisions step by step using an event-driven and async-first syntax. This helps you clearly compose and organize your logic.
- **Advanced Document Parsing with LlamaParse**: LlamaParse was made specifically for LlamaIndex, so the integration is seamless, although it is a paid feature.
- **Many Ready-to-Use Components**: LlamaIndex has been around for a while, so it works with lots of other frameworks. This means it has many tested and reliable components, like LLMs, retrievers, indexes, and more.
- **LlamaHub**: is a registry of hundreds of these components, agents, and tools that you can use within LlamaIndex.

---

---

# LlamaHub

**LlamaHub is a registry of hundreds of integrations, agents and tools that you can use within LlamaIndex.**

most of the **installation commands generally follow an easy-to-remember format**:

```python
pip install llama-index-llms-huggingface-api llama-index-embeddings-huggingface

```

---

---

# Components

There are five key stages within RAG, which in turn will be a part of most larger applications you build. These are:

1. **Loading**: this refers to getting your data from where it lives — whether it’s text files, PDFs, another website, a database, or an API — into your workflow. LlamaHub provides hundreds of integrations to choose from.
2. **Indexing**: this means creating a data structure that allows for querying the data. For LLMs, this nearly always means creating vector embeddings. Which are numerical representations of the meaning of the data. Indexing can also refer to numerous other metadata strategies to make it easy to accurately find contextually relevant data based on properties.
3. **Storing**: once your data is indexed you will want to store your index, as well as other metadata, to avoid having to re-index it.
4. **Querying**: for any given indexing strategy there are many ways you can utilize LLMs and LlamaIndex data structures to query, including sub-queries, multi-step queries and hybrid strategies.
5. **Evaluation**: a critical step in any flow is checking how effective it is relative to other strategies, or when you make changes. Evaluation provides objective measures of how accurate, faithful and fast your responses to queries are.

## loading & embedding documents

hree main ways to load data into LlamaIndex:

1. `SimpleDirectoryReader` 
2. `LlamaParse`
3. `LlamaHub`

---

## Querying

Before we can query our index, we need to convert it to a query interface. The most common conversion options are:

1. `as_retriever`: For basic document retrieval, returning a list of `NodeWithScore` objects with similarity scores
2. `as_query_engine`: For single question-answer interactions, returning a written response
3. `as_chat_engine`: For conversational interactions that maintain memory across multiple messages, returning a written response using chat history and indexed context

- response processing
    - Under the hood, the query engine doesn’t only use the LLM to answer the question but also uses a `ResponseSynthesizer` as a strategy to process the response.
        1. `refine`: create and refine an answer by sequentially going through each retrieved text chunk. This makes a separate LLM call per Node/retrieved chunk.
        2. `compact` (default): similar to refining but concatenating the chunks beforehand, resulting in fewer LLM calls.
        3. `tree_summarize`: create a detailed answer by going through each retrieved text chunk and creating a tree structure of the answer.

---

## Evaluation and Observability

Using LLM-as-Judge

LlamaTrace = LangServe

---

---

# Tools

There are **four main types of tools in LlamaIndex**:

1. `FunctionTool`: Convert any **Python function into a tool** that an agent can use. It automatically figures out how the function works.
2. `QueryEngineTool`: A tool that lets agents use query engines. Since agents are built on query engines, they can also use other agents as tools.
3. `Toolspecs`: Sets of tools created by the community, which often include tools for specific **services like Gmail.**
4. `Utility Tools`: Special tools that help handle large amounts of data from other tools.

## FunctionTool

- example
    
    ```python
    from llama_index.core.tools import FunctionTool
    
    def get_weather(location: str) -> str:
        """Useful for getting the weather for a given location."""
        print(f"Getting weather for {location}")
        return f"The weather in {location} is sunny"
    
    tool = FunctionTool.from_defaults(
        get_weather,
        name="my_weather_tool",
        description="Useful for getting the weather for a given location.",
    )
    tool.call("New York")
    ```
    

little different with smolagent and mcp.

---

## QueryEngineTool

- example
    
    ```python
    from llama_index.core import VectorStoreIndex
    from llama_index.core.tools import QueryEngineTool
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore
    
    embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
    
    db = chromadb.PersistentClient(path="./alfred_chroma_db")
    chroma_collection = db.get_or_create_collection("alfred")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    
    llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
    query_engine = index.as_query_engine(llm=llm)
    tool = QueryEngineTool.from_defaults(query_engine, name="some useful name", description="some useful description")
    ```
    

llamaindex’s query engine is like llm is already embedded to retriever in LangChain.

---

## Toolspecs

### **Model Context Protocol (MCP) in LlamaIndex**

LlamaIndex also allows using MCP tools through a [**ToolSpec on the LlamaHub**](https://llamahub.ai/l/tools/llama-index-tools-mcp?from=). You can simply run an MCP server and start using it through the following implementation.

```python
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# We consider there is a mcp server running on 127.0.0.1:8000, or you can use the mcp client to connect to your own mcp server.
mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
mcp_tool = McpToolSpec(client=mcp_client)

# get the agent
agent = await get_agent(mcp_tool)

# create the agent context
agent_context = Context(agent)
```

---

## Utility tools

Oftentimes, directly querying an API **can return an excessive amount of data**, some of which may be irrelevant, overflow the context window of the LLM, or unnecessarily increase the number of tokens that you are using.

1. `OnDemandToolLoader`: **This tool turns any existing LlamaIndex data loader** (BaseReader class) into a tool that an agent can use. The tool can be called with all the parameters needed to trigger `load_data` from the data loader, along with a natural language query string. During execution, we first load data from the data loader, index it (for instance with a vector store), and then query it ‘on-demand’. All three of these steps happen in a single tool call.
2. `LoadAndSearchToolSpec`: The LoadAndSearchToolSpec takes in any existing Tool as input. As a tool spec, it implements `to_tool_list`, and when that function is called, two tools are **returned: a loading tool and then a search tool.** The load Tool execution would call the underlying Tool, and then index the output (by default with a vector index). The search Tool execution would take in a query string as input and call the underlying index.

---

---

# Agents

LlamaIndex supports **three main types of reasoning agents:**

1. `Function Calling Agents` - These work with AI models that can call specific functions.
2. `ReAct Agents` - These can work with any AI that does chat or text endpoint and deal with complex reasoning tasks.
3. `Advanced Custom Agents` - These use more complex methods to deal with more complex tasks and workflows.

simple agent creation

```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool

# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

# initialize llm
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# initialize agent
agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)],
    llm=llm
)
```

**Agents are stateless by default**, however, they can remember past interactions using a `Context` object.

it is like memory in langchain

```python
# stateless
response = await agent.run("What is 2 times 2?")

# remembering state
from llama_index.core.workflow import Context

ctx = Context(agent)

response = await agent.run("My name is Bob.", ctx=ctx)
response = await agent.run("What was my name again?", ctx=ctx)
```

---

## using QueryingEngineTool to make Agentic RAG

It is easy to **wrap `QueryEngine` as a tool** for an agent. When doing so, we need to **define a name and description**. The LLM will use this information to correctly use the tool.

```python
from llama_index.core.tools import QueryEngineTool

query_engine = index.as_query_engine(llm=llm, similarity_top_k=3) # as shown in the Components in LlamaIndex section

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="name",
    description="a specific description",
    return_direct=False,
)
query_engine_agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. "
)
```

---

## **Creating Multi-agent systems**

The `AgentWorkflow` class also directly supports multi-agent systems. By giving each agent a name and description, the system maintains a single active speaker, with each agent having the ability to hand off to another agent.

**Agents in LlamaIndex can also directly be used as tools** for other agents, for more complex and custom scenarios. (like LangGraph)

```python
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)

# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

# Create agent configs
# NOTE: we can use FunctionAgent or ReActAgent here.
# FunctionAgent works for LLMs with a function calling API.
# ReActAgent works for any LLM.
calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about XYZ",
    system_prompt="Use your tool to query a RAG system to answer information about XYZ",
    tools=[query_engine_tool],
    llm=llm
)

# Create and run the workflow
agent = AgentWorkflow(
    agents=[calculator_agent, query_agent], root_agent="calculator"
)

# Run the system
response = await agent.run(user_msg="Can you add 5 and 3?")

```

---

---

# Workflows

![](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/llama-index/workflows.png)

**Workflows offer several key benefits:**

- Clear organization of code into discrete steps
- Event-driven architecture for flexible control flow
- Type-safe communication between steps
- Built-in state management
- Support for both simple and complex agent interactions

We can create a single-step workflow by defining a class that inherits from `Workflow` and decorating your functions with `@step`.

We will also need to add `StartEvent` and `StopEvent`,  (AKA LangGraph Start, End nodes)

- single step example
    
    ```python
    from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
    
    class MyWorkflow(Workflow):
        @step
        async def my_step(self, ev: StartEvent) -> StopEvent:
            # do something here
            return StopEvent(result="Hello, world!")
    
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run()
    ```
    

---

To connect multiple steps, we **create custom events that carry data between steps.**

- connecting multiple steps
    
    ```python
    from llama_index.core.workflow import Event
    
    class ProcessingEvent(Event):
        intermediate_result: str
    
    class MultiStepWorkflow(Workflow):
        @step
        async def step_one(self, ev: StartEvent) -> ProcessingEvent:
            # Process initial data
            return ProcessingEvent(intermediate_result="Step 1 complete")
    
        @step
        async def step_two(self, ev: ProcessingEvent) -> StopEvent:
            # Use the intermediate result
            final_result = f"Finished processing: {ev.intermediate_result}"
            return StopEvent(result=final_result)
    
    w = MultiStepWorkflow(timeout=10, verbose=False)
    result = await w.run()
    result
    ```
    

steps are not **intuitive to me(personal).**

---

- Loops and Branches
    
    ```python
    from llama_index.core.workflow import Event
    import random
    
    class ProcessingEvent(Event):
        intermediate_result: str
    
    class LoopEvent(Event):
        loop_output: str
    
    class MultiStepWorkflow(Workflow):
        @step
        async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
            if random.randint(0, 1) == 0:
                print("Bad thing happened")
                return LoopEvent(loop_output="Back to step one.")
            else:
                print("Good thing happened")
                return ProcessingEvent(intermediate_result="First step complete.")
    
        @step
        async def step_two(self, ev: ProcessingEvent) -> StopEvent:
            # Use the intermediate result
            final_result = f"Finished processing: {ev.intermediate_result}"
            return StopEvent(result=final_result)
    
    w = MultiStepWorkflow(verbose=False)
    result = await w.run()
    result
    ```
    

The type hinting is the most powerful part of workflows because it allows us to create branches, loops, and joins to facilitate more complex workflows.

 **creating a loop** by using the union operator `|`. In the example below, we see that the `LoopEvent` is taken as input for the step and can also be returned as output.

- **Drawing Workflows**
    
    ```python
    from llama_index.utils.workflow import draw_all_possible_flows
    
    w = ... # as defined in the previous section
    draw_all_possible_flows(w, "flow.html")
    ```
    

---