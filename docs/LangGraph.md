# Unit 2.3 - LangGraph

---

---

## **Module Overview**

### **1️⃣ What is LangGraph, and when to use it?**

### **2️⃣ Building Blocks of LangGraph**

### **3️⃣ Alfred, the mail sorting butler**

### **4️⃣ Alfred, the document Analyst agent**

---

---

# **What is LangGraph ?**

`LangGraph` is a framework developed by [**LangChain**](https://www.langchain.com/) **to manage the control flow of applications that integrate an LLM**.

## **Is LangGraph different from LangChain ?**

LangChain provides a standard interface to interact with models and other components, useful for retrieval, LLM calls and tools calls. The classes from LangChain might be used in LangGraph, but do not **HAVE** to be used.

---

### **Control vs freedom**

When designing AI applications, you face a fundamental trade-off between **control** and **freedom**:

- **Freedom** gives your LLM more room to be creative and tackle unexpected problems.
- **Control** allows you to ensure predictable behavior and maintain guardrails.

Code Agents, like the ones you can encounter in *smolagents*, are very free.

However, this behavior can make them less predictable and less controllable than a regular Agent working with JSON!

`LangGraph` is on the other end of the spectrum, it shines when you need **“Control”** on the execution of your agent.

LangGraph is particularly valuable when you need **Control over your applications**. It gives you the tools to build an application that follows a predictable process while still leveraging the power of LLMs.

Put simply, if your application involves a series of steps that need to be orchestrated in a specific way, with decisions being made at each junction point, **LangGraph provides the structure you need**.

The key scenarios where LangGraph excels include:

- **Multi-step reasoning processes** that need explicit control on the flow
- **Applications requiring persistence of state** between steps
- **Systems that combine deterministic logic with AI capabilities**
- **Workflows that need human-in-the-loop interventions**
- **Complex agent architectures** with multiple components working together

In essence, whenever possible, **as a human**, design a flow of actions based on the output of each action, and decide what to execute next accordingly. In this case, LangGraph is the correct framework for you!

---

## **How does LangGraph work?**

At its core, `LangGraph` uses a directed graph structure to define the flow of your application:

- **Nodes** represent individual processing steps (like calling an LLM, using a tool, or making a decision).
- **Edges** define the possible transitions between steps.
- **State** is user defined and maintained and passed between nodes during execution. When deciding which node to target next, this is the current state that we look at.

---

---

# **Building Blocks of LangGraph**

![](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/Building_blocks.png)

An application in LangGraph starts from an **entrypoint**, and depending on the execution, the flow may go to one function or another until it reaches the END.

![](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/application.png)

## State

**State** is the central concept in LangGraph. It represents all the information that flows through your application.

The state is **User defined**, hence the fields should carefully be crafted to contain all data needed for decision-making process!

> ***💡 Tip: Think carefully about what information your application needs to track between steps.***
> 

---

## Nodes

**Nodes** are python functions. Each node:

- Takes the state as input
- Performs some operation
- Returns updates to the state

---

## Edges

**Edges** connect nodes and define the possible paths through your graph:

Edges can be:

- **Direct**: Always go from node A to node B
- **Conditional**: Choose the next node based on the current state

---

## StateGraph

The **StateGraph** is the container that holds your entire agent workflow:

```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()
```

- visualize

```python
# View
display(Image(graph.get_graph().draw_mermaid_png()))
```

![](https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/unit2/LangGraph/basic_graph.jpeg)

---

---