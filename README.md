# 🤖 HuggingFace_Agent

**HuggingFace Agents 학습 코스 - 완전 실습 가이드**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-orange.svg)](https://huggingface.co/learn/agents-course/)

> **AI 에이전트 개발의 모든 것을 한 곳에서 배우세요!** 🚀

이 프로젝트는 [HuggingFace Agents Course](https://huggingface.co/learn/agents-course/)를 기반으로 한 종합적인 AI 에이전트 학습 리포지토리입니다. SmolAgents, LlamaIndex, LangGraph 등 주요 에이전트 프레임워크를 실습을 통해 마스터할 수 있습니다.

---

## 📚 목차

- [🎯 프로젝트 개요](#-프로젝트-개요)
- [🏗️ 프로젝트 구조](#️-프로젝트-구조)
- [🛠️ 기술 스택](#️-기술-스택)
- [📋 전제 조건](#-전제-조건)
- [🚀 빠른 시작](#-빠른-시작)
- [📖 학습 모듈](#-학습-모듈)
- [🧪 실습 예제](#-실습-예제)
- [🔧 테스트 실행](#-테스트-실행)
- [📄 라이선스](#-라이선스)

---

## 🎯 프로젝트 개요

### 🌟 핵심 특징

- **🎓 체계적 학습**: Unit별 단계적 AI 에이전트 개발 학습
- **🛠️ 실무 중심**: 실제 프로젝트에 적용 가능한 실습 코드
- **🔄 다중 프레임워크**: SmolAgents, LlamaIndex, LangGraph 비교 학습
- **📊 성능 평가**: GAIA 벤치마크를 통한 에이전트 성능 측정
- **🎯 맞춤형 도구**: 각 프레임워크별 특화된 도구 및 에이전트 구현

### 🎯 학습 목표

1. **AI 에이전트 기본 개념 이해**: LLM, Tool, ReAct 패턴 등
2. **다양한 프레임워크 활용**: 각 프레임워크의 장단점과 사용 사례
3. **실무 애플리케이션 개발**: RAG, 멀티에이전트 시스템, 워크플로우 구축
4. **성능 최적화**: 에이전트 성능 측정 및 개선 방법론

---

## 🏗️ 프로젝트 구조

```
HuggingFace_Agent/
├── 📁 docs/                           # 📚 학습 문서 및 이론
│   ├── Intro to Agent.md              # Unit 1: AI 에이전트 기본 개념
│   ├── Smolagents.md                  # Unit 2.1: SmolAgents 프레임워크
│   ├── LLamaIndex.md                  # Unit 2.2: LlamaIndex 프레임워크
│   ├── LangGraph.md                   # Unit 2.3: LangGraph 프레임워크
├── 📁 src/                            # 💻 실습 코드 및 구현
│   ├── framework/                     # 🔧 프레임워크별 구현체
│   │   ├── agents/                    # 🤖 에이전트 구현
│   │   │   ├── smola_code_agents.py           # SmolAgents CodeAgent
│   │   │   ├── smola_tool_calling_agents.py   # SmolAgents ToolCallingAgent
│   │   │   ├── llama_agents.py                # LlamaIndex Agent
│   │   │   └── langg_agent.py                 # LangGraph Agent
│   │   ├── tools/                     # 🛠️ 도구 구현
│   │   │   ├── smola_tools.py                 # SmolAgents 도구
│   │   │   ├── llama_tools.py                 # LlamaIndex 도구
│   │   │   └── langgraph_tools.py             # LangGraph 도구
│   │   ├── rag/                       # 🔍 RAG 시스템 구현
│   │   │   ├── smola_retrieval_agents.py      # SmolAgents RAG
│   │   │   └── llama_components.py            # LlamaIndex RAG
│   │   └── workflow/                  # 🔄 워크플로우 구현
│   │       ├── smola_multiagent_notebook.py   # SmolAgents 멀티에이전트
│   │       ├── llama_workflows.py             # LlamaIndex 워크플로우
│   │       └── langg_mail_sorting.py          # LangGraph 메일 분류
│   ├── final_assessment/              # 🎯 최종 평가 시스템
│   │   ├── app.py                     # Gradio 평가 앱
│   │   ├── agents.py                  # GAIA 벤치마크 에이전트
│   │   ├── tools.py                   # 평가용 도구
│   │   └── requirements.txt           # 의존성 목록
│   ├── function_calling/              # 📞 함수 호출 최적화
│   │   └── finetune_calling.py        # 함수 호출 파인튜닝
│   └── test/                          # 🧪 테스트 코드
│       ├── test_agents.py             # 에이전트 테스트
│       ├── test_rag.py                # RAG 시스템 테스트
│       ├── test_tools.py              # 도구 테스트
└──     └── test_workflow.py           # 워크플로우 테스트
```

---

## 🛠️ 기술 스택

### 🎯 핵심 프레임워크

| 프레임워크 | 설명 | 특징 |
|-----------|------|------|
| **SmolAgents** | 간단하고 강력한 AI 에이전트 프레임워크 | 코드 기반 액션, 유연한 LLM 지원 |
| **LlamaIndex** | RAG 및 데이터 중심 에이전트 도구킷 | 고급 문서 파싱, 워크플로우 시스템 |
| **LangGraph** | 제어 가능한 에이전트 워크플로우 프레임워크 | 그래프 기반 제어 흐름, 상태 관리 |

### 🔧 개발 도구 및 라이브러리

- **Python 3.11+**: 메인 프로그래밍 언어
- **Gradio**: 웹 기반 인터페이스 구축
- **ChromaDB**: 벡터 데이터베이스
- **OpenAI/Anthropic APIs**: LLM 서비스 연동
- **Google Generative AI**: Gemini 모델 활용
- **LangChain**: 추가 도구 및 유틸리티

### 📊 데이터 처리 및 분석

- **Pandas**: 데이터 분석 및 처리
- **NumPy**: 수치 계산
- **Beautiful Soup**: 웹 스크래핑
- **PyPDF/DocX2txt**: 문서 처리
- **Selenium**: 브라우저 자동화

---

## 📋 전제 조건

### 🔧 시스템 요구사항

- **Python 3.11 이상**
- **Git** (버전 관리)
- **최소 8GB RAM** (LLM 모델 실행용)
- **안정적인 인터넷 연결** (API 호출용)

### 🔑 API 키 설정

다음 서비스의 API 키가 필요합니다:

```bash
# 환경변수 설정 (.env 파일 생성)
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

---

## 🚀 빠른 시작

### 1️⃣ 환경 설정

```bash
# 저장소 클론
git clone https://github.com/your-username/HuggingFace_Agent.git
cd HuggingFace_Agent

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는 .venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
pip install -r src/final_assessment/requirements.txt # for assessment
```

### 2️⃣ 환경변수 설정

```bash
# .env 파일 생성
cp .env.example .env
# .env 파일을 편집하여 API 키 설정
```

### 3️⃣ 기본 예제 실행

```bash
# SmolAgents 코드 에이전트 예제
cd src/framework/agents
python smola_code_agents.py

# LlamaIndex RAG 시스템 예제
cd ../rag
python llama_components.py

# LangGraph 워크플로우 예제
cd ../workflow
python langg_mail_sorting.py
```

### 4️⃣ 최종 평가 시스템 실행

```bash
# GAIA 벤치마크 평가 앱 실행
cd src/final_assessment
python app.py
```

브라우저에서 `http://localhost:7860`으로 접속하여 에이전트 성능을 평가할 수 있습니다.

---

## 📖 학습 모듈

### 📚 이론 학습 (docs/)

#### 🎯 Unit 1: AI 에이전트 기본 개념
- **에이전트 정의**: AI 모델의 추론, 계획, 환경 상호작용 능력
- **LLM 기초**: Transformer 아키텍처, 토큰 예측, 채팅 템플릿
- **도구(Tools)**: LLM 능력 확장을 위한 함수 인터페이스
- **ReAct 패턴**: Thought-Action-Observation 사이클

#### 🛠️ Unit 2.1: SmolAgents 프레임워크
- **CodeAgent**: Python 코드 기반 액션 실행
- **ToolCallingAgent**: JSON 기반 도구 호출
- **도구 생성**: `@tool` 데코레이터, `Tool` 클래스 상속
- **RAG 에이전트**: 검색 증강 생성 시스템
- **멀티에이전트**: 협업 에이전트 시스템

#### 🗂️ Unit 2.2: LlamaIndex 프레임워크
- **컴포넌트**: 로딩, 인덱싱, 저장, 쿼리, 평가
- **도구 시스템**: FunctionTool, QueryEngineTool, ToolSpecs
- **에이전트**: Function Calling, ReAct, Custom Agents
- **워크플로우**: 이벤트 기반 단계별 처리

#### 🔄 Unit 2.3: LangGraph 프레임워크
- **상태 관리**: 사용자 정의 상태 객체
- **노드와 엣지**: 함수 노드, 조건부/직접 엣지
- **StateGraph**: 전체 워크플로우 컨테이너
- **제어 흐름**: 예측 가능한 에이전트 동작

### 🛠️ 실습 학습 (src/)

#### 🔧 프레임워크별 구현 (framework/)

**에이전트 구현**:
- `smola_code_agents.py`: SmolAgents CodeAgent 기본 사용법
- `smola_tool_calling_agents.py`: JSON 기반 도구 호출
- `llama_agents.py`: LlamaIndex 에이전트 생성 및 활용
- `langg_agent.py`: LangGraph 상태 기반 에이전트

**도구 개발**:
- `smola_tools.py`: 커스텀 도구 생성 (`@tool`, `Tool` 클래스)
- `llama_tools.py`: FunctionTool, QueryEngineTool 구현
- `langgraph_tools.py`: LangGraph 전용 도구 개발

**RAG 시스템**:
- `smola_retrieval_agents.py`: BM25 기반 검색 에이전트
- `llama_components.py`: VectorStore, QueryEngine 활용

**워크플로우**:
- `smola_multiagent_notebook.py`: 협업 에이전트 시스템
- `llama_workflows.py`: 이벤트 기반 워크플로우
- `langg_mail_sorting.py`: 메일 분류 자동화

#### 🎯 최종 평가 (final_assessment/)

**GAIA 벤치마크 시스템**:
- `app.py`: Gradio 기반 평가 인터페이스
- `agents.py`: 종합 성능 평가용 에이전트
- `tools.py`: 평가 전용 도구 모음

---

## 🧪 실습 예제

### 🤖 SmolAgents 기본 예제

```python
from smolagents import CodeAgent, InferenceClientModel

# 에이전트 초기화
model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct")
agent = CodeAgent(tools=[], model=model)

# 질문 실행
result = agent.run("파이썬으로 피보나치 수열 10개를 계산해주세요.")
print(result)
```

### 🗂️ LlamaIndex RAG 예제

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# 문서 로딩 및 인덱싱
documents = SimpleDirectoryReader("data_test").load_data()
index = VectorStoreIndex.from_documents(documents)

# 쿼리 엔진 생성
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
query_engine = index.as_query_engine(llm=llm)

# 질의 실행
response = query_engine.query("페르소나에 대해 설명해주세요.")
print(response)
```

### 🔄 LangGraph 워크플로우 예제

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    message: str
    processed: bool

def process_message(state: State) -> State:
    return {"message": f"처리됨: {state['message']}", "processed": True}

# 그래프 구성
builder = StateGraph(State)
builder.add_node("process", process_message)
builder.add_edge(START, "process")
builder.add_edge("process", END)

graph = builder.compile()

# 실행
result = graph.invoke({"message": "안녕하세요", "processed": False})
print(result)
```

---

## 🔧 테스트 실행

### 🧪 전체 테스트 실행

```bash
# 가상환경 활성화
source .venv/bin/activate

# 모든 테스트 실행
cd src/test
python -m pytest . -v

# 특정 모듈 테스트
python -m pytest test_agents.py -v
python -m pytest test_rag.py -v
python -m pytest test_tools.py -v
python -m pytest test_workflow.py -v
```

### 📊 개별 프레임워크 테스트

```bash
# SmolAgents 테스트
cd src/framework/agents
python smola_code_agents.py

# LlamaIndex 테스트
cd ../rag
python llama_components.py

# LangGraph 테스트
cd ../workflow
python langg_mail_sorting.py
```

### 🎯 성능 벤치마크

```bash
# GAIA 벤치마크 실행
cd src/final_assessment
python app.py

# 브라우저에서 http://localhost:7860 접속
# HuggingFace 계정으로 로그인 후 평가 실행
```

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 📚 참고 자료

### 📖 공식 문서

- [HuggingFace Agents Course](https://huggingface.co/learn/agents-course/)
- [SmolAgents Documentation](https://huggingface.co/docs/smolagents/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

### 🎓 추가 학습 자료

- [Transformer 모델 이해하기](https://huggingface.co/course/chapter1/1)
- [RAG 시스템 구축 가이드](https://python.langchain.com/docs/tutorials/rag/)
- [AI 에이전트 디자인 패턴](https://www.deeplearning.ai/short-courses/)

---

## 🌟 인용 및 크레딧

이 프로젝트는 HuggingFace의 공식 Agents Course를 기반으로 합니다:

```bibtex
@misc{huggingface-agents-course,
  title = {HuggingFace Agents Course},
  author = {HuggingFace Team},
  year = {2024},
  url = {https://huggingface.co/learn/agents-course/},
}
```

**원본 코스**: [HuggingFace Agents Course](https://huggingface.co/learn/agents-course/)