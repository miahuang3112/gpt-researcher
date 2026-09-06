<div align="center" id="top">

<img src="https://github.com/assafelovic/gpt-researcher/assets/13554167/20af8286-b386-44a5-9a83-3be1365139c3" alt="Logo" width="80">

####

[![Website](https://img.shields.io/badge/Official%20Website-gptr.dev-teal?style=for-the-badge&logo=world&logoColor=white&color=0891b2)](https://gptr.dev)
[![Documentation](https://img.shields.io/badge/Documentation-DOCS-f472b6?logo=googledocs&logoColor=white&style=for-the-badge)](https://docs.gptr.dev)
[![Discord](https://img.shields.io/discord/1127851779011391548?logo=discord&logoColor=white&label=Discord&color=34b76a&style=for-the-badge)](https://discord.gg/QgZXvJAccX)


[![PyPI version](https://img.shields.io/pypi/v/gpt-researcher?logo=pypi&logoColor=white&style=flat)](https://badge.fury.io/py/gpt-researcher)
![GitHub Release](https://img.shields.io/github/v/release/assafelovic/gpt-researcher?style=flat&logo=github)
[![Open In Colab](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=grey&color=yellow&label=%20&style=flat&logoSize=40)](https://colab.research.google.com/github/assafelovic/gpt-researcher/blob/master/docs/docs/examples/pip-run.ipynb)
[![Docker Image Version](https://img.shields.io/docker/v/elestio/gpt-researcher/latest?arch=amd64&style=flat&logo=docker&logoColor=white&color=1D63ED)](https://hub.docker.com/r/gptresearcher/gpt-researcher)
[![Skill](https://img.shields.io/badge/Claude%20Skill-skills.sh-blueviolet?style=flat&logo=anthropic&logoColor=white)](https://skills.sh/assafelovic/gpt-researcher/gpt-researcher)
[![Twitter Follow](https://img.shields.io/twitter/follow/assaf_elovic?style=social)](https://twitter.com/assaf_elovic)

[English](README.md) | [中文](README-zh_CN.md) | [日本語](README-ja_JP.md) | [한국어](README-ko_KR.md)

</div>

# 🔎 GPT Researcher

**GPT Researcher the first open deep research agent designed for both web and local research on any given task.** 

The agent produces detailed, factual, and unbiased research reports with citations. GPT Researcher provides a full suite of customization options to create tailor made and domain specific research agents. Inspired by the recent [Plan-and-Solve](https://arxiv.org/abs/2305.04091) and [RAG](https://arxiv.org/abs/2005.11401) papers, GPT Researcher addresses misinformation, speed, determinism, and reliability by offering stable performance and increased speed through parallelized agent work.

**Our mission is to empower individuals and organizations with accurate, unbiased, and factual information through AI.**

## Academic Paper Evaluation Assistant

This repository also contains a portfolio-focused computer-science paper
evaluation product. It accepts a PDF, runs an asynchronous evidence-grounded
nine-dimension assessment, and exposes the same persisted jobs through a web
workspace, a versioned REST API, MCP tools, and the `academic-paper-review`
Skill.

> **Automated academic assessment — not peer review.** The current scores have
> been measured against a versioned ChatGPT reference, not an independent
> human-scored evaluation set, and must not be used for publication, hiring, or
> funding decisions.

### Verified capabilities

- secure bounded PDF upload with signature, structure, page, active-content and
  prompt-injection checks;
- Alembic-managed MySQL persistence and transactional outbox publication;
- Redis Streams delivery with ACK, stale-message recovery, bounded retries and
  a dead-letter queue;
- checkpointed nine-dimension evaluation with persisted progress, retry and
  cooperative cancellation;
- responsive `/academic` submission, history, progress, evidence and report UI;
- stable-ID SSE with reconnect and polling fallback;
- sanitized `academic-evaluation.v1` API responses and versioned JSON/PDF
  exports;
- seven stdio MCP tools plus the repository `academic-paper-review` Skill;
- structured logs, correlation IDs, readiness checks, retention/deletion rules,
  Prometheus metrics with optional Grafana/alerts, and a path-filtered Academic
  CI gate;
- opt-in per-paper source caps for future multi-paper retrieval experiments,
  disabled by default until benchmark evidence supports activation.

As of 2026-09-05, the fast Academic suite passes **363 tests** with one opt-in
real-Ollama E2E test skipped. The Next.js production build, desktop/mobile web
checks, Skill validation, isolated MCP subprocess demo, migrations and deployed
container health checks also pass. These are engineering verification counts,
not model-quality metrics.

### Start the Academic stack

Requirements: Docker, Docker Compose, a local Ollama server reachable from
Docker, and the configured `qwen3.5:9b` model for real evaluations.

```bash
docker compose -f compose.academic.yml build academic-api
docker compose -f compose.academic.yml up -d --wait
curl http://127.0.0.1:8001/health/ready
```

The migration service runs to completion before API/worker startup. MySQL,
Redis and the API are bound to loopback ports by default. For a shared or remote
deployment, switch from `trusted_local` to `api_key` authentication and provide
`ACADEMIC_API_KEY` outside version control.

The RET-10B v1 retrieval policy freezes Hybrid as the production default using
development-only selection. The stack defaults are:

```bash
export ACADEMIC_HYBRID_RETRIEVAL_ENABLED=true
export ACADEMIC_HYBRID_CANDIDATE_TOP_K=20
export ACADEMIC_HYBRID_RRF_K=100
export ACADEMIC_EMBEDDING_MODEL=qwen3-embedding:4b
docker compose -f compose.academic.yml up -d --wait
```

One BM25/Vector/Hybrid retriever set is created per paper evaluation. All nine
dimension queries reuse the same in-memory document vector index. With the flag
set to `false`, the established section-priority selector remains available for
ablations and rollback.

The one-time two-paper held-out run found that frozen Hybrid v1 underperformed
BM25 (Recall@10 `0.1824` versus `0.2555`; nDCG@10 `0.3462` versus `0.4338`).
The policy was not changed using test results. A default change requires new
development data and a separately versioned policy. Labels have mixed
ChatGPT/project-owner provenance, so these are project diagnostics rather than
independent human-gold performance estimates.

RET-11A also compared the actual four-chunk prompt inputs: section-only beat
frozen Hybrid on held-out Recall@4 (`0.1846` versus `0.0755`) and nDCG@4
(`0.6066` versus `0.3930`), while avoiding approximately `364s` of two-paper
index construction. RET-11A alone is an automatic evidence-selection result,
not a claim about final scoring quality. RET-11B subsequently reused those
exact held-out inputs with `qwen3.5:9b` and
the versioned ChatGPT reference. Section-only had lower dimension MAE (`0.471`
versus `0.588`) and lower weighted overall-score absolute error (`5.61` versus
`12.34` points). This is a descriptive two-paper AI-reference comparison, not
human agreement or permission to retune on the test split.

Across the five-paper, 45-dimension ChatGPT reference, EVAL-02A produced 42
scorable results with zero terminal failures, MAE `0.500`, and `92.86%`
within-one-point agreement. EVAL-02B's 28 controlled held-out cases measured
citation four-class accuracy `81.25%` (macro-F1 `78.18%`) and contradiction
six-class accuracy `83.33%` (macro-F1 `77.78%`); the production binary citation
decision and `0.95` contradiction-warning policy both had F1 `100%` on their
small controlled sets.

EVAL-03 then ran 81 development-only prompt/retriever evaluations. Removing
numeric anchors worsened MAE from `0.304348` to `0.521739`. Section-only
improved MAE to `0.28`, but its `0.024348` gain missed the predeclared `0.1`
adoption threshold, so the anchored Hybrid production configuration remained
frozen. These are small-sample agreement measurements against one AI annotator,
not human-quality or generalisation claims.

Run the web workspace separately:

```bash
cd frontend/nextjs
npm install
npm run dev
```

Open `http://localhost:3000/academic`. The Next.js rewrite sends Academic calls
to `http://127.0.0.1:8001` unless `ACADEMIC_API_URL` is overridden.

The project [.mcp.json](.mcp.json) registers `academic-evaluation`. MCP clients
started from the repository root can launch it with:

```bash
python3 -m backend.academic_mcp
```

The MCP process accepts `ACADEMIC_API_BASE_URL`, `ACADEMIC_API_KEY`, and
`ACADEMIC_MCP_TIMEOUT_SECONDS`. `submit_evaluation` uses a PDF path visible to
the API and allowed by `ACADEMIC_PDF_IMPORT_ROOTS`.

### Verify

```bash
python -m pip install -r requirements-academic.txt \
  "pytest>=8,<10" "pytest-asyncio>=0.24,<2" "PyYAML>=6,<7"
python -m pytest tests/academic
python -m evals.verify_academic_benchmarks
python scripts/demo_academic_mcp.py

cd frontend/nextjs
npm run build
```

The deterministic benchmark check validates 11 pinned hashes and 15
zero-failure/data-boundary invariants without model or network access. A weekly
and manually dispatchable CI workflow re-runs eight real Ollama retrieval and
quality reports with checkpoints and 30-day artifacts. Run the same suite
locally against an existing Ollama service with:

```bash
./scripts/run_academic_benchmarks.sh artifacts/academic-benchmarks
```

The real Ollama E2E remains opt-in because it is slow and depends on local model
availability:

```bash
RUN_ACADEMIC_E2E=1 \
ACADEMIC_E2E_PDF_PATH=/absolute/path/to/paper.pdf \
python -m pytest tests/academic/test_academic_api_e2e.py -v -s
```

Start the optional observability profile and open Grafana at
`http://127.0.0.1:3002` (development login `admin` / `academic_dev`):

```bash
./scripts/academic_stack.sh observability-up
curl http://127.0.0.1:8001/metrics
```

Prometheus is loopback-bound at `http://127.0.0.1:9090`. Replace the development
Grafana password before any shared deployment.

Architecture and reproducible evidence:

- [Academic architecture and lifecycle](docs/academic_architecture.md)
- [API → queue → worker → database demo](docs/demos/api_queue_worker_database.md)
- [Web → MCP → Skill demo](docs/demos/web_mcp_skill.md)
- [Security and deployment boundary](docs/academic_security.md)
- [Canonical implementation TODO](ACADEMIC_TODO.md)
- [Versioned resume evidence and bullets](docs/academic_resume_bullets.md)

### Known limitations

- The retrieval set has mixed ChatGPT/project-owner review provenance and is not
  an independent human-labelled benchmark. EVAL-01 provides a separate
  five-paper, 45-label ChatGPT paper-quality reference set; it is not human peer
  review or ground truth. Score-correlation, MAE, and ablation results must be
  described as agreement with that versioned AI reference.
- External scholarly evidence can be enabled for novelty/citation scoring with
  `ACADEMIC_EXTERNAL_EVIDENCE_ENABLED=true`, `OPENALEX_API_KEY`, and
  `CROSSREF_MAILTO`. It uses OpenAlex-first/Crossref-fallback discovery,
  target/future-work filtering, checkpointed provenance, separate external IDs,
  and graceful failure. It is not an exhaustive literature review, and remains
  disabled by default so offline evaluations stay deterministic.
- The optional per-source cap is implemented but deliberately disabled until a
  multi-paper benchmark demonstrates a benefit and checks recall trade-offs.
- Real evaluation latency depends on the local Ollama model and current queue;
  health checks prove reachability, not generation latency.
- The Academic frontend is production-buildable but is not a service in
  `compose.academic.yml`; it is started separately.
- Default `trusted_local` authentication is suitable only for a loopback-bound
  single-user environment. Shared deployments must use API-key mode and an
  appropriate network/TLS boundary.
- Outputs are automated screening artifacts. Evidence gaps, `review_required`
  verification, and high-impact decisions require qualified human review.

## Why GPT Researcher?

- Objective conclusions for manual research can take weeks, requiring vast resources and time.
- LLMs trained on outdated information can hallucinate, becoming irrelevant for current research tasks.
- Current LLMs have token limitations, insufficient for generating long research reports.
- Limited web sources in existing services lead to misinformation and shallow results.
- Selective web sources can introduce bias into research tasks.

## Demo
<a href="https://www.youtube.com/watch?v=f60rlc_QCxE" target="_blank" rel="noopener">
  <img src="https://github.com/user-attachments/assets/ac2ec55f-b487-4b3f-ae6f-b8743ad296e4" alt="Demo video" width="800" target="_blank" />
</a>

## Install as Claude Skill

Extend Claude's deep research capabilities by installing GPT Researcher as a [Claude Skill](https://skills.sh/assafelovic/gpt-researcher/gpt-researcher):

```bash
npx skills add assafelovic/gpt-researcher
```

Once installed, Claude can leverage GPT Researcher's deep research capabilities directly within your conversations.

## Architecture

The core idea is to utilize 'planner' and 'execution' agents. The planner generates research questions, while the execution agents gather relevant information. The publisher then aggregates all findings into a comprehensive report.

<div align="center">
<img align="center" height="600" src="https://github.com/assafelovic/gpt-researcher/assets/13554167/4ac896fd-63ab-4b77-9688-ff62aafcc527">
</div>

Steps:
* Create a task-specific agent based on a research query.
* Generate questions that collectively form an objective opinion on the task.
* Use a crawler agent for gathering information for each question.
* Summarize and source-track each resource.
* Filter and aggregate summaries into a final research report.

## Tutorials
 - [How it Works](https://docs.gptr.dev/blog/building-gpt-researcher)
 - [How to Install](https://www.loom.com/share/04ebffb6ed2a4520a27c3e3addcdde20?sid=da1848e8-b1f1-42d1-93c3-5b0b9c3b24ea)
 - [Live Demo](https://www.loom.com/share/6a3385db4e8747a1913dd85a7834846f?sid=a740fd5b-2aa3-457e-8fb7-86976f59f9b8)

## Features

- 📝 Generate detailed research reports using web and local documents.
- 🖼️ Smart image scraping and filtering for reports.
- 🍌 **AI-generated inline images** using Google Gemini (Nano Banana) for visual illustrations.
- 📜 Generate detailed reports exceeding 2,000 words.
- 🌐 Aggregate over 20 sources for objective conclusions.
- 🖥️ Frontend available in lightweight (HTML/CSS/JS) and production-ready (NextJS + Tailwind) versions.
- 🔍 JavaScript-enabled web scraping.
- 📂 Maintains memory and context throughout research.
- 📄 Export reports to PDF, Word, and other formats.

## 📖 Documentation

See the [Documentation](https://docs.gptr.dev/docs/gpt-researcher/getting-started) for:
- Installation and setup guides
- Configuration and customization options
- How-To examples
- Full API references

## ⚙️ Getting Started

### Installation

1. Install Python 3.11 or later. [Guide](https://www.tutorialsteacher.com/python/install-python).
2. Clone the project and navigate to the directory:

    ```bash
    git clone https://github.com/assafelovic/gpt-researcher.git
    cd gpt-researcher
    ```

3. Set up API keys by exporting them or storing them in a `.env` file.

    ```bash
    export OPENAI_API_KEY={Your OpenAI API Key here}
    export TAVILY_API_KEY={Your Tavily API Key here}
    ```

    (Optional) For enhanced tracing and observability, you can also set:
    
    ```bash
    # export LANGCHAIN_TRACING_V2=true
    # export LANGCHAIN_API_KEY={Your LangChain API Key here}
    ```

    For custom OpenAI-compatible APIs (e.g., local models, other providers), you can also set:
    
    ```bash
    export OPENAI_BASE_URL={Your custom API base URL here}
    ```

4. Install dependencies and start the server:

    ```bash
    pip install -r requirements.txt
    python -m uvicorn main:app --reload
    ```

Visit [http://localhost:8000](http://localhost:8000) to start.

For other setups (e.g., Poetry or virtual environments), check the [Getting Started page](https://docs.gptr.dev/docs/gpt-researcher/getting-started).

## Run as PIP package
```bash
pip install gpt-researcher

```
### Example Usage:
```python
...
from gpt_researcher import GPTResearcher

query = "why is Nvidia stock going up?"
researcher = GPTResearcher(query=query)
# Conduct research on the given query
research_result = await researcher.conduct_research()
# Write the report
report = await researcher.write_report()
...
```

**For more examples and configurations, please refer to the [PIP documentation](https://docs.gptr.dev/docs/gpt-researcher/gptr/pip-package) page.**

### 🔧 MCP Client
GPT Researcher supports MCP integration to connect with specialized data sources like GitHub repositories, databases, and custom APIs. This enables research from data sources alongside web search.

```bash
export RETRIEVER=tavily,mcp  # Enable hybrid web + MCP research
```

```python
from gpt_researcher import GPTResearcher
import asyncio
import os

async def mcp_research_example():
    # Enable MCP with web search
    os.environ["RETRIEVER"] = "tavily,mcp"
    
    researcher = GPTResearcher(
        query="What are the top open source web research agents?",
        mcp_configs=[
            {
                "name": "github",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN")}
            }
        ]
    )
    
    research_result = await researcher.conduct_research()
    report = await researcher.write_report()
    return report
```

> For comprehensive MCP documentation and advanced examples, visit the [MCP Integration Guide](https://docs.gptr.dev/docs/gpt-researcher/retrievers/mcp-configs).

## 🍌 Inline Image Generation

GPT Researcher can automatically generate and embed AI-created illustrations in your research reports using Google's Gemini models (Nano Banana).

```bash
# Enable in your .env file
IMAGE_GENERATION_ENABLED=true
GOOGLE_API_KEY=your_google_api_key
IMAGE_GENERATION_MODEL=models/gemini-2.5-flash-image
```

When enabled, the system will:
1. Analyze your research context to identify visualization opportunities
2. Pre-generate 2-3 relevant images during the research phase
3. Embed them inline as the report is written

Images are generated with dark-mode styling that matches the GPT Researcher UI, featuring professional infographic aesthetics with teal accents.

[Learn more about Image Generation](https://docs.gptr.dev/docs/gpt-researcher/gptr/image_generation) in our documentation.

## ✨ Deep Research

GPT Researcher now includes Deep Research - an advanced recursive research workflow that explores topics with agentic depth and breadth. This feature employs a tree-like exploration pattern, diving deeper into subtopics while maintaining a comprehensive view of the research subject.

- 🌳 Tree-like exploration with configurable depth and breadth
- ⚡️ Concurrent processing for faster results
- 🤝 Smart context management across research branches
- ⏱️ Takes ~5 minutes per deep research
- 💰 Costs ~$0.4 per research (using `o3-mini` on "high" reasoning effort)

[Learn more about Deep Research](https://docs.gptr.dev/docs/gpt-researcher/gptr/deep_research) in our documentation.

## Run with Docker

> **Step 1** - [Install Docker](https://docs.gptr.dev/docs/gpt-researcher/getting-started/getting-started-with-docker)

> **Step 2** - Clone the '.env.example' file, add your API Keys to the cloned file and save the file as '.env'

> **Step 3** - Within the docker-compose file comment out services that you don't want to run with Docker.

```bash
docker-compose up --build
```

If that doesn't work, try running it without the dash:
```bash
docker compose up --build
```

> **Step 4** - By default, if you haven't uncommented anything in your docker-compose file, this flow will start 2 processes:
 - the Python server running on localhost:8000<br>
 - the React app running on localhost:3000<br>

Visit localhost:3000 on any browser and enjoy researching!


## 📄 Research on Local Documents

You can instruct the GPT Researcher to run research tasks based on your local documents. Currently supported file formats are: PDF, plain text, CSV, Excel, Markdown, PowerPoint, and Word documents.

Step 1: Add the env variable `DOC_PATH` pointing to the folder where your documents are located.

```bash
export DOC_PATH="./my-docs"
```

Step 2: 
 - If you're running the frontend app on localhost:8000, simply select "My Documents" from the "Report Source" Dropdown Options.
 - If you're running GPT Researcher with the [PIP package](https://docs.tavily.com/guides/gpt-researcher/gpt-researcher#pip-package), pass the `report_source` argument as "local" when you instantiate the `GPTResearcher` class [code sample here](https://docs.gptr.dev/docs/gpt-researcher/context/tailored-research).


## 🤖 MCP Server

We've moved our MCP server to a dedicated repository: [gptr-mcp](https://github.com/assafelovic/gptr-mcp).

The GPT Researcher MCP Server enables AI applications like Claude to conduct deep research. While LLM apps can access web search tools with MCP, GPT Researcher MCP delivers deeper, more reliable research results.

Features:
- Deep research capabilities for AI assistants
- Higher quality information with optimized context usage
- Comprehensive results with better reasoning for LLMs
- Claude Desktop integration

For detailed installation and usage instructions, please visit the [official repository](https://github.com/assafelovic/gptr-mcp).


## 👪 Multi-Agent Assistant
As AI evolves from prompt engineering and RAG to multi-agent systems, we're excited to introduce multi-agent assistants built with [LangGraph](https://python.langchain.com/v0.1/docs/langgraph/) and [AG2](https://github.com/ag2ai/ag2).

By using multi-agent frameworks, the research process can be significantly improved in depth and quality by leveraging multiple agents with specialized skills. Inspired by the recent [STORM](https://arxiv.org/abs/2402.14207) paper, this project showcases how a team of AI agents can work together to conduct research on a given topic, from planning to publication.

An average run generates a 5-6 page research report in multiple formats such as PDF, Docx and Markdown.

Check it out [here](https://github.com/assafelovic/gpt-researcher/tree/master/multi_agents) or head over to our documentation for [LangGraph](https://docs.gptr.dev/docs/gpt-researcher/multi_agents/langgraph) and [AG2](https://docs.gptr.dev/docs/gpt-researcher/multi_agents/ag2) for more information.

## 🔍 Observability

GPT Researcher supports **LangSmith** for enhanced tracing and observability, making it easier to debug and optimize complex multi-agent workflows.

To enable tracing:
1. Set the following environment variables:
   ```bash
   export LANGCHAIN_TRACING_V2=true
   export LANGCHAIN_API_KEY=your_api_key
   export LANGCHAIN_PROJECT="gpt-researcher"
   ```
2. Run your research tasks as usual. All LangGraph-based agent interactions will be automatically traced and visualized in your LangSmith dashboard.

## 🖥️ Frontend Applications

GPT-Researcher now features an enhanced frontend to improve the user experience and streamline the research process. The frontend offers:

- An intuitive interface for inputting research queries
- Real-time progress tracking of research tasks
- Interactive display of research findings
- Customizable settings for tailored research experiences

Two deployment options are available:
1. A lightweight static frontend served by FastAPI
2. A feature-rich NextJS application for advanced functionality

For detailed setup instructions and more information about the frontend features, please visit our [documentation page](https://docs.gptr.dev/docs/gpt-researcher/frontend/introduction).

## 🚀 Contributing
We highly welcome contributions! Please check out [contributing](https://github.com/assafelovic/gpt-researcher/blob/master/CONTRIBUTING.md) if you're interested.

Please check out our [roadmap](https://trello.com/b/3O7KBePw/gpt-researcher-roadmap) page and reach out to us via our [Discord community](https://discord.gg/QgZXvJAccX) if you're interested in joining our mission.
<a href="https://github.com/assafelovic/gpt-researcher/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=assafelovic/gpt-researcher&max=1000" />
</a>
## ✉️ Support / Contact us
- [Community Discord](https://discord.gg/spBgZmm3Xe)
- Author Email: assaf.elovic@gmail.com

## 🛡 Disclaimer

This project, GPT Researcher, is an experimental application and is provided "as-is" without any warranty, express or implied. We are sharing codes for academic purposes under the Apache 2 license. Nothing herein is academic advice, and NOT a recommendation to use in academic or research papers.

Our view on unbiased research claims:
1. The main goal of GPT Researcher is to reduce incorrect and biased facts. How? We assume that the more sites we scrape the less chances of incorrect data. By scraping multiple sites per research, and choosing the most frequent information, the chances that they are all wrong is extremely low.
2. We do not aim to eliminate biases; we aim to reduce it as much as possible. **We are here as a community to figure out the most effective human/llm interactions.**
3. In research, people also tend towards biases as most have already opinions on the topics they research about. This tool scrapes many opinions and will evenly explain diverse views that a biased person would never have read.

---

<p align="center">
<a href="https://star-history.com/#assafelovic/gpt-researcher">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=assafelovic/gpt-researcher&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=assafelovic/gpt-researcher&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=assafelovic/gpt-researcher&type=Date" />
  </picture>
</a>
</p>


<p align="right">
  <a href="#top">⬆️ Back to Top</a>
</p>
