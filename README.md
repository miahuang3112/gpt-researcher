# Academic Paper Evaluation Agent

面向计算机科学论文的自动化评估系统。用户上传 PDF 后，系统异步完成论文解析、证据检索、九维评分、引用核验和结果导出；同一能力同时通过 Web、REST API、MCP tools 和 Agent Skill 提供。

> 本项目提供辅助评审，不替代同行评议、录用决定或其他高影响决策。评测中的参考标签由 ChatGPT 生成或复核，不是独立人类金标准。

## 核心能力

- 九维评分：novelty、methodological rigor、experiments、clarity、reproducibility、impact、citation quality、related work、overall assessment。
- 可追踪证据：BM25、向量检索和 RRF Hybrid，输出引用到具体 PDF chunk。
- Agent 接口：7 个 MCP tools，以及 `skills/academic-paper-review` 中的可复用 Skill。
- 异步任务：FastAPI + MySQL + Redis Streams，包含事务 Outbox、消费幂等、checkpoint、`XAUTOCLAIM`、有限重试和 DLQ。
- 安全边界：PDF 类型/大小校验、受管存储、API key、路径与错误脱敏、prompt-injection 防护。
- 可观测性：Prometheus 指标、告警规则和 Grafana dashboard。
- 产品界面：PDF 上传、任务列表、SSE 进度、重试/取消、证据查看、JSON/PDF 导出。

## 架构

```text
Next.js / MCP Client
        │
        ▼
FastAPI REST API ── MySQL (jobs, papers, results, outbox)
        │
        ├── SSE progress
        └── Redis Stream ── Worker ── Ollama
                                ├── PDF chunks / checkpoints
                                └── optional OpenAlex → Crossref evidence
```

完整设计、状态机与恢复流程见 [docs/academic_architecture.md](docs/academic_architecture.md)。

## 快速启动

要求：Docker Desktop、Docker Compose，以及本机 [Ollama](https://ollama.com/)。

```bash
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:4b
./scripts/academic_stack.sh up
```

启动后：

- API 文档：`http://127.0.0.1:8001/docs`
- API readiness：`http://127.0.0.1:8001/health/ready`
- MySQL：`127.0.0.1:3307`
- Redis：`127.0.0.1:6380`

启动网页：

```bash
cd frontend/nextjs
npm ci
npm run dev
```

打开 `http://127.0.0.1:3000`。Next.js 会把 `/api/academic/*` 转发到 Academic API。

## MCP 与 Skill

仓库根目录的 `.mcp.json` 已配置本地 stdio MCP server。API 启动后，支持：

- `submit_evaluation`
- `get_evaluation_status`
- `get_evaluation_result`
- `get_evaluation_dimensions`
- `get_evaluation_evidence`
- `retry_evaluation`
- `cancel_evaluation`

Skill 入口位于 `skills/academic-paper-review/SKILL.md`，规定 Agent 如何提交论文、轮询、读取证据并明确人工复核边界。

## 测试与评测

```bash
python3 -m pip install -r requirements-academic.txt pytest pytest-asyncio pyyaml
PYTHONPATH=. python3 -m pytest tests/academic
PYTHONPATH=. python3 -m evals.verify_academic_benchmarks
```

版本化证据包含检索 Recall/nDCG、评分 MAE/一致率、引用核验和矛盾检测。真实 PDF 语料不提交到公开仓库；仓库只保留合成示例、协议、哈希、汇总和不含论文正文的报告。完整 Ollama benchmark 需在本地准备受保护语料后手动运行：

```bash
./scripts/run_academic_benchmarks.sh artifacts/academic-benchmarks
```

指标与诚实表述见 [docs/academic_resume_bullets.md](docs/academic_resume_bullets.md)，安全设计见 [docs/academic_security.md](docs/academic_security.md)。

## 目录

```text
backend/academic_api         REST API、SQL、Redis Worker
backend/academic_mcp         MCP stdio server 与 tools
gpt_researcher/academic     论文解析、检索、评分与核验核心
frontend/nextjs             Academic Web 工作台
skills/academic-paper-review Agent Skill
evals                       数据协议、评测程序与版本化结果
observability               Prometheus / Grafana
tests/academic              单元、API、队列、迁移与安全测试
```

## 来源与许可

该项目最初基于 [GPT Researcher](https://github.com/assafelovic/gpt-researcher) 的代码库开展，现已精简为独立的 Academic Paper Evaluation Agent。保留及修改部分遵循仓库中的 Apache-2.0 `LICENSE`；详情见 `NOTICE.md`。
