# Academic Paper Review Agent — 修改记录

> 项目：基于 GPT Researcher 改造的学术论文检索与评估智能体  
> 当前开发分支：`feat/academic-paper-review`  
> 基线版本：`main` / `5d84d2f5`  
> 记录建立日期：2026-08-06  
> 最后更新日期：2026-08-26

## 0. 当前权威状态（2026-08-26）

> 本节取代下方 2026-08-06 的阶段性“正在进行/路线图”描述；旧内容保留作
> 历史记录。逐项依赖和验收标准以 `ACADEMIC_TODO.md` 为准。

### 已完成并验证

- Wave 1：DBQ 数据库/队列可靠性与 OPS 部署基线；
- Wave 2 中不依赖人工数据的工程任务：检索数据导出、标注编译和冻结契约；
- Wave 3：SEC 安全边界、Web 工作台、MCP Server 与
  `academic-paper-review` Skill；
- Wave 4：README、架构/生命周期图、API→队列→Worker→数据库演示记录、
  Web/MCP/Skill 演示记录；
- Stretch：Prometheus `/metrics`、可选 Grafana 仪表板/告警，以及默认关闭的
  多论文 per-source 检索上限。

### 当前实测工程结果

| 检查项 | 结果 |
|---|---|
| Academic 快速测试 | `286 passed, 1 skipped`（真实 Ollama E2E 为显式 opt-in） |
| Next.js production build | 通过 |
| MCP 独立进程生命周期演示 | 通过，协议 `2025-06-18`，7 个工具 |
| Skill 校验 | `Skill is valid!` |
| Web 桌面/390×844 移动端 | 通过，无横向溢出，9 个维度与导出入口可用 |
| Alembic / Compose | 迁移 head `b7d2f6c8a104`；API、Worker、MySQL、Redis 健康 |
| 可观测性配置 | Prometheus/Grafana Compose profile、5 条告警、6 个面板通过结构校验 |

录制新提交演示时发现并修复了一个 SQLAlchemy 异步响应回归：create、retry、
cancel 在 commit 后读取数据库生成字段可能触发 `MissingGreenlet`。当前实现先
flush/refresh 并冻结响应，再 commit；回归测试覆盖三条路径。真实 E2E 也将排队
等待与模型运行分开计时，避免队首慢任务造成误报。

### 尚未完成/不可宣称

- `RET-07` 人工相关性标签和 `EVAL-01` 人工九维评分集仍需人工工作；
- 因此 `RET-11B`、`EVAL-03` 与依赖它们的 `PORT-04` 简历量化文案仍阻塞；
- 当前不得宣称 Recall@K 提升、与人工评分相关性、MAE 或消融收益；
- per-source 上限仅作为 opt-in 实验开关，默认关闭，等待多论文 benchmark 决定
  是否启用。

## 1. 文档用途与更新规则

本文件持续记录本项目相对 GPT Researcher 原始版本所做的修改，包括：

- 修改目的与解决的原始问题；
- 新增或修改的文件；
- 新增的数据结构、模块和接口；
- 测试方法与已确认的结果；
- Git 提交记录；
- 已知问题与下一步计划。

状态定义：

- **已完成并验证**：代码已经运行或测试成功；
- **已实现待验证**：代码已经提供或写入，但尚未确认测试结果；
- **计划中**：尚未开始编码。

以后每完成一个修改，应先更新本文件，再进入下一项功能。

---

## 2. 项目改造目标

将 GPT Researcher 从普通网页研究报告生成器改造成学术论文检索与评估智能体，重点解决原系统暴露出的以下问题：

1. 学术来源不足，原报告主要引用企业网站和博客；
2. 不同学术数据库的返回格式不统一；
3. 单一搜索服务失败时，整个研究流程可能中断；
4. 纯向量检索容易漏掉 DOI、模型名、数据集名和专业缩写；
5. 引用链接存在，但缺少 claim-to-source 证据对齐；
6. 缺少矛盾观点识别和来源可信度评分；
7. 报告可能中途截断，缺少完整性检查；
8. 缺少能够证明改进效果的自动评测体系。

---

## 3. 已确认完成的修改

### 3.1 创建独立开发分支

**状态：已完成并验证**

- 从原始 `main` 分支创建：`feat/academic-paper-review`；
- 创建时基线提交：`5d84d2f5`；
- 保留原始可运行版本，避免实验性修改污染主分支；
- 原先已经存在的 `gpt_researcher/actions/query_processing.py` 修改继续保留，没有纳入本轮学术模块提交。

### 3.2 新建学术模块

**状态：已完成并验证**

新增目录：

```text
gpt_researcher/academic/
├── __init__.py
├── schemas.py
└── source_retriever.py
```

目的：将学术评估功能放入独立模块，减少对 GPT Researcher 原核心代码的直接侵入。

### 3.3 建立统一数据结构

**状态：已完成并验证**

文件：`gpt_researcher/academic/schemas.py`

已加入的数据结构：

| 数据结构 | 用途 |
|---|---|
| `PaperMetadata` | 统一保存来自不同学术数据库的论文元数据 |
| `EvidenceChunk` | 保存可引用的论文片段及检索分数 |
| `AcademicClaim` | 保存从论文中抽取的结构化主张 |
| `ContradictionPair` | 保存两条主张之间的矛盾关系 |
| `CitationValidation` | 保存论断与证据之间的引用验证结果 |
| `EvaluationScore` | 保存论文各评价维度的 0–10 分数 |

`PaperMetadata` 当前包含：

- 论文 ID、标题、作者、年份、DOI；
- 期刊或会议、URL、摘要；
- 引用次数、开放获取状态；
- 来源类型；
- 数据库来源（OpenAlex、Crossref、Semantic Scholar、arXiv、Web 或上传文件）。

已验证：

- 所有 Schema 均可正常导入；
- `EvaluationScore.score` 被限制在 0–10；
- 输入 `score=11` 会正确触发 Pydantic `ValidationError`。

### 3.4 接入 OpenAlex 学术检索

**状态：已完成并验证**

文件：`gpt_researcher/academic/source_retriever.py`

新增类：`OpenAlexRetriever`

功能：

- 使用 OpenAlex `/works` API 搜索论文；
- 从 `.env` 读取 `OPENALEX_API_KEY`；
- 提取标题、作者、年份、DOI、发表来源和引用次数；
- 恢复 OpenAlex 倒排索引格式的摘要；
- 识别开放获取状态；
- 将 OpenAlex 原始结果转换成统一的 `PaperMetadata`；
- 将 OpenAlex 类型映射为项目内部类型。

真实检索已成功返回三篇 RAG 相关论文，包括：

- *Retrieval-Augmented Generation for Large Language Models: A Survey*；
- *Active Retrieval Augmented Generation*；
- *Benchmarking Large Language Models in Retrieval-Augmented Generation*。

### 3.5 修复 API Key 错误日志泄露

**状态：已完成并验证**

发现的问题：

- `httpx.Response.raise_for_status()` 的异常信息会包含完整请求 URL；
- OpenAlex Key 作为查询参数发送，因此 401 traceback 会暴露 Key。

修改：

- 不再直接调用会输出完整 URL 的 `raise_for_status()`；
- 改为检查 `response.is_error`；
- 错误信息只保留 HTTP 状态码；
- 401 错误现在显示为安全的摘要信息，不再打印 Key。

安全要求：

- `.env` 不得提交到 Git；
- 本日志不记录任何 API Key 或邮箱值；
- 已出现在截图或日志中的 Key 必须撤销并重新生成。

### 3.6 添加 OpenAlex 单元测试

**状态：已完成并验证**

文件：`tests/academic/test_source_retriever.py`

已验证的行为：

1. DOI 标准化；
2. OpenAlex 摘要恢复；
3. OpenAlex 原始记录到 `PaperMetadata` 的转换；
4. 来源类型映射。

测试结果：

```text
4 passed
```

测试环境：

- Python 3.11.15；
- pytest 9.1.1；
- 初次测试出现两个异步配置 warning，原因是缺少 `pytest-asyncio`；
- 另有一个来自原项目 `langchain-community` 的弃用 warning，与本次学术检索代码无关。

### 3.7 接入 Crossref 学术检索

**状态：已完成并验证（自动测试状态待补充）**

文件：`gpt_researcher/academic/source_retriever.py`

新增类：`CrossrefRetriever`

功能：

- 使用 Crossref `/works` API 搜索论文；
- 从 `.env` 读取 `CROSSREF_MAILTO`，进入 Crossref Polite Pool；
- 使用 `query.bibliographic` 搜索文献；
- 提取 DOI、标题、作者、年份、期刊或会议、摘要和引用次数；
- 清理摘要中的 HTML/JATS 标签；
- 将 Crossref 类型映射为项目内部类型；
- 输出统一的 `PaperMetadata`。

真实 Crossref 搜索已经成功。Crossref 自动测试代码已经设计，但当前记录中尚未确认最终测试输出和 Git 提交。

---

## 4. Git 提交记录

| 提交 | 内容 | 状态 |
|---|---|---|
| `f82bbbf` | `feat: add academic schemas and OpenAlex retriever` | 已确认 |
| `test: add OpenAlex retriever unit tests` | OpenAlex 单元测试 | 提交哈希待确认 |
| `feat: add Crossref academic retriever` | Crossref 检索与测试 | 待确认提交 |

注意：`gpt_researcher/actions/query_processing.py` 是此前已有修改，不属于本轮学术模块提交。

---

## 5. 当前正在进行的修改

### 5.1 学术检索 Fallback

**状态：已提供实现方案，待确认写入和测试**

目标行为：

```text
OpenAlex 有结果
    → 使用 OpenAlex

OpenAlex 报错或返回空结果
    → 自动切换 Crossref

Crossref 也失败或无结果
    → 汇总失败原因并停止
```

计划新增或修改：

```text
gpt_researcher/academic/schemas.py
    └── AcademicSearchResult

gpt_researcher/academic/fallback_policy.py
    └── AcademicSearchFallback

tests/academic/test_fallback_policy.py
    └── 4 个不访问真实 API 的 fallback 测试
```

计划验证的场景：

1. OpenAlex 成功时不调用 Crossref；
2. OpenAlex 报错时调用 Crossref；
3. OpenAlex 返回空列表时调用 Crossref；
4. 两个数据源都失败时抛出汇总错误。

预期学术模块测试总数：

```text
11 passed
```

---

## 6. 后续路线图

### 第一阶段：核心质量改造

- [x] 学术数据Schema；
- [x] OpenAlex检索；
- [x] Crossref检索；
- [ ] 搜索fallback；
- [ ] BM25关键词检索；
- [ ] 向量检索；
- [ ] RRF混合排序；
- [ ] 学术来源质量评分；
- [ ] Claim抽取；
- [ ] 矛盾信息检测；
- [ ] 引用支持度验证；
- [ ] 多维论文评分；
- [ ] 报告完整性检查与续写。

### 第二阶段：工程化与评测

- [ ] 建立检索评测集；
- [ ] 对比 BM25 only、Vector only、Hybrid；
- [ ] 评估 Recall@K、Precision@K、nDCG、MRR；
- [ ] 评估 Citation Coverage 和 Evidence Support Rate；
- [ ] Qdrant向量持久化；
- [ ] Docker Compose；
- [ ] 依赖和健康检查；
- [ ] 自动化测试与运行日志。

### 第三阶段：Agent集成

- [ ] 学术评估 MCP Server；
- [ ] `academic-paper-review` Skill；
- [ ] 让其他 Agent 通过 Skill + MCP 调用学术评估后端。

### 暂不加入

- Kafka；
- 大规模微服务拆分；
- 无实验依据的复杂多Agent讨论。

原因：当前单机、单用户流程没有足够的吞吐或分布式任务需求，Kafka现阶段会增加不必要的复杂度。

---

## 7. 当前已知问题

1. `pytest` 和 `pytest-asyncio` 尚未正式记录到项目开发依赖中；
2. 原项目的 `langchain-community` PDF加载器产生弃用warning；
3. Crossref的 `is_open_access` 当前保守设置为 `False`，尚未基于license元数据做可靠判断；
4. Fallback当前设计只在“错误或零结果”时切换，不会自动合并两个数据库；
5. 尚未处理OpenAlex和Crossref之间的重复DOI；
6. 学术模块尚未接入GPT Researcher原有的Researcher和Writer主流程。

---

## 8. 下一次更新要求

Fallback测试完成后，应更新：

- `AcademicSearchResult`和`AcademicSearchFallback`的实际实现状态；
- fallback测试实际通过数量；
- Git提交哈希；
- 是否保留或修改触发规则；
- 新发现的问题。

之后每完成一项功能，都按相同结构追加记录。

---

## 9. 2026-09-05 评测闭环与简历证据更新

本节覆盖并取代上文早期“尚未实现”的路线图状态；最终完成状态以
`ACADEMIC_TODO.md` 为准。

### 9.1 EVAL-03 开发集消融

- 冻结后运行 3 个配置、3 篇开发论文、每配置 27 个维度，共 81 条评分；
- 4 条 `invalid_json` 经受限紧凑恢复后全部成功，最终终止失败为 0；
- 保留评分锚点：Hybrid anchored MAE `0.304348`，去锚点 MAE `0.521739`；
- 保留 Hybrid：section-only MAE `0.28`，仅改善 `0.024348`，未达到冻结的
  `0.1` 采纳门槛；
- 保留矛盾告警阈值 `0.95`；未访问 LoRA/MBR held-out 标签进行调参；
- 完整报告由 `eval03_development_ablation_summary.json` 进行 SHA-256 固定。

### 9.2 EVAL-04 基准自动化

- PR/main CI 新增无网络确定性门禁，检查 11 个 artifact hash 和 15 条
  零失败、分割隔离、禁止 test retuning、冻结决策不变量；
- 新增每周及手动完整 benchmark：真实运行 8 组 retrieval/quality 评测，保存
  checkpoint、SHA-256 运行清单和 30 天 artifact；
- 定时输出验证为 fail-closed，终止失败或数据边界违规会使任务失败。

### 9.3 PORT-04 简历证据

- 新增 `docs/academic_resume_evidence.v1.json`，所有可写入简历的数字均绑定
  版本化来源和 SHA-256；
- 新增 `docs/academic_resume_bullets.md` 中英文岗位 bullet 与面试边界；
- 明确禁止把 ChatGPT 参考标签表述为人工 gold、把自动评分表述为同行评审，
  或隐藏 Hybrid 在 held-out 上低于 BM25 的负结果。
