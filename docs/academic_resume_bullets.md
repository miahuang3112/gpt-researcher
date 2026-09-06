# Academic Paper Evaluation Agent — Resume Bullets

All numbers below come from `academic_resume_evidence.v1.json`, whose sources
are SHA-256 pinned. The evaluation references were authored by ChatGPT, not an
independent human panel; retain that qualifier in interviews and long-form
project descriptions.

## Recommended Chinese version

- 设计并实现计算机论文自动评估 Agent：以 FastAPI、MySQL、Redis Streams 和
  Ollama 构建异步九维评分流水线，采用事务 Outbox、幂等消费、逐维 checkpoint、
  `XAUTOCLAIM` 恢复、有限重试和 DLQ；实测 Worker 在 8/9 维中断后由第 2 次投递
  恢复并完成，毒消息在第 3 次投递进入 DLQ。
- 将同一持久化任务能力产品化为响应式 Web 工作台、版本化 REST API、7 个 MCP
  tools 和 `academic-paper-review` Skill，支持 PDF 安全校验、进度流、断点续查、
  证据追踪、人工复核提示及 JSON/PDF 导出；完成桌面与 `390×844` 移动端验收。
- 建立论文评分评测体系：在 5 篇论文、45 个 ChatGPT 参考维度上获得 42 个可评分
  结果、0 个终止失败，MAE `0.500`、一分内一致率 `92.86%`；16 个可比维度的独立
  temperature-zero 重跑达到 `100%` 精确一致（仅代表与版本化 AI 参考的一致性）。
- 构造 28 条 held-out 受控验证样本：引用四分类准确率 `81.25%`、macro-F1
  `78.18%`，生产 supported/not-supported 决策 F1 `100%`；矛盾六分类准确率
  `83.33%`、macro-F1 `77.78%`，冻结阈值 `0.95` 下告警 F1 `100%`。
- 实现 BM25、向量检索与 RRF Hybrid 的论文级隔离评测和 dev/test 冻结策略；发现
  Hybrid 在 18 条 held-out query 上低于 BM25 后拒绝使用测试集调参，并通过 81 次
  development-only 消融保留带评分锚点的 Hybrid 配置，展示可复现、可否证的 Agent
  评测与决策流程。
- 建立双层质量门禁：PR CI 无网络校验 11 个版本化 SHA-256 与 15 条失败/分割/冻结
  策略不变量；完整 8 组 Ollama benchmark 通过手动 CI/本地流程运行并保存
  checkpoint、运行清单和 30 天 CI artifacts，论文语料不进入公开仓库。

建议简历选前两条，再从第 3、4、5、6 条中按岗位描述选择两条。

## Recommended English version

- Built an asynchronous nine-dimension computer-science paper evaluation agent
  with FastAPI, MySQL, Redis Streams, and Ollama, adding a transactional outbox,
  idempotent consumers, per-dimension checkpoints, stale-message recovery,
  bounded retries, and a DLQ; verified recovery after a worker interruption at
  8/9 dimensions and DLQ routing on delivery three.
- Productized the persisted evaluation workflow through a responsive web app,
  versioned REST API, seven MCP tools, and an `academic-paper-review` Skill,
  including secure PDF validation, resumable progress, evidence traceability,
  human-review flags, and JSON/PDF export; verified desktop and 390×844 mobile
  flows.
- Created a versioned score-reliability benchmark across five papers and 45
  ChatGPT-reference dimensions: 42 scorable results, zero terminal failures,
  MAE 0.500, 92.86% within one point, and 100% exact repeat agreement across 16
  comparable temperature-zero items (AI-reference agreement, not human gold).
- Built 28 controlled held-out validation cases, measuring 81.25% citation
  four-class accuracy (78.18% macro-F1), 100% supported/not-supported F1,
  83.33% contradiction six-class accuracy (77.78% macro-F1), and 100% warning
  F1 at the frozen 0.95 threshold.
- Implemented paper-disjoint BM25/vector/RRF evaluation with frozen dev/test
  policy: surfaced a negative held-out Hybrid result without test-set retuning,
  then ran 81 development-only ablations to retain the anchored Hybrid baseline
  under a predeclared adoption rule.
- Added two-tier benchmark automation: network-free PR checks for 11 pinned
  hashes and 15 policy/data invariants, plus a manually dispatched eight-report
  Ollama suite with checkpoints, fail-closed validation, protected-corpus
  boundaries, and 30-day CI artifact retention.

## Interview boundaries

- Say “automated assessment, not peer review” when defining the product.
- Say “agreement with a versioned ChatGPT reference,” not “human-level scoring.”
- Explain that the negative Hybrid held-out result was preserved and did not
  trigger test-set tuning; this is evidence of evaluation discipline.
- Treat `100%` figures as small controlled-set engineering measurements and
  always give their denominators.
- Position the system as decision support requiring qualified human review, not
  as an automated peer-review or publication decision service.
