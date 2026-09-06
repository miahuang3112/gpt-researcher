# Demo Record: Web + MCP + Skill

## Web workflow

Start the API stack, then the frontend:

```bash
docker compose -f compose.academic.yml up -d --wait
cd frontend/nextjs
npm install
npm run dev
```

Open `http://localhost:3000/academic`. The workspace supports secure PDF upload,
paginated history, live status/progress, retry/cancel, nine dimension cards,
evidence references, verification notes, concerns, and authenticated JSON/PDF
downloads. The page visibly labels output `Automated assessment · Not peer
review`.

Recorded browser checks on 2026-08-26:

- desktop: history, completed report, 9 evidence actions, JSON/PDF controls and
  disclaimer visible;
- mobile `390 × 844`: body width equals viewport width, with no horizontal
  overflow; the same 9 dimensions and export controls remain available;
- selected history state is visually distinct and unselected jobs remain
  transparent.

The production command `npm run build` succeeds. Existing root metadata emits
Next.js viewport/theme-color migration warnings but does not fail the build.

## MCP workflow

The repository `.mcp.json` registers `academic-evaluation`. A direct isolated
demo requires only Python standard-library runtime support:

```bash
python scripts/demo_academic_mcp.py
```

Recorded output on 2026-08-26:

```json
{
  "demo": "academic-mcp-fresh-process",
  "protocol": "2025-06-18",
  "tools": [
    "cancel_evaluation",
    "get_evaluation_dimensions",
    "get_evaluation_evidence",
    "get_evaluation_result",
    "get_evaluation_status",
    "retry_evaluation",
    "submit_evaluation"
  ],
  "job_id": "fresh-demo-job",
  "overall_score": 84,
  "evidence_refs": 1,
  "result": "passed"
}
```

The isolated API data above is contract-test data, not a quality result. A
separate real MCP subprocess read deployed job
`59c18c8435fa4d54ad36665603cc311d`, returning `completed`, score `85.0`,
evidence coverage `1.0`, and all nine dimension summaries. That score is an
example persisted assessment, not a benchmark metric.

## Skill workflow

Invoke `$academic-paper-review` with a PDF and paper ID, or with an existing
`job_id` to resume. The Skill:

1. submits once and preserves the returned job ID;
2. polls compact status without repeatedly fetching the full result;
3. resumes an interrupted conversation from the same job ID;
4. retrieves dimension and evidence projections after completion;
5. does not retry a failed/cancelled job without user authorization;
6. recommends qualified human review for evidence gaps or high-impact use.

Validation command:

```bash
python /path/to/skill-creator/scripts/quick_validate.py \
  skills/academic-paper-review
```

Recorded result: `Skill is valid!`.

## Safety and evidence boundary

The web page, MCP server instructions, export document and Skill all state that
the output is an automated assessment rather than peer review. Portfolio claims
remain limited to versioned engineering evidence until human retrieval labels
and paper-quality calibration are complete.
