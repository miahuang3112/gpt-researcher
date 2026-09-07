"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import styles from "./page.module.css";

type JobStatus =
  | "queued"
  | "running"
  | "cancel_requested"
  | "completed"
  | "failed"
  | "cancelled";

type JobSummary = {
  job_id: string;
  paper_id: string;
  title?: string | null;
  status: JobStatus;
  state_version: number;
  retry_count: number;
  completed_dimensions: number;
  total_dimensions: number;
  progress: number;
  current_dimension?: string | null;
  created_at: string;
  completed_at?: string | null;
};

type JobDetail = JobSummary & {
  result?: EvaluationResult | null;
  error_code?: string | null;
  error_message?: string | null;
};

type DimensionResult = {
  dimension: string;
  score: number;
  status?: string;
  rationale?: string;
  strengths?: string[];
  concerns?: string[];
  evidence_chunk_ids?: string[];
  external_evidence_ids?: string[];
  verification?: {
    status?: string;
    reasons?: string[];
    metrics?: Record<string, number>;
  };
};

type EvaluationResult = {
  overall_score?: number;
  assessment?: string;
  evidence_coverage?: number;
  strengths?: string[];
  major_concerns?: string[];
  missing_dimensions?: string[];
  dimensions?: DimensionResult[];
  external_evidence?: {
    status?: string;
    provider_used?: string | null;
    fallback_used?: boolean;
    target_year?: number | null;
    records?: Array<{
      external_evidence_id: string;
      paper?: { title?: string; year?: number | null; doi?: string | null };
      temporal_note?: string;
    }>;
    warnings?: string[];
  };
};

type JobList = {
  items: JobSummary[];
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
};

const DIMENSION_LABELS: Record<string, string> = {
  research_question_clarity: "Research question clarity",
  methodological_rigor: "Methodological rigor",
  experimental_design: "Experimental design",
  baseline_quality: "Baseline quality",
  result_validity: "Result validity",
  reproducibility: "Reproducibility",
  novelty: "Novelty",
  limitations_discussion: "Limitations discussion",
  citation_quality: "Citation quality",
};

const TERMINAL: JobStatus[] = ["completed", "failed", "cancelled"];

export default function AcademicEvaluationPage() {
  const [jobs, setJobs] = useState<JobList>({ items: [], page: 1, page_size: 10, total: 0, total_pages: 0 });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selected, setSelected] = useState<JobDetail | null>(null);
  const [selectedDimension, setSelectedDimension] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState("");
  const lastEventId = useRef("0");

  useEffect(() => {
    setApiKey(sessionStorage.getItem("academic-api-key") || "");
  }, []);

  const headers = useCallback((extra?: HeadersInit) => {
    const value = new Headers(extra);
    if (apiKey) value.set("X-API-Key", apiKey);
    return value;
  }, [apiKey]);

  const apiFetch = useCallback(async (path: string, init?: RequestInit) => {
    const response = await fetch(path, { ...init, headers: headers(init?.headers) });
    if (!response.ok) {
      const body = await response.json().catch(() => ({ detail: `Request failed (${response.status})` }));
      throw new Error(body.detail || `Request failed (${response.status})`);
    }
    return response;
  }, [headers]);

  const loadJobs = useCallback(async () => {
    try {
      const response = await apiFetch(`/api/academic/evaluations?page=${page}&page_size=10`);
      const payload = (await response.json()) as JobList;
      setJobs(payload);
      if (!selectedId && payload.items.length) setSelectedId(payload.items[0].job_id);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not load jobs.");
    }
  }, [apiFetch, page, selectedId]);

  const loadJob = useCallback(async (jobId: string) => {
    const response = await apiFetch(`/api/academic/evaluations/${jobId}`);
    const payload = (await response.json()) as JobDetail;
    setSelected(payload);
    if (payload.status === "completed" && !selectedDimension) {
      setSelectedDimension(payload.result?.dimensions?.[0]?.dimension || null);
    }
    return payload;
  }, [apiFetch, selectedDimension]);

  useEffect(() => { void loadJobs(); }, [loadJobs]);

  useEffect(() => {
    if (!selectedId) return;
    let stopped = false;
    let pollTimer: ReturnType<typeof setInterval> | undefined;
    const controller = new AbortController();
    lastEventId.current = "0";

    const fallbackToPolling = () => {
      if (pollTimer || stopped) return;
      pollTimer = setInterval(async () => {
        try {
          const job = await loadJob(selectedId);
          if (TERMINAL.includes(job.status)) {
            if (pollTimer) clearInterval(pollTimer);
            void loadJobs();
          }
        } catch { /* keep the last visible state */ }
      }, 3000);
    };

    const connect = async () => {
      try {
        await loadJob(selectedId);
        const response = await apiFetch(`/api/academic/evaluations/${selectedId}/events`, {
          headers: { "Accept": "text/event-stream", "Last-Event-ID": lastEventId.current },
          signal: controller.signal,
        });
        const reader = response.body?.getReader();
        if (!reader) throw new Error("Streaming is unavailable.");
        const decoder = new TextDecoder();
        let buffer = "";
        while (!stopped) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split("\n\n");
          buffer = frames.pop() || "";
          for (const frame of frames) {
            const id = frame.split("\n").find((line) => line.startsWith("id: "))?.slice(4);
            const data = frame.split("\n").find((line) => line.startsWith("data: "))?.slice(6);
            if (id) lastEventId.current = id;
            if (!data) continue;
            const event = JSON.parse(data) as Partial<JobDetail>;
            setSelected((current) => current ? { ...current, ...event } : event as JobDetail);
            if (event.status && TERMINAL.includes(event.status)) {
              await loadJob(selectedId);
              await loadJobs();
              return;
            }
          }
        }
        if (!stopped) fallbackToPolling();
      } catch {
        if (!stopped) fallbackToPolling();
      }
    };
    void connect();
    return () => {
      stopped = true;
      controller.abort();
      if (pollTimer) clearInterval(pollTimer);
    };
  }, [apiFetch, loadJob, loadJobs, selectedId]);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy(true);
    setNotice(null);
    const form = new FormData(event.currentTarget);
    try {
      const response = await apiFetch("/api/academic/evaluations/upload", { method: "POST", body: form });
      const job = (await response.json()) as JobDetail;
      setSelectedId(job.job_id);
      setSelected(job);
      setPage(1);
      await loadJobs();
      event.currentTarget.reset();
      setNotice("PDF accepted. The evaluation is queued.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setBusy(false);
    }
  };

  const action = async (name: "retry" | "cancel") => {
    if (!selectedId) return;
    setBusy(true);
    try {
      const response = await apiFetch(`/api/academic/evaluations/${selectedId}/${name}`, { method: "POST" });
      setSelected(await response.json());
      await loadJobs();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : `${name} failed.`);
    } finally { setBusy(false); }
  };

  const downloadExport = async (format: "json" | "pdf") => {
    if (!selectedId) return;
    try {
      const response = await apiFetch(`/api/academic/evaluations/${selectedId}/export.${format}`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${selectedId}.academic-evaluation.v1.${format}`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Export failed.");
    }
  };

  const dimensions = selected?.result?.dimensions || [];
  const evidence = dimensions.find((item) => item.dimension === selectedDimension);

  return (
    <main className={styles.shell}>
      <header className={styles.hero}>
        <div>
          <span className={styles.eyebrow}>Academic Evaluation Workspace</span>
          <h1>Evaluate a computer-science paper</h1>
          <p>Upload a PDF, follow evidence-grounded scoring, and inspect all nine dimensions.</p>
        </div>
        <div className={styles.disclaimer}>Automated assessment · Not peer review</div>
      </header>

      <section className={styles.grid}>
        <aside className={styles.sidebar}>
          <form className={styles.card} onSubmit={submit}>
            <h2>New evaluation</h2>
            <label>PDF<input name="file" type="file" accept="application/pdf,.pdf" required /></label>
            <label>Paper ID (optional)<input name="paper_id" placeholder="Generated automatically when empty" maxLength={255} /></label>
            <label>Title<input name="title" placeholder="Optional title" maxLength={500} /></label>
            <label>Source URL (optional)<input name="source_url" type="url" placeholder="Leave empty for unpublished work" /></label>
            <div className={styles.formRow}>
              <label>Model<input name="model_name" defaultValue="qwen3.5:9b" /></label>
              <label>Chunks<input name="max_chunks" type="number" min="1" max="20" defaultValue="4" /></label>
            </div>
            <button disabled={busy} type="submit">{busy ? "Working…" : "Upload and evaluate"}</button>
            {notice && <p className={styles.notice}>{notice}</p>}
            <details>
              <summary>API authentication</summary>
              <label>Optional API key<input type="password" value={apiKey} onChange={(event) => {
                setApiKey(event.target.value);
                sessionStorage.setItem("academic-api-key", event.target.value);
              }} /></label>
            </details>
          </form>

          <div className={styles.card}>
            <div className={styles.sectionTitle}><h2>Job history</h2><span>{jobs.total}</span></div>
            <div className={styles.jobList}>
              {jobs.items.map((job) => (
                <button
                  key={job.job_id}
                  className={`${styles.job} ${selectedId === job.job_id ? styles.activeJob : ""}`}
                  style={{ background: "transparent" }}
                  onClick={() => setSelectedId(job.job_id)}
                >
                  <strong>{job.title || job.paper_id}</strong>
                  <span><i className={`${styles.dot} ${styles[job.status]}`} />{job.status.replace("_", " ")} · {Math.round(job.progress * 100)}%</span>
                </button>
              ))}
              {!jobs.items.length && <p className={styles.muted}>No evaluations yet.</p>}
            </div>
            <div className={styles.pagination}>
              <button disabled={page <= 1} onClick={() => setPage((value) => value - 1)}>Previous</button>
              <span>{page} / {Math.max(1, jobs.total_pages)}</span>
              <button disabled={page >= jobs.total_pages} onClick={() => setPage((value) => value + 1)}>Next</button>
            </div>
          </div>
        </aside>

        <section className={styles.content}>
          {!selected ? <div className={styles.empty}>Select or submit an evaluation to begin.</div> : <>
            <div className={styles.statusCard}>
              <div><span className={styles.eyebrow}>{selected.paper_id}</span><h2>{selected.title || "Evaluation report"}</h2></div>
              <div className={styles.statusActions}>
                <span className={styles.status}>{selected.status.replace("_", " ")}</span>
                {["queued", "running", "cancel_requested"].includes(selected.status) && <button disabled={busy || selected.status === "cancel_requested"} onClick={() => action("cancel")}>Cancel</button>}
                {["failed", "cancelled"].includes(selected.status) && <button disabled={busy} onClick={() => action("retry")}>Retry</button>}
                {selected.status === "completed" && <><button onClick={() => downloadExport("json")}>JSON</button><button onClick={() => downloadExport("pdf")}>PDF</button></>}
              </div>
              <div className={styles.progress}><span style={{ width: `${selected.progress * 100}%` }} /></div>
              <p>{selected.completed_dimensions}/{selected.total_dimensions} dimensions · {selected.current_dimension ? DIMENSION_LABELS[selected.current_dimension] || selected.current_dimension : "Waiting for the next state"}</p>
              {selected.error_message && <p className={styles.error}>{selected.error_message}</p>}
            </div>

            {selected.status === "completed" && selected.result && <>
              <div className={styles.scoreBand}>
                <div><span>Overall score</span><strong>{selected.result.overall_score ?? "—"}</strong></div>
                <div><span>Evidence coverage</span><strong>{Math.round((selected.result.evidence_coverage || 0) * 100)}%</strong></div>
                <div><span>Assessment</span><strong>{selected.result.assessment || "Complete"}</strong></div>
              </div>
              {selected.result.external_evidence && <section className={styles.evidencePanel}>
                <div className={styles.sectionTitle}><h2>External scholarly evidence</h2><span>{selected.result.external_evidence.status || "unknown"}</span></div>
                <p>Provider: {selected.result.external_evidence.provider_used || "none"} · Target year: {selected.result.external_evidence.target_year ?? "unverified"}</p>
                {!!selected.result.external_evidence.records?.length && <ul>{selected.result.external_evidence.records.map((record) => <li key={record.external_evidence_id}><strong>{record.paper?.title || record.external_evidence_id}</strong>{record.paper?.year ? ` (${record.paper.year})` : ""}<br /><small>{record.temporal_note}</small></li>)}</ul>}
                {!!selected.result.external_evidence.warnings?.length && <ul>{selected.result.external_evidence.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>}
              </section>}
              <div className={styles.dimensionGrid}>
                {dimensions.map((dimension) => <article key={dimension.dimension} className={styles.dimensionCard}>
                  <div><span>{DIMENSION_LABELS[dimension.dimension] || dimension.dimension}</span><strong>{dimension.score}/5</strong></div>
                  <p>{dimension.rationale}</p>
                  <button onClick={() => setSelectedDimension(dimension.dimension)}>Inspect evidence</button>
                </article>)}
              </div>
              {evidence && <section className={styles.evidencePanel}>
                <div className={styles.sectionTitle}><h2>{DIMENSION_LABELS[evidence.dimension] || evidence.dimension}</h2><span>{evidence.verification?.status || "unverified"}</span></div>
                <p>{evidence.rationale}</p>
                <h3>Evidence references</h3>
                <div className={styles.chips}>{(evidence.evidence_chunk_ids || []).map((id) => <code key={id}>{id}</code>)}</div>
                {!!evidence.external_evidence_ids?.length && <><h3>External scholarly references</h3><div className={styles.chips}>{evidence.external_evidence_ids.map((id) => <code key={id}>{id}</code>)}</div></>}
                {!!evidence.verification?.reasons?.length && <><h3>Verification notes</h3><ul>{evidence.verification.reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul></>}
                {!!evidence.concerns?.length && <><h3>Concerns</h3><ul>{evidence.concerns.map((concern) => <li key={concern}>{concern}</li>)}</ul></>}
              </section>}
            </>}
          </>}
        </section>
      </section>
    </main>
  );
}
