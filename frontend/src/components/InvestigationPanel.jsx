import { useState } from "react";

import { investigateRepository } from "../api/repositories.js";

function findingStatus(finding) {
  if (finding.verification === "verified") return "Verified";
  if (finding.verification === "partially_supported") return "Partially supported";
  if (finding.status === "hypothesis") return "Hypothesis";
  return "Unknown";
}

function Finding({ finding }) {
  const status = findingStatus(finding);
  const evidence = Array.isArray(finding.evidence) ? finding.evidence : [];
  return (
    <article className="finding-card">
      <div className={`finding-status finding-status--${status === "Verified" ? "verified" : status === "Unknown" || status === "Hypothesis" ? "unknown" : "partial"}`}>
        {status}
        {finding.verification_method && <span> · {finding.verification_method.replaceAll("_", " ")}</span>}
      </div>
      <p>{finding.claim}</p>
      {finding.explanation && <small>{finding.explanation}</small>}
      {evidence.map((item) => (
        <Evidence key={`${item.evidence_id}-${item.source_file_path}-${item.start_line}`} item={item} />
      ))}
      {finding.evidence_ids?.length > 0 && (
        <div className="finding-citations">Citations: {finding.evidence_ids.join(", ")}</div>
      )}
    </article>
  );
}

function Evidence({ item }) {
  const lines = item.start_line === item.end_line
    ? `line ${item.start_line}`
    : `lines ${item.start_line}–${item.end_line}`;
  return (
    <details className="evidence-source">
      <summary><code>{item.source_file_path}</code><span>{lines}</span></summary>
      {item.source_content ? <pre><code>{item.source_content}</code></pre> : <p>Source snippet unavailable.</p>}
    </details>
  );
}

export default function InvestigationPanel({ repositories, repositoriesError, initialRepositoryId }) {
  const [repositoryId, setRepositoryId] = useState(initialRepositoryId || "");
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState(null);
  const [error, setError] = useState("");
  const [isAsking, setIsAsking] = useState(false);

  async function ask(event) {
    event.preventDefault();
    if (isAsking || !repositoryId || !question.trim()) return;
    setIsAsking(true);
    setError("");
    setResponse(null);
    try {
      setResponse(await investigateRepository(repositoryId, question.trim()));
    } catch (requestError) {
      setError(requestError.message || "Investigation failed. Please try again.");
    } finally {
      setIsAsking(false);
    }
  }

  const verified = response?.verified_findings || [];
  const unverified = response?.unverified_findings || [];
  const traceSteps = response?.trace?.steps || [];
  const warnings = response?.warnings || [];
  const generationUnavailable = warnings.some((warning) =>
    /no investigation provider is configured|generated claims are unavailable|investigation provider failed|provider returned no usable response/i.test(warning),
  );

  return (
    <section className="investigation-section" aria-labelledby="investigation-title">
      <div className="investigation-heading">
        <span className="eyebrow">ASK ABOUT YOUR CODE</span>
        <h2 id="investigation-title">Investigate a repository</h2>
        <p>Ask a question and review the source evidence returned by RepoMind.</p>
      </div>
      <div className="investigation-grid">
        <form className="investigation-form" onSubmit={ask}>
          <label htmlFor="investigation-repository">Repository</label>
          <select
            id="investigation-repository"
            value={repositoryId}
            onChange={(event) => setRepositoryId(event.target.value)}
            disabled={isAsking}
          >
            <option value="">Choose an ingested repository</option>
            {repositories.map((repository) => (
              <option key={repository.repository_id} value={repository.repository_id}>
                Repository {repository.repository_id.slice(0, 8)} · {repository.total_files} files
              </option>
            ))}
          </select>
          {repositoriesError && <p className="investigation-hint" role="status">{repositoriesError}</p>}
          {!repositoriesError && repositories.length === 0 && (
            <p className="investigation-hint">Ingest a repository above to start an investigation.</p>
          )}
          <label htmlFor="investigation-question">Your question</label>
          <textarea
            id="investigation-question"
            rows="5"
            maxLength={20000}
            placeholder="How does authentication work in this repository?"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            disabled={isAsking}
            required
          />
          {error && <div className="error-banner" role="alert"><span>!</span><p>{error}</p></div>}
          <button className="submit-button" type="submit" disabled={isAsking || !repositoryId || !question.trim()}>
            {isAsking ? <><span className="button-spinner" aria-hidden="true" /> Investigating…</> : <>Ask RepoMind <span aria-hidden="true">→</span></>}
          </button>
        </form>

        <div className="investigation-result" aria-live="polite">
          {!response && !isAsking && (
            <div className="investigation-empty"><span aria-hidden="true">⌕</span><h3>Investigation results appear here</h3><p>Answers include verification status and source locations when available.</p></div>
          )}
          {isAsking && <div className="investigation-empty" role="status"><span className="button-spinner" aria-hidden="true" /><h3>Following the evidence</h3><p>Preparing repository context and checking source references.</p></div>}
          {response && (
            <div className="result-content">
              <div className="result-question"><span className="eyebrow">QUESTION</span><p>{response.question}</p></div>
              <div className="result-answer"><span className="eyebrow">INVESTIGATION</span><p>{response.answer}</p></div>
              {warnings.map((warning) => <p className="result-warning" key={warning}>{warning}</p>)}
              {verified.length > 0 && <section className="finding-group"><h3>Findings and verification</h3>{verified.map((item, index) => <Finding key={`${item.claim}-${index}`} finding={item} />)}</section>}
              {unverified.length > 0 && <section className="finding-group"><h3>Unverified claims</h3>{unverified.map((item, index) => <Finding key={`${item.claim}-${index}`} finding={item} />)}</section>}
              {verified.length === 0 && unverified.length === 0 && (
                <p className={generationUnavailable ? "result-warning" : "insufficient-evidence"}>
                  {generationUnavailable
                    ? "Answer generation is unavailable because no investigation provider is configured. Retrieved evidence and trace, if present, are shown below."
                    : "No supported findings were returned for this investigation."}
                </p>
              )}
              {response.evidence?.length > 0 && (
                <section className="finding-group"><h3>Retrieved source evidence</h3>
                  {response.evidence.map((item) => (
                    <Evidence key={item.evidence_id} item={item} />
                  ))}
                </section>
              )}
              {traceSteps.length > 0 && (
                <details className="trace-details">
                  <summary>Investigation trace ({traceSteps.length} steps)</summary>
                  <ol>{traceSteps.map((step) => <li key={step.step_number}><strong>{step.action.replaceAll("_", " ")}</strong><p>{step.result}</p><small>{step.continuation_reason}</small></li>)}</ol>
                  {response.trace?.stop_reason && <p className="trace-stop">{response.trace.stop_reason}</p>}
                </details>
              )}
              {response.test_recommendations?.length > 0 && (
                <section className="finding-group">
                  <h3>Related test recommendations</h3>
                  {response.test_recommendations.map((item, index) => (
                    <article className="finding-card" key={`${item.test_file_path || "test"}-${index}`}>
                      {item.test_file_path && <code>{item.test_file_path}</code>}
                      <p>{item.reason}</p>
                      {item.matched_symbols?.length > 0 && <small>Symbols: {item.matched_symbols.join(", ")}</small>}
                    </article>
                  ))}
                </section>
              )}
              {verified.length === 0 && unverified.length > 0 && (
                <p className="insufficient-evidence">The available repository evidence was insufficient to support a factual finding.</p>
              )}
            </div>
          )}
        </div>
      </div>
      <p className="verification-note">“Verified” means the backend found a literal source-text match. It does not prove runtime behavior or semantic entailment.</p>
    </section>
  );
}
