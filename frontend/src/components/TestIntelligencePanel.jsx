import { useEffect, useRef, useState } from "react";

import { analyzeRepositoryTestIntelligence } from "../api/repositories.js";

function ComponentName({ node }) {
  return node.qualified_name || node.file_path || node.node_id;
}

const EVIDENCE_LABELS = {
  direct_resolved_call: "Direct resolved test call",
  import_only_reference: "Import only · call not established",
  uncertain_candidate: "Uncertain call candidate",
  no_discoverable_test_reference: "No discoverable test reference",
};

export default function TestIntelligencePanel({ selectedRepositoryId }) {
  const [changedFilePath, setChangedFilePath] = useState("");
  const [qualifiedSymbolName, setQualifiedSymbolName] = useState("");
  const [maxDepth, setMaxDepth] = useState(3);
  const [maxResults, setMaxResults] = useState(200);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const requestSequence = useRef(0);

  useEffect(() => {
    requestSequence.current += 1;
    setReport(null);
    setError("");
    setLoading(false);
  }, [selectedRepositoryId]);

  async function submit(event) {
    event.preventDefault();
    if (loading || !selectedRepositoryId) return;
    if (qualifiedSymbolName.trim() && !changedFilePath.trim()) {
      setError("Enter a changed file path when specifying a qualified symbol.");
      return;
    }
    const maxResultsText = String(maxResults).trim();
    const parsedMaxResults = Number(maxResultsText);
    if (
      !maxResultsText ||
      !Number.isInteger(parsedMaxResults) ||
      parsedMaxResults < 1 ||
      parsedMaxResults > 1000
    ) {
      setError("Maximum components must be an integer between 1 and 1000.");
      return;
    }
    setLoading(true);
    setError("");
    setReport(null);
    const sequence = requestSequence.current + 1;
    requestSequence.current = sequence;
    const request = { max_depth: Number(maxDepth), max_results: parsedMaxResults };
    if (changedFilePath.trim()) request.changed_file_path = changedFilePath.trim();
    if (qualifiedSymbolName.trim()) request.qualified_symbol_name = qualifiedSymbolName.trim();
    try {
      const result = await analyzeRepositoryTestIntelligence(selectedRepositoryId, request);
      if (requestSequence.current === sequence) setReport(result);
    } catch (requestError) {
      if (requestSequence.current === sequence) {
        setError(requestError.message || "Test intelligence analysis failed. Please try again.");
      }
    } finally {
      if (requestSequence.current === sequence) setLoading(false);
    }
  }

  return (
    <section className="test-intelligence-section" aria-labelledby="test-intelligence-title">
      <div className="test-intelligence-heading">
        <span className="eyebrow">TEST DISCOVERY</span>
        <h2 id="test-intelligence-title">Test intelligence</h2>
        <p>Inspect Python test files and source references identified from AST and graph evidence.</p>
      </div>
      <form className="test-intelligence-form" onSubmit={submit} noValidate>
        <label htmlFor="test-impact-file">Changed file <span>(optional; leave blank for repository-wide mapping)</span></label>
        <input
          id="test-impact-file"
          placeholder="src/services/auth.py"
          value={changedFilePath}
          onChange={(event) => {
            setChangedFilePath(event.target.value);
            setError("");
          }}
          disabled={loading || !selectedRepositoryId}
        />
        <label htmlFor="test-impact-symbol">Qualified symbol <span>(optional)</span></label>
        <input
          id="test-impact-symbol"
          placeholder="AuthService.validate"
          value={qualifiedSymbolName}
          onChange={(event) => {
            setQualifiedSymbolName(event.target.value);
            setError("");
          }}
          disabled={loading || !selectedRepositoryId}
        />
        {changedFilePath.trim() && <label htmlFor="test-impact-depth">Impact depth</label>}
        {changedFilePath.trim() && <select id="test-impact-depth" value={maxDepth} onChange={(event) => setMaxDepth(event.target.value)} disabled={loading}>
          {[1, 2, 3, 4, 5, 6, 8, 10].map((depth) => <option key={depth} value={depth}>{depth}</option>)}
        </select>}
        <label htmlFor="test-impact-limit">Maximum components</label>
        <input
          id="test-impact-limit"
          type="number"
          required
          min="1"
          max="1000"
          value={maxResults}
          onChange={(event) => {
            setMaxResults(event.target.value);
            setError("");
          }}
          disabled={loading || !selectedRepositoryId}
        />
        {error && <div className="error-banner" role="alert"><span>!</span><p>{error}</p></div>}
        <button className="submit-button" type="submit" disabled={loading || !selectedRepositoryId}>
          {loading ? <><span className="button-spinner" aria-hidden="true" /> Analyzing tests…</> : <>Analyze tests <span aria-hidden="true">→</span></>}
        </button>
      </form>

      <div className="test-intelligence-results" aria-live="polite">
        {!selectedRepositoryId && <div className="investigation-empty"><h3>Select a repository</h3><p>Choose an ingested repository before analyzing tests.</p></div>}
        {selectedRepositoryId && !report && !loading && !error && <div className="investigation-empty"><h3>Test report appears here</h3><p>Mappings show static references; they do not establish that tests ran or passed.</p></div>}
        {loading && <div className="investigation-empty" role="status"><span className="button-spinner" aria-hidden="true" /><h3>Analyzing test sources</h3><p>Reading cached Python AST and graph evidence.</p></div>}
        {report && (
          <div className="test-intelligence-report">
            {report.warnings.map((warning) => <p className="result-warning" key={warning}>{warning}</p>)}
            {report.result_limit_reached && <p className="test-intelligence-limit" role="status">Showing a limited component set ({report.analyzed_component_count} of a larger result). Narrow the file or symbol scope, or raise the maximum component limit.</p>}
            <p className="test-intelligence-summary">Analyzed {report.analyzed_component_count} components · {report.mappings.filter((item) => item.evidence_classification === "direct_resolved_call").length} direct resolved calls · {report.mappings.filter((item) => item.evidence_classification === "import_only_reference").length} import-only references · {report.mappings.filter((item) => item.evidence_classification === "uncertain_candidate").length} uncertain candidates · {report.uncovered_components.filter((item) => item.evidence_classification === "no_discoverable_test_reference").length} with no discoverable reference</p>
            <section className="test-intelligence-group">
              <h3>Discovered test files ({report.discovered_test_files.length})</h3>
              {report.discovered_test_files.length ? report.discovered_test_files.map((file) => (
                <article className="test-intelligence-card" key={file.file_path}>
                  <strong><code>{file.file_path}</code></strong>
                  <p>{file.tests.length ? file.tests.map((test) => `${test.qualified_name} · lines ${test.start_line}–${test.end_line}`).join(" · ") : "No test classes or test functions were identified."}</p>
                  {file.imports.length > 0 && <small>{file.imports.length} import binding{file.imports.length === 1 ? "" : "s"} parsed</small>}
                  {file.parse_errors.map((parseError, index) => <small className="test-intelligence-error" key={`${parseError.error_type}-${parseError.line}-${index}`}>{parseError.error_type}{parseError.line ? ` at line ${parseError.line}` : ""}: {parseError.message}</small>)}
                </article>
              )) : <p>No Python test files were discovered.</p>}
            </section>
            <section className="test-intelligence-group">
              <h3>Static test mappings ({report.mappings.length})</h3>
              {report.mappings.length ? report.mappings.map((mapping, index) => (
                <article className="test-intelligence-card" key={`${mapping.test_file_path}-${mapping.test_qualified_name}-${mapping.source_component.node_id}-${index}`}>
                  <div className={`test-intelligence-card-title evidence-${mapping.evidence_classification}`}><strong>{mapping.test_qualified_name}</strong><span>{EVIDENCE_LABELS[mapping.evidence_classification]} · {mapping.certainty} certainty</span></div>
                  <code>{mapping.test_file_path} → <ComponentName node={mapping.source_component} /></code>
                  <small>Source location: {mapping.source_component.file_path}{mapping.source_component.start_line ? `:${mapping.source_component.start_line}${mapping.source_component.end_line && mapping.source_component.end_line !== mapping.source_component.start_line ? `–${mapping.source_component.end_line}` : ""}` : ""}</small>
                  <p>{mapping.explanation}</p>
                  <ul>{mapping.evidence.map((evidence, evidenceIndex) => <li key={`${evidence.evidence_type}-${evidence.line_number}-${evidenceIndex}`}>
                    <strong>{evidence.evidence_type}</strong> · test line {evidence.line_number}: {evidence.description}
                    {evidence.expression && <code>{evidence.expression}</code>}
                  </li>)}</ul>
                </article>
              )) : <p>No source-symbol mappings were identified from available graph evidence.</p>}
            </section>
            <section className="test-intelligence-group">
              <h3>Components without direct resolved test calls ({report.uncovered_components.length})</h3>
              {report.uncovered_components.length ? report.uncovered_components.map((item) => <article className="test-intelligence-card" key={item.source_component.node_id}>
                <span className={`evidence-classification evidence-${item.evidence_classification}`}>{EVIDENCE_LABELS[item.evidence_classification]}</span><strong><ComponentName node={item.source_component} /></strong><code>{item.source_component.file_path}{item.source_component.start_line ? `:${item.source_component.start_line}` : ""}</code><p>{item.explanation}</p>
              </article>) : <p>Every in-scope component has a direct resolved test-call mapping.</p>}
            </section>
            <section className="test-intelligence-group inferred-group">
              <h3>Inferred recommendations ({report.inferred_recommendations.length})</h3>
              {report.inferred_recommendations.length ? report.inferred_recommendations.map((item) => <article className="test-intelligence-card" key={item.source_component.node_id}>
                <span className="inferred-badge">INFERRED · NOT AN EXISTING TEST</span><span className={`evidence-classification evidence-${item.evidence_classification}`}>{EVIDENCE_LABELS[item.evidence_classification]}</span><strong><ComponentName node={item.source_component} /></strong><code>{item.source_component.file_path}{item.source_component.start_line ? `:${item.source_component.start_line}` : ""}</code><p>{item.explanation}</p>
              </article>) : <p>No inferred recommendations for this scope.</p>}
            </section>
          </div>
        )}
      </div>
    </section>
  );
}
