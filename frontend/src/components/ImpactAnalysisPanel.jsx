import { useEffect, useRef, useState } from "react";

import { analyzeRepositoryImpact } from "../api/repositories.js";

function AffectedItem({ item }) {
  return (
    <article className="impact-result-card">
      <div className="impact-item-heading">
        <strong>{item.node.qualified_name || item.node.file_path || item.node.node_id}</strong>
        <span>{item.impact_type} · {item.certainty} certainty</span>
      </div>
      {item.node.file_path && <code>{item.node.file_path}{item.node.start_line ? `:${item.node.start_line}${item.node.end_line && item.node.end_line !== item.node.start_line ? `–${item.node.end_line}` : ""}` : ""}</code>}
      <p>{item.explanation}</p>
      {item.paths.map((path, index) => (
        <details className="impact-path" key={`${path.node_ids.join("|")}-${index}`}>
          <summary>Dependency path {index + 1}: {path.node_ids.map((nodeId) => nodeId.replace(/^file:/, "")).join(" → ")}</summary>
          <ol>{path.relationships.map((edge, edgeIndex) => (
            <li key={`${edge.source_node_id}-${edge.target_node_id}-${edge.relationship_type}-${edgeIndex}`}>
              <strong>{edge.relationship_type}</strong>
              <span>{edge.source_node_id} → {edge.target_node_id}</span>
              <small>{edge.resolution_status || "resolution not recorded"}{edge.source_lines.length ? ` · line${edge.source_lines.length > 1 ? "s" : ""} ${edge.source_lines.join(", ")}` : ""}</small>
              {edge.uncertain_candidate && <em>Potential match only: {edge.evidence_expression || "the target was not statically resolved"}</em>}
            </li>
          ))}</ol>
        </details>
      ))}
    </article>
  );
}

export default function ImpactAnalysisPanel({
  repositories,
  selectedRepositoryId,
  initialTarget,
  fileOptions,
}) {
  const [changedFilePath, setChangedFilePath] = useState("");
  const [qualifiedSymbolName, setQualifiedSymbolName] = useState("");
  const [maxDepth, setMaxDepth] = useState(3);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const requestSequence = useRef(0);

  useEffect(() => {
    requestSequence.current += 1;
    setChangedFilePath(initialTarget?.changed_file_path || "");
    setQualifiedSymbolName(initialTarget?.qualified_symbol_name || "");
    setReport(null);
    setError("");
    setLoading(false);
  }, [selectedRepositoryId, initialTarget]);

  async function submit(event) {
    event.preventDefault();
    if (loading || !selectedRepositoryId || !changedFilePath.trim()) return;
    setLoading(true);
    setError("");
    setReport(null);
    const sequence = requestSequence.current + 1;
    requestSequence.current = sequence;
    const body = {
      changed_file_path: changedFilePath.trim(),
      max_depth: Number(maxDepth),
    };
    if (qualifiedSymbolName.trim()) body.qualified_symbol_name = qualifiedSymbolName.trim();
    try {
      const result = await analyzeRepositoryImpact(selectedRepositoryId, body);
      if (requestSequence.current === sequence) setReport(result);
    } catch (requestError) {
      if (requestSequence.current === sequence) {
        setError(requestError.message || "Impact analysis failed. Please try again.");
      }
    } finally {
      if (requestSequence.current === sequence) setLoading(false);
    }
  }

  return (
    <section className="impact-section" aria-labelledby="impact-title">
      <div className="impact-heading">
        <span className="eyebrow">CHANGE PLANNING</span>
        <h2 id="impact-title">Change impact</h2>
        <p>Trace incoming dependencies to see which components may need review.</p>
      </div>
      <div className="impact-workspace">
        <form className="impact-form" onSubmit={submit}>
          <label htmlFor="impact-file">Changed file</label>
          <input
            id="impact-file"
            list="impact-file-options"
            placeholder="src/services/auth.py"
            value={changedFilePath}
            onChange={(event) => setChangedFilePath(event.target.value)}
            disabled={loading}
            required
          />
          <datalist id="impact-file-options">
            {fileOptions.map((path) => <option key={path} value={path} />)}
          </datalist>
          <p className="impact-hint">Enter a repository-relative path, or select a file from the graph suggestions.</p>
          <label htmlFor="impact-symbol">Qualified symbol <span>(optional)</span></label>
          <input
            id="impact-symbol"
            placeholder="AuthService.validate"
            value={qualifiedSymbolName}
            onChange={(event) => setQualifiedSymbolName(event.target.value)}
            disabled={loading}
          />
          <label htmlFor="impact-depth">Maximum dependency depth</label>
          <select id="impact-depth" value={maxDepth} onChange={(event) => setMaxDepth(event.target.value)} disabled={loading}>
            {[1, 2, 3, 4, 5, 6, 8, 10].map((depth) => <option key={depth} value={depth}>{depth}</option>)}
          </select>
          {error && <div className="error-banner" role="alert"><span>!</span><p>{error}</p></div>}
          <button className="submit-button" type="submit" disabled={loading || !selectedRepositoryId || !changedFilePath.trim()}>
            {loading ? <><span className="button-spinner" aria-hidden="true" /> Analyzing impact…</> : <>Analyze impact <span aria-hidden="true">→</span></>}
          </button>
        </form>

        <div className="impact-report" aria-live="polite">
          {!report && !loading && <div className="investigation-empty"><span aria-hidden="true">↗</span><h3>Impact report appears here</h3><p>Results describe potential impact from recorded graph relationships, not guaranteed breakage.</p></div>}
          {loading && <div className="investigation-empty" role="status"><span className="button-spinner" aria-hidden="true" /><h3>Tracing dependents</h3><p>Following incoming imports and resolved calls in the selected repository.</p></div>}
          {report && (
            <div className="impact-report-content">
              <div className="impact-change-target"><span className="eyebrow">CHANGED COMPONENT</span><strong>{report.changed_component.qualified_name || report.changed_component.file_path}</strong><code>{report.changed_file_path}{report.changed_component.start_line ? `:${report.changed_component.start_line}` : ""}</code></div>
              {report.warnings.map((warning) => <p className="result-warning" key={warning}>{warning}</p>)}
              <div className="impact-file-summary"><span>Affected files</span><strong>{report.affected_files.length}</strong>{report.affected_files.map((file) => <code key={file.file_path}>{file.file_path} · {file.impact_type} · {file.certainty} certainty</code>)}</div>
              <section className="impact-group"><h3>Directly affected ({report.directly_affected.length})</h3>{report.directly_affected.length ? report.directly_affected.map((item) => <AffectedItem key={item.node.node_id} item={item} />) : <p>No direct dependents were found.</p>}</section>
              <section className="impact-group"><h3>Indirectly affected ({report.indirectly_affected.length})</h3>{report.indirectly_affected.length ? report.indirectly_affected.map((item) => <AffectedItem key={item.node.node_id} item={item} />) : <p>No transitive dependents were found within the selected depth.</p>}</section>
              <section className="impact-group"><h3>Discovered test files ({report.discovered_test_files.length})</h3>{report.discovered_test_files.length ? report.discovered_test_files.map((test) => <article className="impact-result-card" key={test.test_file_path}><code>{test.test_file_path}</code><p>{test.reason}</p>{test.matched_symbols.length > 0 && <small>Referenced symbols: {test.matched_symbols.join(", ")}</small>}</article>) : <p>No existing test source matched the affected file names or symbols.</p>}</section>
              <section className="impact-group"><h3>Inferred test suggestions ({report.inferred_test_recommendations.length})</h3>{report.inferred_test_recommendations.length ? report.inferred_test_recommendations.map((suggestion) => <article className="impact-result-card" key={suggestion.affected_file_path}><code>{suggestion.affected_file_path}</code><p>{suggestion.explanation}</p></article>) : <p>No inferred suggestions were needed.</p>}</section>
            </div>
          )}
        </div>
      </div>
      <p className="verification-note">Impact indicates possible downstream review based on available Python graph data. Unresolved and ambiguous references are labeled as candidates. No component is asserted to be broken.</p>
    </section>
  );
}
