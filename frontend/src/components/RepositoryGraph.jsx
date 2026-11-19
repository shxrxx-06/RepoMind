import { memo, useEffect, useMemo, useState } from "react";
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Panel,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { getRepositoryGraph } from "../api/repositories.js";

const NODE_TYPES = ["File", "Class", "Function", "Method"];
const EDGE_TYPES = ["IMPORTS", "DEFINES", "CONTAINS", "CALLS"];
const NODE_COLORS = {
  File: "#467e62",
  Class: "#6274a2",
  Function: "#b27836",
  Method: "#8d659e",
};
const EDGE_COLORS = {
  IMPORTS: "#568c70",
  DEFINES: "#799078",
  CONTAINS: "#8490b2",
  CALLS: "#c18542",
};
const MAX_VISIBLE_NODES = 350;

function RepositoryNode({ data, selected }) {
  const node = data.node;
  const label = node.qualified_name || node.file_path || node.node_id;
  return (
    <div className={`repo-graph-node${selected ? " is-selected" : ""}${data.isConnected ? " is-connected" : ""}`}>
      <Handle type="target" position={Position.Left} />
      <span className="repo-node-type" style={{ color: NODE_COLORS[node.node_type] }}>{node.node_type}</span>
      <strong title={label}>{label}</strong>
      {node.file_path && <small title={node.file_path}>{node.file_path}</small>}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { repository: memo(RepositoryNode) };

function graphLayout(nodes, relationships) {
  const outgoing = new Map(nodes.map((node) => [node.node_id, []]));
  for (const edge of relationships) {
    outgoing.get(edge.source)?.push(edge.target);
  }
  const depth = new Map();
  const queue = nodes.filter((node) => node.node_type === "File").map((node) => node.node_id).sort();
  queue.forEach((nodeId) => depth.set(nodeId, 0));
  for (let index = 0; index < queue.length; index += 1) {
    const current = queue[index];
    for (const target of outgoing.get(current) || []) {
      if (!depth.has(target)) {
        depth.set(target, depth.get(current) + 1);
        queue.push(target);
      }
    }
  }
  const maxDepth = Math.max(0, ...depth.values());
  const layers = new Map();
  for (const node of nodes) {
    const layer = depth.get(node.node_id) ?? maxDepth + 1;
    if (!layers.has(layer)) layers.set(layer, []);
    layers.get(layer).push(node);
  }
  const positions = new Map();
  for (const [layer, layerNodes] of layers) {
    layerNodes.sort((left, right) => left.node_id.localeCompare(right.node_id));
    layerNodes.forEach((node, index) => {
      positions.set(node.node_id, { x: 100 + layer * 290, y: 80 + index * 125 });
    });
  }
  return positions;
}

function TypeFilter({ title, options, selected, onChange }) {
  return (
    <fieldset className="graph-filter-group">
      <legend>{title}</legend>
      {options.map((option) => (
        <label key={option}>
          <input
            type="checkbox"
            checked={selected.includes(option)}
            onChange={() => onChange(
              selected.includes(option)
                ? selected.filter((item) => item !== option)
                : [...selected, option],
            )}
          />
          <span>{option}</span>
        </label>
      ))}
    </fieldset>
  );
}

function FlowCanvas({ nodes, edges, onNodeClick, onPaneClick }) {
  const { fitView } = useReactFlow();
  useEffect(() => {
    if (nodes.length) fitView({ padding: 0.18 });
  }, [fitView, nodes.length, edges.length]);
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodeClick={onNodeClick}
      onPaneClick={onPaneClick}
      fitView
      minZoom={0.08}
      maxZoom={2}
      nodesDraggable
      nodesConnectable={false}
      elementsSelectable
      proOptions={{ hideAttribution: true }}
    >
      <Background color="#dce5e0" gap={22} size={1} />
      <Controls showInteractive={false} />
      <MiniMap pannable zoomable nodeColor={(node) => NODE_COLORS[node.data.node.node_type] || "#8a9690"} />
      <Panel position="top-right">
        <button className="graph-fit-button" type="button" onClick={() => fitView({ padding: 0.18, duration: 180 })}>
          Fit graph
        </button>
      </Panel>
    </ReactFlow>
  );
}

function NodeDetails({ node, relationships, nodeById }) {
  if (!node) {
    return <aside className="graph-details"><span className="eyebrow">NODE DETAILS</span><p>Select a node to inspect its metadata and connected relationships.</p></aside>;
  }
  const related = relationships.filter((edge) => edge.source === node.node_id || edge.target === node.node_id);
  return (
    <aside className="graph-details">
      <span className="eyebrow">{node.node_type} DETAILS</span>
      <h3>{node.qualified_name || node.file_path || node.node_id}</h3>
      <dl>
        <dt>Node ID</dt><dd><code>{node.node_id}</code></dd>
        {node.file_path && <><dt>File path</dt><dd><code>{node.file_path}</code></dd></>}
        {node.symbol_type && <><dt>Symbol type</dt><dd>{node.symbol_type}</dd></>}
        {node.start_line != null && <><dt>Lines</dt><dd>{node.start_line}{node.end_line != null && node.end_line !== node.start_line ? `–${node.end_line}` : ""}</dd></>}
      </dl>
      <div className="graph-related-title">Connected relationships <span>{related.length}</span></div>
      {related.length ? (
        <ul>{related.map((edge, index) => {
          const incoming = edge.target === node.node_id;
          const neighborId = incoming ? edge.source : edge.target;
          const neighbor = nodeById.get(neighborId);
          return <li key={`${edge.source}-${edge.target}-${edge.relationship_type}-${index}`}>
            <strong style={{ color: EDGE_COLORS[edge.relationship_type] }}>{edge.relationship_type}</strong>
            <span>{incoming ? "←" : "→"} {neighbor?.qualified_name || neighbor?.file_path || neighborId}</span>
            {edge.source_lines?.length > 0 && <small>source lines {edge.source_lines.join(", ")}</small>}
            {edge.resolution_status && <small>{edge.resolution_status}</small>}
          </li>;
        })}</ul>
      ) : <p>No visible relationships for this node under the current filters.</p>}
    </aside>
  );
}

export default function RepositoryGraph({ repositories, selectedRepositoryId, repositoriesError }) {
  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [nodeTypesSelected, setNodeTypesSelected] = useState(NODE_TYPES);
  const [edgeTypesSelected, setEdgeTypesSelected] = useState(EDGE_TYPES);
  const [selectedNodeId, setSelectedNodeId] = useState("");

  useEffect(() => {
    setGraph(null);
    setSelectedNodeId("");
    setError("");
    if (!selectedRepositoryId) return undefined;
    let active = true;
    setLoading(true);
    getRepositoryGraph(selectedRepositoryId)
      .then((result) => { if (active) setGraph(result); })
      .catch((requestError) => { if (active) setError(requestError.message); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [selectedRepositoryId]);

  const allNodes = graph?.nodes || [];
  const allEdges = graph?.relationships || [];
  const searchTerm = search.trim().toLocaleLowerCase();
  const matchingNodes = useMemo(() => allNodes
    .filter((node) => nodeTypesSelected.includes(node.node_type))
    .filter((node) => !searchTerm || [node.qualified_name, node.file_path, node.node_id]
      .some((value) => value?.toLocaleLowerCase().includes(searchTerm)))
    .sort((left, right) => left.node_id.localeCompare(right.node_id)),
  [allNodes, nodeTypesSelected, searchTerm]);
  const overflow = matchingNodes.length > MAX_VISIBLE_NODES;
  const visibleNodes = overflow ? matchingNodes.slice(0, MAX_VISIBLE_NODES) : matchingNodes;
  const visibleNodeIds = new Set(visibleNodes.map((node) => node.node_id));
  const visibleRelationships = allEdges.filter((edge) => (
    edgeTypesSelected.includes(edge.relationship_type)
    && visibleNodeIds.has(edge.source)
    && visibleNodeIds.has(edge.target)
  ));
  const selectedNode = visibleNodes.find((node) => node.node_id === selectedNodeId) || null;
  const nodeById = new Map(allNodes.map((node) => [node.node_id, node]));
  const positions = graphLayout(visibleNodes, visibleRelationships);
  const connectedIds = new Set();
  for (const edge of visibleRelationships) {
    if (edge.source === selectedNodeId) connectedIds.add(edge.target);
    if (edge.target === selectedNodeId) connectedIds.add(edge.source);
  }
  const flowNodes = visibleNodes.map((node) => ({
    id: node.node_id,
    type: "repository",
    position: positions.get(node.node_id),
    data: { node, isConnected: connectedIds.has(node.node_id) },
    selected: node.node_id === selectedNodeId,
  }));
  const flowEdges = visibleRelationships.map((edge, index) => {
    const connected = edge.source === selectedNodeId || edge.target === selectedNodeId;
    return {
      id: `${edge.source}-${edge.target}-${edge.relationship_type}-${index}`,
      source: edge.source,
      target: edge.target,
      label: edge.relationship_type,
      markerEnd: { type: MarkerType.ArrowClosed, color: EDGE_COLORS[edge.relationship_type] },
      style: { stroke: EDGE_COLORS[edge.relationship_type], strokeWidth: connected ? 2.6 : 1.5, opacity: selectedNodeId && !connected ? 0.22 : 0.88 },
      labelStyle: { fill: EDGE_COLORS[edge.relationship_type], fontSize: 9, fontWeight: 600 },
      labelBgStyle: { fill: "#ffffff", fillOpacity: 0.88 },
      animated: edge.relationship_type === "CALLS" && connected,
    };
  });

  return (
    <section className="repository-graph-section" aria-labelledby="repository-graph-title">
      <div className="graph-heading">
        <div><span className="eyebrow">STRUCTURE & RELATIONSHIPS</span><h2 id="repository-graph-title">Repository graph</h2>
          <p>Explore files, symbols, imports, definitions, containment, and resolved calls.</p></div>
        {selectedRepositoryId && <span className="graph-repository-id">Repository {selectedRepositoryId.slice(0, 8)}</span>}
      </div>
      {!selectedRepositoryId && !repositoriesError && <div className="graph-state">Ingest a repository to view its graph.</div>}
      {!selectedRepositoryId && repositoriesError && <div className="graph-state graph-state--error" role="alert">{repositoriesError}</div>}
      {selectedRepositoryId && loading && <div className="graph-state" role="status"><span className="button-spinner" /> Loading repository graph…</div>}
      {selectedRepositoryId && !loading && error && <div className="graph-state graph-state--error" role="alert">{error}</div>}
      {graph && !loading && !error && allNodes.length === 0 && <div className="graph-state">This repository has no graph nodes to display.</div>}
      {graph && !loading && !error && allNodes.length > 0 && (
        <>
          <div className="graph-toolbar">
            <label className="graph-search"><span>Search graph</span><input type="search" value={search} onChange={(event) => { setSearch(event.target.value); setSelectedNodeId(""); }} placeholder="File, class, function, method…" /></label>
            <TypeFilter title="Node types" options={NODE_TYPES} selected={nodeTypesSelected} onChange={(types) => { setNodeTypesSelected(types); setSelectedNodeId(""); }} />
            <TypeFilter title="Relationships" options={EDGE_TYPES} selected={edgeTypesSelected} onChange={(types) => { setEdgeTypesSelected(types); setSelectedNodeId(""); }} />
          </div>
          <div className="graph-legend" aria-label="Graph legend">
            {NODE_TYPES.map((type) => <span key={type}><i style={{ background: NODE_COLORS[type] }} />{type}</span>)}
            {EDGE_TYPES.map((type) => <span key={type}><i className="legend-line" style={{ background: EDGE_COLORS[type] }} />{type}</span>)}
          </div>
          {overflow && <p className="graph-limit-note" role="status">Showing {visibleNodes.length} of {matchingNodes.length} matching nodes. Search or filter the graph to narrow the view.</p>}
          {visibleNodes.length === 0 ? <div className="graph-state">No nodes match the current search and filters.</div> : (
            <div className="graph-workspace">
              <div className="graph-canvas" aria-label="Interactive repository graph">
                <ReactFlowProvider><FlowCanvas nodes={flowNodes} edges={flowEdges} onNodeClick={(_, node) => setSelectedNodeId(node.id)} onPaneClick={() => setSelectedNodeId("")} /></ReactFlowProvider>
              </div>
              <NodeDetails node={selectedNode} relationships={visibleRelationships} nodeById={nodeById} />
            </div>
          )}
          <p className="graph-count-note">Showing {visibleNodes.length} of {allNodes.length} nodes · {visibleRelationships.length} visible relationships of {allEdges.length} total</p>
        </>
      )}
    </section>
  );
}
