const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");
const API_ROOT = `${API_BASE_URL}/api/v1`;

export function getApiBaseUrl() {
  return API_BASE_URL;
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function describeDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const location = Array.isArray(item.loc) ? item.loc.slice(1).join(" → ") : "";
        return [location, item.msg].filter(Boolean).join(": ");
      })
      .filter(Boolean)
      .join(" ");
  }
  return "The backend could not process this request.";
}

async function send(path, options, requireRepositoryMetadata = true) {
  let response;
  try {
    response = await fetch(`${API_ROOT}${path}`, options);
  } catch {
    const destination = API_BASE_URL || "the configured API origin";
    throw new ApiError(`Could not reach the RepoMind backend at ${destination}. Check that it is running and try again.`);
  }

  let payload = {};
  try {
    payload = await response.json();
  } catch {
    // An empty or non-JSON error body is handled by the status fallback below.
  }

  if (!response.ok) {
    const detail = describeDetail(payload.detail);
    throw new ApiError(detail, response.status);
  }
  if (!payload || typeof payload !== "object" || (requireRepositoryMetadata && !payload.metadata)) {
    throw new ApiError("The backend response did not include repository metadata.", response.status);
  }
  return payload;
}

export function ingestGithubRepository(url) {
  return send("/repositories/github", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
}

export function uploadRepositoryZip(file) {
  const form = new FormData();
  form.append("file", file);
  return send("/repositories/upload", {
    method: "POST",
    body: form,
  });
}

export async function listRepositories() {
  let response;
  try {
    response = await fetch(`${API_ROOT}/repositories`);
  } catch {
    const destination = API_BASE_URL || "the configured API origin";
    throw new ApiError(`Could not reach the RepoMind backend at ${destination}. Check that it is running and try again.`);
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    if (response.ok) throw new ApiError("The backend returned an invalid repository list response.", response.status);
    payload = {};
  }
  if (!response.ok) throw new ApiError(describeDetail(payload.detail), response.status);
  if (!Array.isArray(payload)) {
    throw new ApiError("The backend repository list response must be an array.", response.status);
  }
  const validList = payload.every((repository) => (
    repository !== null
    && typeof repository === "object"
    && typeof repository.repository_id === "string"
    && repository.repository_id.trim().length > 0
    && Number.isInteger(repository.total_files)
    && repository.total_files >= 0
    && repository.language_distribution !== null
    && typeof repository.language_distribution === "object"
    && !Array.isArray(repository.language_distribution)
    && Object.entries(repository.language_distribution).every(([language, count]) => (
      language.length > 0 && Number.isInteger(count) && count >= 0
    ))
  ));
  if (!validList) {
    throw new ApiError("The backend repository list contains an item with an invalid shape.", response.status);
  }
  return payload;
}

export async function investigateRepository(repositoryId, question) {
  return send(`/repositories/${encodeURIComponent(repositoryId)}/investigate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  }, false);
}

const graphRequests = new Map();

export function getRepositoryGraph(repositoryId) {
  if (graphRequests.has(repositoryId)) return graphRequests.get(repositoryId);
  const pending = send(
    `/repositories/${encodeURIComponent(repositoryId)}/graph`,
    { method: "GET" },
    false,
  ).then((payload) => {
    const validGraph = Array.isArray(payload.nodes)
      && payload.nodes.every((node) => (
        node
        && typeof node.node_id === "string"
        && ["File", "Class", "Function", "Method"].includes(node.node_type)
      ))
      && Array.isArray(payload.relationships)
      && payload.relationships.every((edge) => (
        edge
        && typeof edge.source === "string"
        && typeof edge.target === "string"
        && ["IMPORTS", "DEFINES", "CONTAINS", "CALLS"].includes(edge.relationship_type)
      ));
    if (!validGraph) {
      throw new ApiError("The backend returned a graph response with an invalid structure.");
    }
    return payload;
  }).finally(() => graphRequests.delete(repositoryId));
  graphRequests.set(repositoryId, pending);
  return pending;
}

export function analyzeRepositoryImpact(repositoryId, request) {
  return send(
    `/repositories/${encodeURIComponent(repositoryId)}/impact-analysis`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    },
    false,
  ).then((payload) => {
    if (
      !payload
      || typeof payload !== "object"
      || !payload.changed_component
      || !Array.isArray(payload.directly_affected)
      || !Array.isArray(payload.indirectly_affected)
      || !Array.isArray(payload.affected_files)
    ) {
      throw new ApiError("The backend returned an impact report with an invalid structure.");
    }
    return payload;
  });
}
