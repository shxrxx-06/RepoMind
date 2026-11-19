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

async function send(path, options) {
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
  if (!payload || typeof payload !== "object" || !payload.metadata) {
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
