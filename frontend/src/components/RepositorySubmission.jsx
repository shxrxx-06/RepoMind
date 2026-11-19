import { useState } from "react";

function GitHubIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="source-icon">
      <path fill="currentColor" d="M12 .8a11.3 11.3 0 0 0-3.57 22.02c.57.1.77-.25.77-.55v-2.1c-3.14.68-3.8-1.33-3.8-1.33-.52-1.3-1.26-1.65-1.26-1.65-1.03-.7.08-.69.08-.69 1.14.08 1.74 1.17 1.74 1.17 1.01 1.73 2.64 1.23 3.29.94.1-.73.4-1.23.72-1.52-2.5-.29-5.13-1.25-5.13-5.58 0-1.23.44-2.24 1.17-3.03-.12-.29-.51-1.44.11-2.99 0 0 .95-.3 3.11 1.16a10.82 10.82 0 0 1 5.66 0c2.16-1.46 3.11-1.16 3.11-1.16.62 1.55.23 2.7.11 2.99.73.79 1.16 1.8 1.16 3.03 0 4.34-2.63 5.28-5.14 5.56.4.35.76 1.03.76 2.08v3.11c0 .3.2.66.78.55A11.3 11.3 0 0 0 12 .8Z" />
    </svg>
  );
}

function ArchiveIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="source-icon">
      <path d="M4 7.5h16v12H4zM3 4.5h18v3H3zM9 12h6" />
    </svg>
  );
}

export default function RepositorySubmission({ onSubmit, isSubmitting, error }) {
  const [source, setSource] = useState("github");
  const [githubUrl, setGithubUrl] = useState("");
  const [zipFile, setZipFile] = useState(null);
  const [validationError, setValidationError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();
    setValidationError("");
    if (source === "github") {
      try {
        const parsed = new URL(githubUrl.trim());
        const parts = parsed.pathname.split("/").filter(Boolean);
        if (parsed.protocol !== "https:" || parsed.hostname !== "github.com" || parts.length !== 2) {
          setValidationError("Enter a repository URL like https://github.com/owner/repository.");
          return;
        }
      } catch {
        setValidationError("Enter a valid GitHub repository URL.");
        return;
      }
      await onSubmit({ type: "github", value: githubUrl.trim() });
      return;
    }

    if (!zipFile) {
      setValidationError("Choose a repository ZIP file to continue.");
      return;
    }
    if (!zipFile.name.toLowerCase().endsWith(".zip")) {
      setValidationError("The selected file must have a .zip extension.");
      return;
    }
    await onSubmit({ type: "zip", value: zipFile });
  }

  return (
    <section className="submit-card" aria-labelledby="submit-title">
      <div className="card-heading">
        <span className="eyebrow">START WITH A REPOSITORY</span>
        <h2 id="submit-title">Bring your codebase into focus.</h2>
        <p>Connect a public GitHub repository or upload a ZIP to inspect its structure and languages.</p>
      </div>

      <div className="source-switch" role="tablist" aria-label="Repository source">
        <button
          className={source === "github" ? "source-tab active" : "source-tab"}
          type="button"
          role="tab"
          aria-selected={source === "github"}
          onClick={() => setSource("github")}
          disabled={isSubmitting}
        >
          <GitHubIcon /> GitHub URL
        </button>
        <button
          className={source === "zip" ? "source-tab active" : "source-tab"}
          type="button"
          role="tab"
          aria-selected={source === "zip"}
          onClick={() => setSource("zip")}
          disabled={isSubmitting}
        >
          <ArchiveIcon /> Upload ZIP
        </button>
      </div>

      <form onSubmit={handleSubmit} noValidate>
        {source === "github" ? (
          <div className="field-block">
            <label htmlFor="github-url">GitHub repository URL</label>
            <div className="input-wrap">
              <GitHubIcon />
              <input
                id="github-url"
                name="github-url"
                type="url"
                inputMode="url"
                autoComplete="url"
                placeholder="https://github.com/owner/repository"
                value={githubUrl}
                onChange={(event) => setGithubUrl(event.target.value)}
                disabled={isSubmitting}
                required
              />
            </div>
            <p className="field-hint">Public repositories hosted on github.com.</p>
          </div>
        ) : (
          <div className="field-block">
            <label htmlFor="zip-file">Repository ZIP file</label>
            <div className={`file-picker${zipFile ? " has-file" : ""}`}>
              <input
                id="zip-file"
                name="file"
                type="file"
                accept=".zip,application/zip"
                onChange={(event) => setZipFile(event.target.files?.[0] || null)}
                disabled={isSubmitting}
                required
              />
              <span className="file-picker-icon"><ArchiveIcon /></span>
              <span className="file-picker-copy">
                <strong>{zipFile ? zipFile.name : "Choose a ZIP file"}</strong>
                <small>{zipFile ? `${(zipFile.size / (1024 * 1024)).toFixed(1)} MB` : "Select a local repository archive"}</small>
              </span>
              <span className="browse-button">Browse</span>
            </div>
            <p className="field-hint">Upload a .zip archive of your repository.</p>
          </div>
        )}

        {(validationError || error) && (
          <div className="error-banner" role="alert">
            <span aria-hidden="true">!</span>
            <p>{validationError || error}</p>
          </div>
        )}

        <button className="submit-button" type="submit" disabled={isSubmitting}>
          {isSubmitting ? (
            <><span className="button-spinner" aria-hidden="true" /> Ingesting repository…</>
          ) : (
            <>Ingest repository <span aria-hidden="true">→</span></>
          )}
        </button>
      </form>
      <p className="privacy-note"><span aria-hidden="true">◇</span> Repository files stay in your local RepoMind environment.</p>
    </section>
  );
}
