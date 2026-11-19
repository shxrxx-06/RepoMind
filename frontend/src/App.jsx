import { useEffect, useState } from "react";

import { getApiBaseUrl, ingestGithubRepository, listRepositories, uploadRepositoryZip } from "./api/repositories.js";
import Brand from "./components/Brand.jsx";
import InvestigationPanel from "./components/InvestigationPanel.jsx";
import RepositoryOverview from "./components/RepositoryOverview.jsx";
import RepositorySubmission from "./components/RepositorySubmission.jsx";

export default function App() {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [repository, setRepository] = useState(null);
  const [repositories, setRepositories] = useState([]);
  const [repositoriesError, setRepositoriesError] = useState("");
  const [investigationKey, setInvestigationKey] = useState(0);

  useEffect(() => {
    listRepositories()
      .then((items) => {
        setRepositories(items);
        setRepositoriesError("");
      })
      .catch((requestError) => setRepositoriesError(requestError.message));
  }, []);

  async function submitRepository({ type, value }) {
    if (isSubmitting) return;
    setIsSubmitting(true);
    setError("");
    setRepository(null);
    try {
      const result = type === "github"
        ? await ingestGithubRepository(value)
        : await uploadRepositoryZip(value);
      setRepository(result);
      setInvestigationKey((key) => key + 1);
      try {
        const items = await listRepositories();
        setRepositories(items);
        setRepositoriesError("");
      } catch (requestError) {
        setRepositoriesError(requestError.message);
      }
    } catch (submissionError) {
      setError(submissionError.message || "Repository ingestion failed. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div id="top" className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <Brand />
          <div className="topbar-right">
            <span className="api-status"><span /> Repository ingestion</span>
            <a href={`${getApiBaseUrl()}/docs`} target="_blank" rel="noreferrer" className="docs-link">API docs <span aria-hidden="true">↗</span></a>
          </div>
        </div>
      </header>

      <main>
        <section className="hero-grid" aria-labelledby="hero-title">
          <div className="hero-copy">
            <div className="hero-kicker"><span className="kicker-line" /> REPOSITORY INTELLIGENCE</div>
            <h1 id="hero-title">Understand the code.<br /><span>See the connections.</span></h1>
            <p className="hero-description">
              RepoMind maps the structure of your repository so you can get oriented before you dive in.
            </p>
            <div className="capability-row" aria-label="Current capabilities">
              <span><i className="capability-icon files-icon" aria-hidden="true" /> File inventory</span>
              <span><i className="capability-icon language-icon" aria-hidden="true" /> Language breakdown</span>
              <span><i className="capability-icon shield-icon" aria-hidden="true" /> Local processing</span>
            </div>
          </div>

          <div className="submission-column">
            <RepositorySubmission
              onSubmit={submitRepository}
              isSubmitting={isSubmitting}
              error={error}
            />
            {repository && <RepositoryOverview repository={repository} />}
          </div>
        </section>

        <InvestigationPanel
          key={investigationKey}
          repositories={repositories}
          repositoriesError={repositoriesError}
          initialRepositoryId={repository?.repository_id || ""}
        />

        <section className="how-it-works" aria-label="How RepoMind works">
          <div className="how-heading">
            <span className="eyebrow">A CLEARER STARTING POINT</span>
            <p>From repository to useful overview in a few steps.</p>
          </div>
          <div className="steps-row">
            <div className="step-item"><span className="step-number">01</span><span><strong>Connect</strong><small>Choose GitHub or ZIP</small></span></div>
            <span className="step-connector" aria-hidden="true" />
            <div className="step-item"><span className="step-number">02</span><span><strong>Scan</strong><small>Inventory files and languages</small></span></div>
            <span className="step-connector" aria-hidden="true" />
            <div className="step-item"><span className="step-number">03</span><span><strong>Get oriented</strong><small>Review your repository overview</small></span></div>
          </div>
        </section>
      </main>

      <footer className="footer">
        <Brand compact />
        <span>Repository intelligence, grounded in your code.</span>
        <span className="footer-version">RepoMind · Day 9</span>
      </footer>
    </div>
  );
}
