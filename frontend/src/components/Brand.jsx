export default function Brand({ compact = false }) {
  return (
    <a className={`brand${compact ? " brand--compact" : ""}`} href="#top" aria-label="RepoMind home">
      <span className="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 36 36" fill="none">
          <path d="M10 7.5h12l5 5V28a2 2 0 0 1-2 2H10a2 2 0 0 1-2-2V9.5a2 2 0 0 1 2-2Z" />
          <path d="M21.5 7.5v5.8h5.4M13 18h10M13 22.5h6" />
          <circle cx="24.5" cy="24" r="4" />
          <path d="m27.5 27 2.8 2.8" />
        </svg>
      </span>
      <span className="brand-word">Repo<span>Mind</span></span>
      {!compact && <span className="brand-badge">BETA</span>}
    </a>
  );
}
