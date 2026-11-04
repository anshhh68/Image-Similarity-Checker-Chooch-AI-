import { useEffect, useRef, useState } from "react";

const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

const FIELDS = [
  ["x", "X"],
  ["y", "Y"],
  ["width", "W"],
  ["height", "H"],
];

/** Load a File into { file, url, width, height }, or reject if it isn't an image. */
function readImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve({ file, url, width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(`"${file.name}" could not be read as an image.`));
    };
    img.src = url;
  });
}

function boxFits(box, image) {
  return (
    box.width > 0 &&
    box.height > 0 &&
    box.x >= 0 &&
    box.y >= 0 &&
    box.x + box.width <= image.width &&
    box.y + box.height <= image.height
  );
}

const pct = (v) => (v * 100).toFixed(1);

/* ------------------------------------------------------------------------ */
/* One image slot: drop / browse, preview, and drag-to-draw the region       */
/* ------------------------------------------------------------------------ */
function ImagePanel({ name, image, box, onFile, onDraw }) {
  const [dragOver, setDragOver] = useState(false);
  const start = useRef(null);

  const pick = (e) => {
    const file = e.target.files?.[0];
    if (file) onFile(file);
    e.target.value = "";
  };

  // Pointer position -> pixel coordinates in the original image.
  const toPixel = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const clamp = (v) => Math.min(1, Math.max(0, v));
    return {
      x: Math.round(clamp((e.clientX - r.left) / r.width) * image.width),
      y: Math.round(clamp((e.clientY - r.top) / r.height) * image.height),
    };
  };

  const onPointerDown = (e) => {
    // Touch is left for scrolling the page; phones type the numbers instead.
    if (e.pointerType === "touch" || e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    start.current = toPixel(e);
  };

  const onPointerMove = (e) => {
    if (!start.current) return;
    const p = toPixel(e);
    const s = start.current;
    const width = Math.abs(p.x - s.x);
    const height = Math.abs(p.y - s.y);
    if (width >= 2 && height >= 2) {
      onDraw({ x: Math.min(p.x, s.x), y: Math.min(p.y, s.y), width, height });
    }
  };

  const endDraw = () => {
    start.current = null;
  };

  const showBox = image && box.width > 0 && box.height > 0;

  return (
    <section
      className={`panel ${dragOver ? "is-over" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) onFile(file);
      }}
    >
      <header className="panel-head">
        <span className="slot">Image {name}</span>
        <span className="mono muted">{image ? `${image.width} × ${image.height}` : "—"}</span>
      </header>

      {image ? (
        <div className="stage">
          <div
            className="frame"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endDraw}
            onPointerCancel={endDraw}
          >
            <img src={image.url} alt={`Image ${name}`} draggable={false} />
            {showBox && (
              <div
                className={`bbox ${boxFits(box, image) ? "" : "is-out"}`}
                style={{
                  left: `${(box.x / image.width) * 100}%`,
                  top: `${(box.y / image.height) * 100}%`,
                  width: `${(box.width / image.width) * 100}%`,
                  height: `${(box.height / image.height) * 100}%`,
                }}
              >
                <span className="bbox-tag mono">
                  {box.width}×{box.height}
                </span>
              </div>
            )}
          </div>
        </div>
      ) : (
        <label className="stage is-empty">
          <input className="sr-only" type="file" accept="image/*" onChange={pick} />
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <circle cx="9" cy="9" r="1.6" />
            <path d="m21 15-4.5-4.5L6 21" />
          </svg>
          <span className="empty-title">Drop image {name}</span>
          <span className="empty-sub">
            or <u>browse files</u>
          </span>
        </label>
      )}

      <footer className="panel-foot">
        <span className="file-name" title={image?.file.name}>
          {image ? image.file.name : "PNG, JPG, WEBP or BMP"}
        </span>
        {image && (
          <label className="text-btn">
            Replace
            <input className="sr-only" type="file" accept="image/*" onChange={pick} />
          </label>
        )}
      </footer>
    </section>
  );
}

/* ------------------------------------------------------------------------ */
/* Result report                                                             */
/* ------------------------------------------------------------------------ */
function Result({ result }) {
  const ok = result.result === "SIMILAR";
  const threshold = result.threshold ?? 0.6;
  const metrics = [
    ["Structure", "SSIM on lighting-normalized grayscale", result.details.structural_similarity],
    ["Color", "a/b chroma histogram, brightness ignored", result.details.color_similarity],
  ];

  return (
    <section className={`result ${ok ? "is-ok" : "is-bad"}`} aria-live="polite">
      <div className="result-head">
        <div>
          <span className="eyebrow">Verdict</span>
          <h2 className="verdict">
            <span className="verdict-dot" aria-hidden="true" />
            {result.result}
          </h2>
          <p className="muted">
            {ok
              ? "Both regions show the same visual content."
              : "The regions show different visual content."}
          </p>
        </div>
        <div className="score">
          <span className="eyebrow">Overall score</span>
          <span className="score-num mono">
            {pct(result.score)}
            <small>%</small>
          </span>
        </div>
      </div>

      <div className="meter" role="img" aria-label={`Score ${pct(result.score)}%, threshold ${pct(threshold)}%`}>
        <div className="meter-fill" style={{ width: `${result.score * 100}%` }} />
        <div className="meter-mark" style={{ left: `${threshold * 100}%` }}>
          <span className="mono">threshold {Math.round(threshold * 100)}%</span>
        </div>
      </div>

      <div className="metrics">
        {metrics.map(([label, note, value]) => (
          <div className="metric" key={label}>
            <div className="metric-row">
              <span className="metric-label">{label}</span>
              <span className="mono">{pct(value)}%</span>
            </div>
            <div className="mini-bar">
              <i style={{ width: `${value * 100}%` }} />
            </div>
            <span className="metric-note">{note}</span>
          </div>
        ))}
      </div>

      {result.analysis && <Differences analysis={result.analysis} />}

      <div className="crops">
        {[
          [result.crop1_preview, "Image A", "selected region"],
          [result.crop2_preview, "Image B", "selected region"],
          [result.analysis?.difference_map, "Differences", diffCaption(result.analysis)],
        ]
          .filter(([src]) => src)
          .map(([src, name, note]) => (
            <figure key={name}>
              <img src={src} alt={`${name}: ${note}`} />
              <figcaption>
                <span className="slot">{name}</span>
                <span className="muted">{note}</span>
              </figcaption>
            </figure>
          ))}
      </div>
    </section>
  );
}

const KIND_LABELS = { content: "Content", lighting: "Lighting", color: "Color", detail: "Detail" };

function diffCaption(analysis) {
  if (!analysis) return "";
  return analysis.changed_area > 0
    ? `orange = changed content (${pct(analysis.changed_area)}%)`
    : "no content changes found";
}

/* Plain-language explanation of what differs between the two regions */
function Differences({ analysis }) {
  return (
    <div className="differences">
      <span className="eyebrow">What's different</span>
      <p className="diff-summary">{analysis.summary}</p>
      <ul className="diff-list">
        {analysis.observations.map((o) => (
          <li key={o.title}>
            <span className={`kind kind-${o.kind}`}>{KIND_LABELS[o.kind] || o.kind}</span>
            <div>
              <strong>{o.title}</strong>
              <p>{o.detail}</p>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------------------ */
/* App                                                                       */
/* ------------------------------------------------------------------------ */
export default function App() {
  const [images, setImages] = useState({ a: null, b: null });
  const [box, setBox] = useState({ x: 0, y: 0, width: 100, height: 100 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [api, setApi] = useState("checking");
  const latestFile = useRef({}); // newest file picked per slot

  // Light API status indicator in the top bar.
  useEffect(() => {
    let alive = true;
    const check = () =>
      fetch(`${API_URL}/health`)
        .then((r) => (r.ok ? "online" : "offline"))
        .catch(() => "offline")
        .then((s) => alive && setApi(s));
    check();
    const id = setInterval(check, 15000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const setImage = async (slot, file) => {
    latestFile.current[slot] = file;
    try {
      const next = await readImage(file);
      // A newer file was picked for this slot while this one loaded: drop this one.
      if (latestFile.current[slot] !== file) {
        URL.revokeObjectURL(next.url);
        return;
      }
      setImages((prev) => {
        if (prev[slot]) URL.revokeObjectURL(prev[slot].url); // revoking twice is harmless
        return { ...prev, [slot]: next };
      });
      setResult(null);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  };

  // Any change to the region makes the previous verdict stale.
  const changeBox = (next) => {
    setBox(next);
    setResult(null);
  };

  const updateField = (key, raw) => {
    const value = raw === "" ? "" : Math.trunc(Number(raw));
    changeBox({ ...box, [key]: Number.isNaN(value) ? 0 : value });
  };

  const { a, b } = images;
  const loaded = [a, b].filter(Boolean);
  const numericBox = {
    x: Number(box.x) || 0,
    y: Number(box.y) || 0,
    width: Number(box.width) || 0,
    height: Number(box.height) || 0,
  };

  const fitToImages = () =>
    changeBox({
      x: 0,
      y: 0,
      width: Math.min(...loaded.map((i) => i.width)),
      height: Math.min(...loaded.map((i) => i.height)),
    });

  const outside = loaded.filter((img) => !boxFits(numericBox, img));
  let hint = "Drag across an image to draw the region, or type exact pixel values.";
  if (a && b) {
    hint = outside.length
      ? `Region falls outside ${outside.length === 2 ? "both images" : `image ${outside[0] === a ? "A" : "B"}`}.`
      : `Region fits both images.`;
  }

  const handleCompare = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    const form = new FormData();
    form.append("image1", a.file);
    form.append("image2", b.file);
    Object.entries(numericBox).forEach(([k, v]) => form.append(k, String(v)));

    try {
      const res = await fetch(`${API_URL}/compare`, { method: "POST", body: form });
      const data = await res.json().catch(() => null);
      if (!res.ok || !data) {
        let message = `Request failed with status ${res.status}.`;
        if (typeof data?.detail === "string") message = data.detail;
        else if (Array.isArray(data?.detail))
          message = data.detail.map((d) => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join("; ");
        throw new Error(message);
      }
      setResult(data);
    } catch (err) {
      setError(
        err instanceof TypeError ? `Could not reach the API at ${API_URL}. Is the backend running?` : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="wordmark">
          <span className="logo" aria-hidden="true" />
          Similarity Checker
        </div>
        <span className={`status is-${api}`}>
          <i aria-hidden="true" />
          API {api}
        </span>
      </header>

      <main className="main">
        <div className="intro">
          <h1>Compare a region across two images</h1>
          <p>Brightness, shadows and contrast are normalized first, so only the visual content counts.</p>
        </div>

        <div className="panels">
          <ImagePanel name="A" image={a} box={numericBox} onFile={(f) => setImage("a", f)} onDraw={changeBox} />
          <ImagePanel name="B" image={b} box={numericBox} onFile={(f) => setImage("b", f)} onDraw={changeBox} />
        </div>

        <div className="controls">
          <div className="fields">
            {FIELDS.map(([key, label]) => (
              <label className="field" key={key}>
                <span>{label}</span>
                <input
                  className="mono"
                  type="number"
                  inputMode="numeric"
                  step="1"
                  min={key === "x" || key === "y" ? 0 : 1}
                  value={box[key]}
                  onChange={(e) => updateField(key, e.target.value)}
                  aria-label={key}
                />
              </label>
            ))}
          </div>
          <div className="actions">
            <button type="button" className="btn btn-quiet" onClick={fitToImages} disabled={!loaded.length}>
              Fit to images
            </button>
            <button type="button" className="btn btn-primary" onClick={handleCompare} disabled={!a || !b || loading}>
              {loading && <span className="spinner" aria-hidden="true" />}
              {loading ? "Comparing" : "Compare"}
            </button>
          </div>
        </div>
        <p className={`hint ${a && b && outside.length ? "is-warn" : ""}`}>{hint}</p>

        {error && (
          <div className="alert" role="alert">
            <strong>Couldn't compare</strong>
            <span>{error}</span>
          </div>
        )}

        {result && <Result result={result} />}
      </main>

      <footer className="footnote mono">{API_URL}</footer>
    </div>
  );
}
