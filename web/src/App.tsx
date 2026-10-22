import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, ChangeEvent, DragEvent, MouseEvent } from "react";

type Mode = "color" | "bokeh" | "color+bokeh";
type Crop = { left: number; top: number; right: number; bottom: number };
type Suggestion = {
  crop: Crop;
  score: number;
  improvement_over_original: number;
};
type TaskInfo = {
  id: string;
  status: string;
  stage: string;
  progress: number;
  error: string | null;
  access_token?: string;
  status_url?: string;
  cancel_url?: string;
  result?: { width: number; height: number };
  preview_url?: string;
  download_url?: string;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const APERTURES = [0.95, 1.8, 2.0, 2.8, 4.0, 5.6];
const FULL_FRAME: Crop = { left: 0, top: 0, right: 1, bottom: 1 };

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    return body.detail ?? `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

function cropTuple(crop: Crop): number[] {
  return [crop.left, crop.top, crop.right, crop.bottom];
}

function isFullFrame(crop: Crop): boolean {
  return crop.left === 0 && crop.top === 0 && crop.right === 1 && crop.bottom === 1;
}

function sleep(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [sourceSize, setSourceSize] = useState({ width: 0, height: 0 });
  const [crop, setCrop] = useState<Crop>(FULL_FRAME);
  const [mode, setMode] = useState<Mode>("color+bokeh");
  const [focus, setFocus] = useState({ x: 0.5, y: 0.5 });
  const [aperture, setAperture] = useState(2.8);
  const [comparison, setComparison] = useState(50);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [task, setTask] = useState<TaskInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [serviceOnline, setServiceOnline] = useState<boolean | null>(null);
  const frameRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let mounted = true;
    fetch(`${API_BASE}/api/v1/health`)
      .then((response) => {
        if (mounted) setServiceOnline(response.ok);
      })
      .catch(() => {
        if (mounted) setServiceOnline(false);
      });
    return () => { mounted = false; };
  }, []);

  useEffect(() => {
    if (!file) {
      setOriginalUrl(null);
      setSourceSize({ width: 0, height: 0 });
      return;
    }
    const url = URL.createObjectURL(file);
    setOriginalUrl(url);
    setResultUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return null;
    });
    setTask(null);
    setSuggestions([]);
    setCrop(FULL_FRAME);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(
    () => () => {
      if (resultUrl) URL.revokeObjectURL(resultUrl);
    },
    [resultUrl],
  );

  const displayedCrop = crop;
  const frameStyle = useMemo<CSSProperties>(() => {
    if (!sourceSize.width || !sourceSize.height) return {};
    const width = sourceSize.width * displayedCrop.right - sourceSize.width * displayedCrop.left;
    const height = sourceSize.height * displayedCrop.bottom - sourceSize.height * displayedCrop.top;
    return { aspectRatio: `${width} / ${height}` };
  }, [displayedCrop, sourceSize]);

  const originalLayerStyle = useMemo<CSSProperties>(() => {
    const cropWidth = displayedCrop.right - displayedCrop.left;
    const cropHeight = displayedCrop.bottom - displayedCrop.top;
    return {
      width: `${100 / cropWidth}%`,
      height: `${100 / cropHeight}%`,
      left: `${(-displayedCrop.left / cropWidth) * 100}%`,
      top: `${(-displayedCrop.top / cropHeight) * 100}%`,
    };
  }, [displayedCrop]);

  const originalMarker = {
    left: `${((focus.x - displayedCrop.left) / (displayedCrop.right - displayedCrop.left)) * 100}%`,
    top: `${((focus.y - displayedCrop.top) / (displayedCrop.bottom - displayedCrop.top)) * 100}%`,
  };

  function chooseFile(nextFile: File | null) {
    setError(null);
    if (nextFile && !["image/jpeg", "image/png", "image/webp"].includes(nextFile.type)) {
      setError("Choose a JPEG, PNG, or WebP image.");
      return;
    }
    setFile(nextFile);
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    chooseFile(event.target.files?.[0] ?? null);
    event.target.value = "";
  }

  function onPreviewClick(event: MouseEvent<HTMLDivElement>) {
    if (mode === "color" || !frameRef.current) return;
    const bounds = frameRef.current.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const visibleX = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width));
    const visibleY = Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height));
    setFocus({
      x: Math.min(0.999999, displayedCrop.left + visibleX * (displayedCrop.right - displayedCrop.left)),
      y: Math.min(0.999999, displayedCrop.top + visibleY * (displayedCrop.bottom - displayedCrop.top)),
    });
  }

  async function requestComposition() {
    if (!file || busy) return;
    setBusy(true);
    setError(null);
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("strategy", "low-memory");
      const response = await fetch(`${API_BASE}/api/v1/compositions`, {
        method: "POST",
        body,
      });
      if (!response.ok) throw new Error(await readError(response));
      const result = (await response.json()) as { suggestions: Suggestion[] };
      setSuggestions(result.suggestions);
      if (!result.suggestions.length) setError("The composition model returned no crop suggestions.");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not analyze composition.");
    } finally {
      setBusy(false);
    }
  }

  async function processPhoto() {
    if (!file || busy) return;
    setBusy(true);
    setError(null);
    setTask(null);
    setResultUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return null;
    });
    try {
      const body = new FormData();
      body.append("file", file);
      body.append(
        "options",
        JSON.stringify({
          mode,
          strategy: "low-memory",
          focus_x: focus.x,
          focus_y: focus.y,
          aperture,
          crop: isFullFrame(crop) ? undefined : cropTuple(crop),
        }),
      );
      const createResponse = await fetch(`${API_BASE}/api/v1/tasks`, {
        method: "POST",
        body,
      });
      if (!createResponse.ok) throw new Error(await readError(createResponse));
      const created = (await createResponse.json()) as TaskInfo;
      if (!created.access_token || !created.status_url || !created.cancel_url) {
        throw new Error("The processing service returned an incomplete task response.");
      }
      const accessToken = created.access_token;
      setTask(created);

      for (let attempt = 0; attempt < 3600; attempt += 1) {
        await sleep(750);
        const statusResponse = await fetch(`${API_BASE}${created.status_url}`, {
          headers: { "X-Laika-Task-Token": accessToken },
        });
        if (!statusResponse.ok) throw new Error(await readError(statusResponse));
        const current = (await statusResponse.json()) as TaskInfo;
        current.access_token = accessToken;
        current.cancel_url = created.cancel_url;
        setTask(current);
        if (current.status === "failed") throw new Error(current.error ?? "Processing failed.");
        if (current.status === "cancelled") break;
        if (current.status === "completed" && current.preview_url) {
          const previewResponse = await fetch(`${API_BASE}${current.preview_url}`, {
            headers: { "X-Laika-Task-Token": accessToken },
          });
          if (!previewResponse.ok) throw new Error(await readError(previewResponse));
          const previewBlob = await previewResponse.blob();
          setResultUrl(URL.createObjectURL(previewBlob));
          break;
        }
        if (attempt === 3599) throw new Error("This task is taking longer than expected.");
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not process this photo.");
    } finally {
      setBusy(false);
    }
  }

  async function cancelTask() {
    if (!task?.cancel_url || !task.access_token) return;
    try {
      const response = await fetch(`${API_BASE}${task.cancel_url}`, {
        method: "POST",
        headers: { "X-Laika-Task-Token": task.access_token },
      });
      if (!response.ok) throw new Error(await readError(response));
      setTask((await response.json()) as TaskInfo);
      setTask((current) =>
        current ? { ...current, access_token: task.access_token, cancel_url: task.cancel_url } : current,
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not cancel the task.");
    }
  }

  async function downloadResult() {
    if (!task?.download_url || !task.access_token) return;
    try {
      const response = await fetch(`${API_BASE}${task.download_url}`, {
        headers: { "X-Laika-Task-Token": task.access_token },
      });
      if (!response.ok) throw new Error(await readError(response));
      const url = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "leicashot-result.png";
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not download the result.");
    }
  }

  function onDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files[0] ?? null);
  }

  const activeCrop = !isFullFrame(crop);

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="LeicaShot home">
          <span className="brand-mark">L</span>
          <span>LEICASHOT</span>
        </a>
        <div className={`service-indicator ${serviceOnline ? "online" : serviceOnline === false ? "offline" : "checking"}`}>
          <span /> {serviceOnline === false ? "Engine offline" : serviceOnline ? "Python image engine" : "Connecting to engine"}
        </div>
      </header>

      <div className="workspace" id="top">
        <aside className="control-panel">
          <div className="eyebrow">PHOTO LAB <span>01 / EDIT</span></div>
          <h1>Shape the light.<br /><em>Keep the detail.</em></h1>
          <p className="intro">A CPU-first photo workflow, rendered at the photo’s original size.</p>

          <label
            className={`upload-card ${dragging ? "is-dragging" : ""} ${file ? "has-file" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <input aria-label="Choose a photograph" type="file" accept="image/jpeg,image/png,image/webp" onChange={onFileChange} />
            <span className="upload-icon">↥</span>
            <span className="upload-title">{file ? file.name : "Choose a photograph"}</span>
            <span className="upload-caption">{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB · change image` : "Drop JPEG, PNG, or WebP here · up to 50 MB"}</span>
          </label>

          <section className="control-section">
            <div className="section-heading"><span>01</span><h2>Render</h2></div>
            <label className="field-label" htmlFor="mode">Processing mode</label>
            <select id="mode" value={mode} disabled={busy} onChange={(event) => setMode(event.target.value as Mode)}>
              <option value="color+bokeh">M9 color + depth blur</option>
              <option value="color">M9 color only</option>
              <option value="bokeh">Depth blur only</option>
            </select>

            <div className="field-row">
              <label className="field-label" htmlFor="aperture">Aperture</label>
              <select id="aperture" value={aperture} disabled={busy || mode === "color"} onChange={(event) => setAperture(Number(event.target.value))}>
                {APERTURES.map((value) => <option key={value} value={value}>ƒ / {value.toFixed(value % 1 ? 2 : 1)}</option>)}
              </select>
            </div>
            <p className="control-hint">Click the photo to place focus. Aperture changes the depth-of-field strength.</p>
          </section>

          <section className="control-section composition-section">
            <div className="section-heading"><span>02</span><h2>Composition</h2><small>GAIC</small></div>
            <button className="secondary-button" type="button" disabled={!file || busy} onClick={requestComposition}>
              {busy ? <span className="spinner" /> : <span className="sparkle">✳</span>}
              Suggest a crop
            </button>
            {suggestions.length > 0 && (
              <div className="suggestion-list" aria-label="Composition suggestions">
                {suggestions.slice(0, 5).map((suggestion, index) => {
                  const chosen = cropTuple(crop).every((value, i) => Math.abs(value - cropTuple(suggestion.crop)[i]) < 1e-6);
                  return (
                    <button
                      className={`suggestion ${chosen ? "selected" : ""}`}
                      key={`${suggestion.crop.left}-${suggestion.crop.top}-${index}`}
                      type="button"
                      disabled={busy}
                      onClick={() => setCrop(suggestion.crop)}
                    >
                      <span className="suggestion-rank">0{index + 1}</span>
                      <span>Crop {index + 1}<small>+{(suggestion.improvement_over_original * 100).toFixed(1)} score</small></span>
                      <span className="suggestion-score">{suggestion.score.toFixed(2)}</span>
                    </button>
                  );
                })}
              </div>
            )}
            {activeCrop && <button className="text-button" type="button" disabled={busy} onClick={() => setCrop(FULL_FRAME)}>Clear selected crop</button>}
          </section>

          <section className="control-section focus-section">
            <div className="section-heading"><span>03</span><h2>Focus point</h2></div>
            <div className="focus-readout"><span>Click the image to move focus</span><span>{Math.round(focus.x * 100)} · {Math.round(focus.y * 100)}</span></div>
          </section>

          <div className="actions">
            <button className="primary-button" type="button" disabled={!file || busy} onClick={processPhoto}>
              {busy ? <span className="spinner" /> : null}
              {busy ? "Processing photo" : "Process photo"}<span aria-hidden="true">→</span>
            </button>
            {busy && task?.cancel_url && <button className="text-button cancel-button" type="button" onClick={cancelTask}>Cancel task</button>}
            {resultUrl && <button className="download-button" type="button" onClick={downloadResult}>↓ <span>Download full PNG</span></button>}
          </div>
          {task && <div className="task-status" role="status">
            <div className="status-line"><span>{task.stage || task.status}</span><span>{Math.round(task.progress * 100)}%</span></div>
            <div className="progress-track"><span style={{ width: `${Math.round(task.progress * 100)}%` }} /></div>
          </div>}
          {error && <div className="error-message" role="alert">{error}</div>}
          <footer className="panel-footer">M9 · Depth Anything V2 · BokehMe ARNet</footer>
        </aside>

        <section className="preview-panel" aria-label="Photo preview">
          <div className="preview-topline">
            <span>{file ? "PREVIEW" : "YOUR CANVAS"}</span>
            {file && <span>{sourceSize.width} × {sourceSize.height} px</span>}
          </div>
          {originalUrl ? (
            <>
              <div className={`preview-stage ${resultUrl ? "has-result" : ""}`}>
                <div
                  className="comparison-frame"
                  ref={frameRef}
                  style={frameStyle}
                  onClick={onPreviewClick}
                  role="img"
                  aria-label="Photo preview. Click to set the depth focus point."
                >
                  <img
                    className="original-image"
                    src={originalUrl}
                    alt="Original photograph"
                    style={originalLayerStyle}
                    onLoad={(event) => setSourceSize({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })}
                  />
                  {resultUrl && <img className="result-image" src={resultUrl} alt="Processed photograph" style={{ clipPath: `inset(0 ${100 - comparison}% 0 0)` }} />}
                  {mode !== "color" && <span className="focus-marker" style={originalMarker}><i /></span>}
                  {resultUrl && <span className="comparison-divider" style={{ left: `${comparison}%` }} />}
                  {!resultUrl && <span className="image-badge">ORIGINAL</span>}
                  {resultUrl && <><span className="image-badge original-badge">BEFORE</span><span className="image-badge result-badge">AFTER</span></>}
                </div>
              </div>
              {resultUrl ? (
                <div className="compare-control">
                  <span>Original</span>
                  <input aria-label="Before and after comparison" type="range" min="0" max="100" value={comparison} onChange={(event) => setComparison(Number(event.target.value))} />
                  <span>Laika result</span>
                </div>
              ) : <p className="preview-hint">{mode === "color" ? "Upload a photo to begin." : "Click anywhere in the frame to choose the focus point."}</p>}
            </>
          ) : (
            <div className="empty-canvas" onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
              <div className="empty-orbit orbit-one" /><div className="empty-orbit orbit-two" />
              <div className="empty-center"><span>✳</span><p>Start with a photograph</p><small>Your original resolution stays intact.</small></div>
            </div>
          )}
          <div className="preview-caption"><span>CPU-FIRST · FULL RESOLUTION</span><span>LOW MEMORY MODE</span></div>
        </section>
      </div>
    </main>
  );
}
