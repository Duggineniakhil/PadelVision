"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Image as ImageIcon, Maximize, Minus, Plus, Redo2, Trash2, Undo2, Wand2 } from "lucide-react";
import {
  calibrationImageUrl,
  fitCalibration,
  autoCalibrate,
  snapLine,
  uploadCalibrationImage,
  type AutoCalibration,
  type CalibrationFit,
  type CalibrationImage,
  type CalibrationInput,
} from "@/lib/api";
import { CAL_LINES, CAL_POINTS, COURT_L, COURT_LINES, COURT_W, NET, type CalLine, type CalPoint } from "@/lib/court";
import { num } from "@/lib/format";
import { Badge, Button, Callout, Kbd, Spinner, cx } from "./ui";

type Pt = [number, number];
type SnapMode = "paint" | "edge" | "off";
type Target = { kind: "line"; name: CalLine } | { kind: "point"; name: CalPoint } | null;

/** What the user has done (undo/redo works on this). Snap results are derived from it. */
type Doc = {
  guides: Partial<Record<CalLine, Pt[]>>; // clicked points along a line
  modes: Record<CalLine, SnapMode>;
  loaded: Partial<Record<CalLine, Pt[]>>; // a line's points from the court.json being edited
  points: Partial<Record<CalPoint, Pt>>;
  keepLens: boolean;
};

export type EditorState = {
  body: CalibrationInput | null; // null: not enough constraints yet
  fit: CalibrationFit | null;
  fitError: string | null;
  fitting: boolean;
  dirty: boolean;
};

const MIN_CONSTRAINTS = 4;
const SNAP_MIN = 3;
const LINE_COLOR = "#facc15"; // fitted court
const MARK_COLOR = "#f43f5e"; // constraint points
const DEFAULT_MODES = Object.fromEntries(CAL_LINES.map((l) => [l.name, l.mode])) as Record<CalLine, SnapMode>;

function initialDoc(initial: CalibrationInput | null): Doc {
  return {
    guides: {},
    modes: { ...DEFAULT_MODES },
    loaded: (initial?.named_lines ?? {}) as Doc["loaded"],
    points: (initial?.image_points ?? {}) as Doc["points"],
    keepLens: !!initial?.lens && initial.lens.k1 !== 0,
  };
}

const snapKey = (image: string, name: CalLine, mode: SnapMode, guide: Pt[]) =>
  `${image}|${name}|${mode}|${guide.map((p) => p.join(",")).join(";")}`;

/** Points sorted along their main direction, so a click between two others lands in between. */
function ordered(pts: Pt[]): Pt[] {
  if (pts.length < 3) return pts;
  const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const my = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  let sxx = 0, syy = 0, sxy = 0; // prettier-ignore
  for (const [x, y] of pts) {
    sxx += (x - mx) ** 2;
    syy += (y - my) ** 2;
    sxy += (x - mx) * (y - my);
  }
  const a = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const d = (p: Pt) => (p[0] - mx) * Math.cos(a) + (p[1] - my) * Math.sin(a);
  return [...pts].sort((p, q) => d(p) - d(q));
}

function errorTone(px: number | undefined) {
  if (px == null) return "neutral" as const;
  return px < 2 ? ("success" as const) : px < 4 ? ("warning" as const) : ("danger" as const);
}

/** Click-and-snap court calibration on one image. `children` renders the actions (save, export). */
export function CalibrationEditor({
  image,
  initial,
  children,
}: {
  image: CalibrationImage;
  initial: CalibrationInput | null;
  children: (state: EditorState) => ReactNode;
}) {
  const start = useMemo(() => initialDoc(initial), [initial]);
  const [hist, setHist] = useState<{ doc: Doc; past: Doc[]; future: Doc[] }>({ doc: start, past: [], future: [] });
  const { doc, past, future } = hist;
  const [target, setTarget] = useState<Target>({ kind: "line", name: "near_service_line" });
  const [snaps, setSnaps] = useState<Record<string, Pt[] | "none">>({});
  const [fit, setFit] = useState<CalibrationFit | null>(null);
  const [fitError, setFitError] = useState<string | null>(null);
  const [fitting, setFitting] = useState(false);
  const [showCourt, setShowCourt] = useState(true);
  const pending = useRef(new Set<string>());

  /** An edit (one undo step). */
  const setDoc = useCallback((fn: (d: Doc) => Doc) => {
    setHist((h) => {
      const n = fn(h.doc);
      return n === h.doc ? h : { doc: n, past: [...h.past.slice(-49), h.doc], future: [] };
    });
  }, []);
  /** A change that belongs to the previous undo step (dragging a point). */
  const setDocRaw = useCallback((fn: (d: Doc) => Doc) => setHist((h) => ({ ...h, doc: fn(h.doc) })), []);
  const undo = useCallback(
    () =>
      setHist((h) =>
        h.past.length ? { doc: h.past[h.past.length - 1], past: h.past.slice(0, -1), future: [h.doc, ...h.future] } : h,
      ),
    [],
  );
  const redo = useCallback(
    () => setHist((h) => (h.future.length ? { doc: h.future[0], past: [...h.past, h.doc], future: h.future.slice(1) } : h)),
    [],
  );

  // Snap every line whose guide changed.
  useEffect(() => {
    for (const { name } of CAL_LINES) {
      const guide = doc.guides[name] ?? [];
      const mode = doc.modes[name];
      if (guide.length < 2 || mode === "off") continue;
      const key = snapKey(image.id, name, mode, guide);
      if (snaps[key] || pending.current.has(key)) continue;
      pending.current.add(key);
      snapLine(image.id, guide, mode)
        .then((r) => setSnaps((s) => ({ ...s, [key]: r.points.length >= SNAP_MIN ? r.points : "none" })))
        .catch(() => setSnaps((s) => ({ ...s, [key]: "none" })))
        .finally(() => pending.current.delete(key));
    }
  }, [doc.guides, doc.modes, image.id, snaps]);

  // The constraints actually used, per line: snapped points, else the clicks, else the loaded file's.
  const lineInfo = useMemo(() => {
    const out = {} as Record<CalLine, { pts: Pt[]; source: "snapped" | "clicked" | "loaded" | "pending" | null }>;
    for (const { name } of CAL_LINES) {
      const guide = doc.guides[name] ?? [];
      const mode = doc.modes[name];
      if (guide.length >= 2) {
        const snap = mode === "off" ? undefined : snaps[snapKey(image.id, name, mode, guide)];
        if (Array.isArray(snap)) out[name] = { pts: snap, source: "snapped" };
        else out[name] = { pts: guide, source: mode !== "off" && snap === undefined ? "pending" : "clicked" };
      } else if (guide.length === 0 && doc.loaded[name]?.length) {
        out[name] = { pts: doc.loaded[name]!, source: "loaded" };
      } else out[name] = { pts: [], source: null };
    }
    return out;
  }, [doc, snaps, image.id]);

  const body = useMemo<CalibrationInput | null>(() => {
    const named_lines: Record<string, Pt[]> = {};
    for (const { name } of CAL_LINES) if (lineInfo[name].pts.length >= 2) named_lines[name] = lineInfo[name].pts;
    const image_points = doc.points as Record<string, Pt>;
    if (Object.keys(named_lines).length + Object.keys(image_points).length < MIN_CONSTRAINTS) return null;
    return {
      image_size: [image.width, image.height],
      image_points,
      named_lines,
      lines: initial?.lines ?? [],
      lens: doc.keepLens && initial?.lens ? initial.lens : null,
    };
  }, [lineInfo, doc.points, doc.keepLens, image.width, image.height, initial]);
  const bodyKey = useMemo(() => JSON.stringify(body), [body]);
  const anyPending = Object.values(lineInfo).some((l) => l.source === "pending");

  // Fit (debounced) whenever the constraints change.
  useEffect(() => {
    if (!body || anyPending) return;
    let stale = false;
    const t = setTimeout(() => {
      setFitting(true);
      fitCalibration(body)
        .then((f) => {
          if (stale) return;
          setFit(f);
          setFitError(null);
        })
        .catch((e: Error) => !stale && setFitError(e.message))
        .finally(() => !stale && setFitting(false));
    }, 250);
    return () => {
      stale = true;
      clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- bodyKey stands for body
  }, [bodyKey, anyPending]);

  const dirty = JSON.stringify(doc) !== JSON.stringify(start);
  const shownFit = body ? fit : null;
  const errors = shownFit?.accuracy.constraints_px ?? {};

  // --- editing --------------------------------------------------------------------------------
  const addPoint = (p: Pt) => {
    if (!target) return;
    if (target.kind === "point") {
      setDoc((d) => ({ ...d, points: { ...d.points, [target.name]: p } }));
      return;
    }
    setDoc((d) => ({ ...d, guides: { ...d.guides, [target.name]: ordered([...(d.guides[target.name] ?? []), p]) } }));
  };
  const moveHandle = (h: Handle, p: Pt, record: boolean) => {
    const apply = (d: Doc): Doc =>
      h.kind === "point"
        ? { ...d, points: { ...d.points, [h.name]: p } }
        : { ...d, guides: { ...d.guides, [h.name]: (d.guides[h.name] ?? []).map((q, i) => (i === h.index ? p : q)) } };
    if (record) setDoc(apply);
    else setDocRaw(apply);
  };
  const removeHandle = (h: Handle) =>
    setDoc((d) =>
      h.kind === "point"
        ? { ...d, points: Object.fromEntries(Object.entries(d.points).filter(([k]) => k !== h.name)) }
        : { ...d, guides: { ...d.guides, [h.name]: (d.guides[h.name] ?? []).filter((_, i) => i !== h.index) } },
    );
  const clearLine = (name: CalLine) =>
    setDoc((d) => ({
      ...d,
      guides: { ...d.guides, [name]: [] },
      loaded: Object.fromEntries(Object.entries(d.loaded).filter(([k]) => k !== name)),
    }));
  const removeLast = () => {
    if (!target) return;
    if (target.kind === "point") return removeHandle({ kind: "point", name: target.name });
    const g = doc.guides[target.name] ?? [];
    if (g.length) removeHandle({ kind: "line", name: target.name, index: g.length - 1 });
    else if (doc.loaded[target.name]) clearLine(target.name);
  };

  // Keyboard: undo/redo, delete the last point, deselect.
  const keys = useRef({ undo, redo, removeLast });
  useEffect(() => {
    keys.current = { undo, redo, removeLast };
  });
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof Element && e.target.closest("input, textarea, select, dialog")) return;
      const k = keys.current;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) k.redo();
        else k.undo();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
        e.preventDefault();
        k.redo();
      } else if (e.key === "Backspace" || e.key === "Delete") {
        e.preventDefault();
        k.removeLast();
      } else if (e.key === "Escape") setTarget(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const state: EditorState = { body, fit: shownFit, fitError: body ? fitError : null, fitting, dirty };
  const count = body ? Object.keys(body.named_lines).length + Object.keys(body.image_points).length : 0;

  return (
    <div className="grid gap-4 lg:grid-cols-[300px_minmax(0,1fr)]">
      <aside className="space-y-4">
        <ConstraintList
          doc={doc}
          lineInfo={lineInfo}
          errors={errors}
          target={target}
          onTarget={setTarget}
          onMode={(name, mode) => setDoc((d) => ({ ...d, modes: { ...d.modes, [name]: mode } }))}
          onClear={clearLine}
          onClearPoint={(name) => removeHandle({ kind: "point", name })}
        />
      </aside>

      <div className="min-w-0 space-y-4">
        <Viewport
          image={image}
          doc={doc}
          lineInfo={lineInfo}
          target={target}
          fit={showCourt ? shownFit : null}
          onAdd={addPoint}
          onMove={moveHandle}
          onRemove={removeHandle}
          toolbar={
            <>
              <Button variant="ghost" onClick={undo} disabled={!past.length} title="Undo (Ctrl+Z)" aria-label="Undo">
                <Undo2 className="size-4" />
              </Button>
              <Button variant="ghost" onClick={redo} disabled={!future.length} title="Redo (Ctrl+Y)" aria-label="Redo">
                <Redo2 className="size-4" />
              </Button>
              <label className="ml-2 flex items-center gap-1.5 text-sm text-zinc-600">
                <input type="checkbox" checked={showCourt} onChange={(e) => setShowCourt(e.target.checked)} />
                Fitted court
              </label>
            </>
          }
        />
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <AccuracyPanel state={state} count={count} />
          <div className="space-y-4">
            <LensPanel
              canKeep={!!initial?.lens}
              keep={doc.keepLens}
              onKeep={(keepLens) => setDoc((d) => ({ ...d, keepLens }))}
              fit={shownFit}
              initialK1={initial?.lens?.k1 ?? null}
            />
            {children(state)}
          </div>
        </div>
      </div>
    </div>
  );
}

// --- sidebar -----------------------------------------------------------------------------------
function ConstraintList({
  doc,
  lineInfo,
  errors,
  target,
  onTarget,
  onMode,
  onClear,
  onClearPoint,
}: {
  doc: Doc;
  lineInfo: Record<CalLine, { pts: Pt[]; source: string | null }>;
  errors: Record<string, number>;
  target: Target;
  onTarget: (t: Target) => void;
  onMode: (name: CalLine, mode: SnapMode) => void;
  onClear: (name: CalLine) => void;
  onClearPoint: (name: CalPoint) => void;
}) {
  const [showPoints, setShowPoints] = useState(() => Object.keys(doc.points).length > 0);
  return (
    <div className="rounded-lg border border-zinc-200 bg-white">
      <div className="border-b border-zinc-100 px-4 py-3">
        <h2 className="text-sm font-semibold text-zinc-900">Court lines</h2>
        <p className="mt-0.5 text-xs text-zinc-500">
          Pick a line, then click two or more points along it in the image (more where it curves). Clicks snap
          onto the line. Four lines or points are the minimum; the more, the better.
        </p>
      </div>
      <MiniCourt target={target} doc={doc} lineInfo={lineInfo} />
      <ul className="divide-y divide-zinc-100 border-t border-zinc-100">
        {CAL_LINES.map((l) => {
          const info = lineInfo[l.name];
          const active = target?.kind === "line" && target.name === l.name;
          const err = errors[`line:${l.name}`];
          return (
            <li key={l.name} className={cx(active && "bg-accent-soft/60")}>
              <button
                type="button"
                onClick={() => onTarget({ kind: "line", name: l.name })}
                className="flex w-full items-center gap-2 px-4 py-2 text-left"
              >
                <StatusDot on={info.pts.length >= 2} active={active} />
                <span className="min-w-0 flex-1">
                  <span className={cx("block text-sm", active ? "font-medium text-zinc-900" : "text-zinc-700")}>
                    {l.label}
                  </span>
                  <span className="block truncate text-xs text-zinc-500">
                    {info.source === "pending" ? (
                      "Snapping…"
                    ) : info.pts.length >= 2 ? (
                      <>
                        {info.pts.length} points ·{" "}
                        {info.source === "snapped" ? "snapped" : info.source === "loaded" ? "loaded (detected or opened)" : "as clicked"}
                      </>
                    ) : (doc.guides[l.name]?.length ?? 0) === 1 ? (
                      "1 point: add another"
                    ) : (
                      l.hint
                    )}
                  </span>
                </span>
                {err != null && <Badge tone={errorTone(err)}>{num(err, 1)} px</Badge>}
              </button>
              {active && (
                <div className="flex items-center gap-2 px-4 pb-2.5 pl-8">
                  <SnapToggle value={doc.modes[l.name]} onChange={(m) => onMode(l.name, m)} />
                  {info.pts.length > 0 || (doc.guides[l.name]?.length ?? 0) > 0 ? (
                    <Button variant="ghost" className="ml-auto h-7 px-2" onClick={() => onClear(l.name)}>
                      <Trash2 className="size-3.5" /> Clear
                    </Button>
                  ) : null}
                </div>
              )}
              {active && info.source === "clicked" && doc.modes[l.name] !== "off" && (
                <p className="px-4 pb-2.5 pl-8 text-xs text-amber-700">
                  No line found along these clicks, so the clicks are used as they are. Click closer to the line,
                  or try {doc.modes[l.name] === "paint" ? "Edge" : "Paint"}.
                </p>
              )}
            </li>
          );
        })}
      </ul>
      <div className="border-t border-zinc-100">
        <button
          type="button"
          onClick={() => setShowPoints((s) => !s)}
          className="flex w-full items-center justify-between px-4 py-2.5 text-left text-sm font-medium text-zinc-800 hover:bg-zinc-50"
        >
          Corner and line-end points
          <span className="text-xs font-normal text-zinc-500">
            {Object.keys(doc.points).length ? `${Object.keys(doc.points).length} set` : "optional"}
          </span>
        </button>
        {showPoints && (
          <ul className="divide-y divide-zinc-100 border-t border-zinc-100">
            {CAL_POINTS.map((p) => {
              const active = target?.kind === "point" && target.name === p.name;
              const set = !!doc.points[p.name];
              const err = errors[p.name];
              return (
                <li key={p.name} className={cx("flex items-center", active && "bg-accent-soft/60")}>
                  <button
                    type="button"
                    onClick={() => onTarget({ kind: "point", name: p.name })}
                    className="flex min-w-0 flex-1 items-center gap-2 px-4 py-1.5 text-left"
                  >
                    <StatusDot on={set} active={active} />
                    <span className={cx("truncate text-sm", active ? "font-medium text-zinc-900" : "text-zinc-700")}>
                      {p.label}
                    </span>
                    {err != null && <Badge tone={errorTone(err)}>{num(err, 1)} px</Badge>}
                  </button>
                  {set && (
                    <button
                      type="button"
                      onClick={() => onClearPoint(p.name)}
                      className="mr-3 rounded p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700"
                      aria-label={`Remove ${p.label}`}
                    >
                      <Trash2 className="size-3.5" />
                    </button>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 border-t border-zinc-100 px-4 py-2.5 text-xs text-zinc-500">
        <span>
          <Kbd>Scroll</Kbd> zoom
        </span>
        <span>
          <Kbd>Drag</Kbd> pan or move a point
        </span>
        <span>
          <Kbd>Right-click</Kbd> remove a point
        </span>
        <span>
          <Kbd>Backspace</Kbd> remove the last
        </span>
      </div>
    </div>
  );
}

function StatusDot({ on, active }: { on: boolean; active: boolean }) {
  return (
    <span
      aria-hidden
      className={cx(
        "size-2.5 shrink-0 rounded-full border",
        on ? "border-emerald-600 bg-emerald-500" : active ? "border-accent bg-white" : "border-zinc-300 bg-white",
      )}
    />
  );
}

function SnapToggle({ value, onChange }: { value: SnapMode; onChange: (m: SnapMode) => void }) {
  const opts: [SnapMode, string, string][] = [
    ["paint", "Paint", "Snap to a painted line"],
    ["edge", "Edge", "Snap to where two surfaces meet (floor and wall)"],
    ["off", "Off", "Use the clicks as they are"],
  ];
  return (
    <div className="inline-flex rounded-md border border-zinc-200 bg-white p-0.5" role="radiogroup" aria-label="Snap to">
      {opts.map(([m, label, title]) => (
        <button
          key={m}
          type="button"
          role="radio"
          aria-checked={value === m}
          title={title}
          onClick={() => onChange(m)}
          className={cx(
            "rounded px-2 py-0.5 text-xs font-medium",
            value === m ? "bg-zinc-900 text-white" : "text-zinc-600 hover:text-zinc-900",
          )}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

/** Top-down court: which line or point is being placed, which are done. */
function MiniCourt({
  target,
  doc,
  lineInfo,
}: {
  target: Target;
  doc: Doc;
  lineInfo: Record<CalLine, { pts: Pt[] }>;
}) {
  const s = 9; // px per metre
  const pad = 8;
  const w = COURT_W * s + 2 * pad;
  const h = COURT_L * s + 2 * pad;
  const P = ([x, y]: [number, number]) => [pad + (x + COURT_W / 2) * s, pad + (COURT_L / 2 - y) * s] as const;
  return (
    <div className="flex justify-center bg-zinc-50 py-3">
      <svg viewBox={`0 0 ${w} ${h}`} width={w * 0.85} height={h * 0.85} aria-hidden>
        <rect x={pad} y={pad} width={COURT_W * s} height={COURT_L * s} fill="var(--color-court)" opacity={0.12} />
        {[...COURT_LINES, NET].map(([a, b], i) => {
          const [x1, y1] = P(a);
          const [x2, y2] = P(b);
          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#a1a1aa" strokeWidth={1} strokeDasharray={i === COURT_LINES.length ? "2 2" : undefined} />;
        })}
        {CAL_LINES.map((l) => {
          const active = target?.kind === "line" && target.name === l.name;
          const done = lineInfo[l.name].pts.length >= 2;
          if (!active && !done) return null;
          const [x1, y1] = P(l.seg[0]);
          const [x2, y2] = P(l.seg[1]);
          return (
            <line
              key={l.name}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke={active ? "var(--color-accent)" : "#10b981"}
              strokeWidth={active ? 3.5 : 2}
              strokeLinecap="round"
            />
          );
        })}
        {CAL_POINTS.map((p) => {
          const active = target?.kind === "point" && target.name === p.name;
          const set = !!doc.points[p.name];
          if (!active && !set) return null;
          const [x, y] = P(p.xy);
          return <circle key={p.name} cx={x} cy={y} r={active ? 4.5 : 3} fill={active ? "var(--color-accent)" : "#10b981"} />;
        })}
        <text x={w / 2} y={h - 1} textAnchor="middle" fontSize={8} fill="#71717a">
          camera side
        </text>
      </svg>
    </div>
  );
}

// --- image -------------------------------------------------------------------------------------
type Handle = { kind: "line"; name: CalLine; index: number } | { kind: "point"; name: CalPoint };
type View = { x: number; y: number; w: number };

function Viewport({
  image,
  doc,
  lineInfo,
  target,
  fit,
  onAdd,
  onMove,
  onRemove,
  toolbar,
}: {
  image: CalibrationImage;
  doc: Doc;
  lineInfo: Record<CalLine, { pts: Pt[]; source: string | null }>;
  target: Target;
  fit: CalibrationFit | null;
  onAdd: (p: Pt) => void;
  onMove: (h: Handle, p: Pt, record: boolean) => void;
  onRemove: (h: Handle) => void;
  toolbar: ReactNode;
}) {
  const W = image.width;
  const H = image.height;
  const [view, setView] = useState<View>({ x: 0, y: 0, w: W });
  const [screenW, setScreenW] = useState(1000);
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<
    | { kind: "pan"; sx: number; sy: number; view: View; moved: boolean }
    | { kind: "handle"; handle: Handle; moved: boolean }
    | null
  >(null);
  const [loaded, setLoaded] = useState(false);
  const vh = (view.w * H) / W;
  const k = view.w / screenW; // image px per screen px: keeps marks the same size on screen

  const clampView = useCallback(
    (v: View): View => {
      const w = Math.min(W, Math.max(W / 24, v.w));
      const h = (w * H) / W;
      return { w, x: Math.min(W - w, Math.max(0, v.x)), y: Math.min(H - h, Math.max(0, v.y)) };
    },
    [W, H],
  );

  const toImage = useCallback((clientX: number, clientY: number): Pt | null => {
    const svg = svgRef.current;
    const m = svg?.getScreenCTM();
    if (!svg || !m) return null;
    const p = new DOMPoint(clientX, clientY).matrixTransform(m.inverse());
    return [Math.round(p.x * 100) / 100, Math.round(p.y * 100) / 100];
  }, []);

  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const ro = new ResizeObserver(([e]) => setScreenW(e.contentRect.width || 1000));
    ro.observe(svg);
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const p = toImage(e.clientX, e.clientY);
      if (!p) return;
      const f = Math.exp(e.deltaY * 0.0015);
      setView((v) => {
        const w = Math.min(W, Math.max(W / 24, v.w * f));
        const r = w / v.w;
        return clampView({ w, x: p[0] - (p[0] - v.x) * r, y: p[1] - (p[1] - v.y) * r });
      });
    };
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      ro.disconnect();
      svg.removeEventListener("wheel", onWheel);
    };
  }, [toImage, clampView, W]);

  const zoom = (f: number) =>
    setView((v) => {
      const w = Math.min(W, Math.max(W / 24, v.w * f));
      const cx = v.x + v.w / 2;
      const cy = v.y + (v.w * H) / W / 2;
      return clampView({ w, x: cx - w / 2, y: cy - (w * H) / W / 2 });
    });

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    drag.current = { kind: "pan", sx: e.clientX, sy: e.clientY, view, moved: false };
  };
  const onHandleDown = (e: React.PointerEvent, handle: Handle) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    svgRef.current?.setPointerCapture(e.pointerId);
    drag.current = { kind: "handle", handle, moved: false };
  };
  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    if (!d) return;
    if (d.kind === "pan") {
      const dx = e.clientX - d.sx;
      const dy = e.clientY - d.sy;
      if (!d.moved && Math.hypot(dx, dy) < 4) return;
      d.moved = true;
      const s = d.view.w / screenW;
      setView(clampView({ w: d.view.w, x: d.view.x - dx * s, y: d.view.y - dy * s }));
    } else {
      const p = toImage(e.clientX, e.clientY);
      if (!p) return;
      onMove(d.handle, p, !d.moved); // the first move is one undo step
      d.moved = true;
    }
  };
  const onPointerUp = (e: React.PointerEvent<SVGSVGElement>) => {
    const d = drag.current;
    drag.current = null;
    if (d?.kind === "pan" && !d.moved) {
      const p = toImage(e.clientX, e.clientY);
      if (p && p[0] >= 0 && p[1] >= 0 && p[0] <= W && p[1] <= H) onAdd(p);
    }
  };

  const activeLine = target?.kind === "line" ? target.name : null;
  const prompt = !target
    ? "Pick a line or point on the left"
    : target.kind === "line"
      ? `Click along the ${CAL_LINES.find((l) => l.name === target.name)!.label.toLowerCase()}`
      : `Click the ${CAL_POINTS.find((p) => p.name === target.name)!.label.toLowerCase()}`;

  return (
    <div className="overflow-hidden rounded-lg border border-zinc-200 bg-white">
      <div className="flex items-center gap-1 border-b border-zinc-100 px-2 py-1.5">
        {toolbar}
        <span className="ml-3 truncate text-sm text-zinc-500">{prompt}</span>
        <div className="ml-auto flex items-center gap-1">
          <span className="tabular mr-1 text-xs text-zinc-500">{Math.round((W / view.w) * 100)}%</span>
          <Button variant="ghost" onClick={() => zoom(1 / 1.5)} aria-label="Zoom in" title="Zoom in">
            <Plus className="size-4" />
          </Button>
          <Button variant="ghost" onClick={() => zoom(1.5)} aria-label="Zoom out" title="Zoom out">
            <Minus className="size-4" />
          </Button>
          <Button variant="ghost" onClick={() => setView({ x: 0, y: 0, w: W })} aria-label="Fit to view" title="Fit to view">
            <Maximize className="size-4" />
          </Button>
        </div>
      </div>
      <div className="relative bg-zinc-900">
        {!loaded && (
          <div className="absolute inset-0 flex items-center justify-center">
            <Spinner className="border-zinc-600 border-t-zinc-300" />
          </div>
        )}
        <svg
          ref={svgRef}
          viewBox={`${view.x} ${view.y} ${view.w} ${vh}`}
          className={cx("block w-full touch-none select-none", target ? "cursor-crosshair" : "cursor-grab")}
          style={{ aspectRatio: `${W} / ${H}` }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={() => (drag.current = null)}
          onContextMenu={(e) => e.preventDefault()}
        >
          <image
            href={calibrationImageUrl(image.id)}
            width={W}
            height={H}
            onLoad={() => setLoaded(true)}
            style={{ imageRendering: k < 0.5 ? "pixelated" : "auto" }}
          />
          {fit?.court_lines.map((seg, i) => (
            <polyline
              key={i}
              points={seg.map((p) => p.join(",")).join(" ")}
              fill="none"
              stroke={LINE_COLOR}
              strokeWidth={1.4 * k}
              opacity={0.9}
            />
          ))}
          {fit &&
            Object.entries(fit.keypoints).map(([name, [x, y]]) => (
              <circle key={name} cx={x} cy={y} r={3 * k} fill="none" stroke={LINE_COLOR} strokeWidth={1 * k} />
            ))}
          {CAL_LINES.map(({ name }) => {
            const active = name === activeLine;
            return lineInfo[name].pts.map(([x, y], i) => (
              <circle
                key={`${name}-${i}`}
                cx={x}
                cy={y}
                r={(active ? 2.2 : 1.6) * k}
                fill={MARK_COLOR}
                opacity={active ? 1 : 0.75}
                pointerEvents="none"
              />
            ));
          })}
          {CAL_LINES.flatMap(({ name }) =>
            (doc.guides[name] ?? []).map(([x, y], i) => {
              const active = name === activeLine;
              const h: Handle = { kind: "line", name, index: i };
              return (
                <g key={`g-${name}-${i}`}>
                  {active && <circle cx={x} cy={y} r={11 * k} fill="transparent" className="cursor-move"
                    onPointerDown={(e) => onHandleDown(e, h)}
                    onContextMenu={(e) => { e.preventDefault(); e.stopPropagation(); onRemove(h); }} />}
                  <circle
                    cx={x}
                    cy={y}
                    r={(active ? 5 : 3) * k}
                    fill="none"
                    stroke={active ? "#fff" : "rgba(255,255,255,0.6)"}
                    strokeWidth={(active ? 2 : 1) * k}
                    pointerEvents="none"
                  />
                </g>
              );
            }),
          )}
          {CAL_POINTS.map(({ name }) => {
            const p = doc.points[name];
            if (!p) return null;
            const active = target?.kind === "point" && target.name === name;
            const h: Handle = { kind: "point", name };
            const r = (active ? 9 : 6) * k;
            return (
              <g key={name}>
                <circle cx={p[0]} cy={p[1]} r={12 * k} fill="transparent" className="cursor-move"
                  onPointerDown={(e) => onHandleDown(e, h)}
                  onContextMenu={(e) => { e.preventDefault(); e.stopPropagation(); onRemove(h); }} />
                <g stroke={active ? "#fff" : "#22d3ee"} strokeWidth={(active ? 2 : 1.4) * k} pointerEvents="none">
                  <line x1={p[0] - r} y1={p[1]} x2={p[0] - r / 3} y2={p[1]} />
                  <line x1={p[0] + r / 3} y1={p[1]} x2={p[0] + r} y2={p[1]} />
                  <line x1={p[0]} y1={p[1] - r} x2={p[0]} y2={p[1] - r / 3} />
                  <line x1={p[0]} y1={p[1] + r / 3} x2={p[0]} y2={p[1] + r} />
                </g>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-zinc-100 px-3 py-2 text-xs text-zinc-500">
        <Legend color={MARK_COLOR} dot>
          Line points used
        </Legend>
        <Legend color="#ffffff" ring>
          Your clicks
        </Legend>
        <Legend color={LINE_COLOR}>Fitted court: must sit on the real lines</Legend>
        <span className="tabular ml-auto">
          {W}×{H}
        </span>
      </div>
    </div>
  );
}

function Legend({ color, dot, ring, children }: { color: string; dot?: boolean; ring?: boolean; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      {dot ? (
        <span className="size-2 rounded-full" style={{ background: color }} />
      ) : ring ? (
        <span className="size-2.5 rounded-full border-2 border-zinc-400" />
      ) : (
        <span className="h-0.5 w-4" style={{ background: color }} />
      )}
      {children}
    </span>
  );
}

// --- results -----------------------------------------------------------------------------------
const ZONES: [string, string][] = [
  ["near_half", "Near half"],
  ["net", "At the net"],
  ["far_half", "Far half"],
];

function AccuracyPanel({ state, count }: { state: EditorState; count: number }) {
  const { fit, fitError, fitting, body } = state;
  const acc = fit?.accuracy;
  const near = acc?.zones.near_half?.expected_error_m;
  return (
    <section className="rounded-lg border border-zinc-200 bg-white">
      <header className="flex items-center justify-between border-b border-zinc-100 px-4 py-3">
        <h2 className="text-sm font-semibold text-zinc-900">Accuracy</h2>
        {fitting && <Spinner />}
      </header>
      <div className="p-4">
        {!body ? (
          <p className="text-sm text-zinc-500">
            {count === 0 ? "Nothing placed yet." : "Not enough yet."} Place at least {MIN_CONSTRAINTS} lines or points
            (lines on both sides of the net work best).
          </p>
        ) : fitError ? (
          <p className="text-sm text-red-700">{fitError}</p>
        ) : !acc ? (
          <p className="text-sm text-zinc-500">Fitting…</p>
        ) : (
          <>
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-zinc-600">Fit to your lines and points</span>
              <span className="tabular text-sm font-semibold text-zinc-900">{num(acc.rms_px, 1)} px</span>
            </div>
            <p className="mt-0.5 text-xs text-zinc-500">
              Average distance between what you placed and the fitted court. Under 2 px is good.
            </p>
            <table className="mt-3 w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-zinc-500">
                  <th className="pb-1 font-medium">Expected position error</th>
                  <th className="pb-1 text-right font-medium">on the ground</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {ZONES.map(([key, label]) => {
                  const e = acc.zones[key]?.expected_error_m;
                  return (
                    <tr key={key}>
                      <td className="py-1.5 text-zinc-700">{label}</td>
                      <td className="tabular py-1.5 text-right text-zinc-900">
                        {e == null ? <span className="text-zinc-400">not in view</span> : `${num(e * 100, 0)} cm`}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="mt-3">
              {near == null ? null : near <= 0.15 ? (
                <Badge tone="success">Near half within the 15 cm target</Badge>
              ) : (
                <Badge tone="warning">Near half above the 15 cm target</Badge>
              )}
            </div>
            <p className="mt-2 text-xs text-zinc-500">
              A low camera sees the far half at a shallow angle, so 1 px there is 20–30 cm; that is expected.
            </p>
          </>
        )}
      </div>
    </section>
  );
}

function LensPanel({
  canKeep,
  keep,
  onKeep,
  fit,
  initialK1,
}: {
  canKeep: boolean;
  keep: boolean;
  onKeep: (keep: boolean) => void;
  fit: CalibrationFit | null;
  initialK1: number | null;
}) {
  return (
    <section className="rounded-lg border border-zinc-200 bg-white px-4 py-3">
      <h2 className="text-sm font-semibold text-zinc-900">Lens distortion</h2>
      <div className="mt-2 space-y-1.5 text-sm">
        <label className="flex items-start gap-2">
          <input type="radio" className="mt-1" checked={!keep} onChange={() => onKeep(false)} />
          <span>
            <span className="text-zinc-800">Fit from the lines</span>
            <span className="block text-xs text-zinc-500">
              Needs two or more lines with 3+ points; long lines near the image edges work best.
              {fit && !keep && (
                <>
                  {" "}
                  Now k1 {num(fit.lens.k1, 3)}
                  {fit.line_rms_px != null && <>, lines straight to {num(fit.line_rms_px, 1)} px</>}.
                </>
              )}
            </span>
          </span>
        </label>
        {canKeep && (
          <label className="flex items-start gap-2">
            <input type="radio" className="mt-1" checked={keep} onChange={() => onKeep(true)} />
            <span>
              <span className="text-zinc-800">Keep the saved lens</span>
              <span className="block text-xs text-zinc-500">
                k1 {num(initialK1, 3)}. Best when only adjusting a few lines: refitting from short far lines can
                overfit.
              </span>
            </span>
          </label>
        )}
      </div>
    </section>
  );
}

// --- shared pieces for the pages ---------------------------------------------------------------
/** "Detect automatically": finds the court and hands the result to the editor to check. */
export function AutoDetect({
  image,
  onResult,
}: {
  image: CalibrationImage;
  onResult: (calibration: CalibrationInput) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<AutoCalibration | null>(null);
  const [error, setError] = useState<string | null>(null);
  const run = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await autoCalibrate(image.id);
      setResult(r);
      if (r.calibration) onResult(r.calibration);
    } catch (e) {
      setError((e as Error).message);
    }
    setBusy(false);
  };
  const found = result ? Object.keys(result.lines).length : 0;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={run} disabled={busy}>
          {busy ? <Spinner /> : <Wand2 className="size-4" />}
          {busy ? "Looking for the court…" : "Detect automatically"}
        </Button>
        <span className="text-xs text-zinc-500">
          {busy
            ? "Fitting the court to the painted lines, the floor and the net. This takes a minute or two."
            : "Finds the court lines for you; check them before saving."}
        </span>
      </div>
      {error && (
        <Callout level="warning" title="Detection failed">
          {error}
        </Callout>
      )}
      {result &&
        (result.ok ? (
          <Callout level="info" title={`Court found: ${found} lines, fit ${num(result.accuracy?.rms_px, 1)} px`}>
            It passed the quality checks. Check that the yellow court sits on the real lines, then save or copy it.
          </Callout>
        ) : (
          <Callout
            level="warning"
            title={result.calibration ? "Court found, but not reliably" : "The court was not found"}
          >
            {result.reasons.join("; ")}.{" "}
            {result.calibration
              ? "The lines it found are loaded below: fix the ones that are off."
              : "Place the lines by hand."}
          </Callout>
        ))}
    </div>
  );
}

/** Drop or choose a court image (png/jpg) or notebook 08's calibration.zip. */
export function CourtImageDrop({
  onImage,
  hint,
  check,
}: {
  onImage: (image: CalibrationImage) => void;
  hint?: ReactNode;
  /** Return an error message to reject an image (e.g. the wrong resolution). */
  check?: (image: CalibrationImage) => string | null;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const take = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const img = await uploadCalibrationImage(file);
      const problem = check?.(img) ?? null;
      if (problem) setError(problem);
      else onImage(img);
    } catch (e) {
      setError((e as Error).message);
    }
    setBusy(false);
  };
  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!busy) void take(e.dataTransfer.files[0]);
        }}
        onClick={() => !busy && input.current?.click()}
        className={cx(
          "flex cursor-pointer flex-col items-center rounded-md border border-dashed px-4 py-10 text-center",
          dragging ? "border-accent bg-accent-soft" : "border-zinc-300 bg-white hover:bg-zinc-50",
        )}
      >
        {busy ? <Spinner /> : <ImageIcon className="size-6 text-zinc-400" />}
        <div className="mt-2 text-sm font-medium text-zinc-800">
          {busy ? "Reading the image…" : "Drop an image of the court here, or click to choose"}
        </div>
        <div className="mt-0.5 max-w-md text-xs text-zinc-500">
          {hint ?? "PNG or JPG, or the calibration.zip from the full-match notebook (its background.png is used)."}
        </div>
        <input
          ref={input}
          type="file"
          accept=".png,.jpg,.jpeg,.zip"
          hidden
          onChange={(e) => {
            void take(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>
      {error && <p className="mt-2 text-sm text-red-700">{error}</p>}
    </div>
  );
}

export function downloadJson(data: unknown, filename: string) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** A court.json (as saved by the pipeline) -> the editor's input. Throws on anything else. */
export function parseCourtJson(text: string): CalibrationInput {
  const d = JSON.parse(text);
  if (!Array.isArray(d?.image_size) || d.image_size.length !== 2) throw new Error("Not a court.json: no image_size.");
  return {
    image_size: [Number(d.image_size[0]), Number(d.image_size[1])],
    image_points: d.image_points ?? {},
    named_lines: d.named_lines ?? {},
    lines: d.lines ?? [],
    lens: d.lens ?? null,
  };
}
