"use client";

import { useMemo, type ReactNode } from "react";
import type { Shot } from "@/lib/api";
import { COURT_L, COURT_LINES, COURT_W, NET, TEAM_COLOR } from "@/lib/court";

const MARGIN = 1;
const HW = COURT_W / 2;
const HL = COURT_L / 2;

/** Top-down court in metres; the camera (near) side is at the bottom. Children draw in court
 * coordinates via `pt()` (x, y in metres -> SVG x, y). */
export function CourtSvg({
  children,
  className,
  label,
}: {
  children?: ReactNode;
  className?: string;
  label: string;
}) {
  return (
    <svg
      viewBox={`${-HW - MARGIN} ${-HL - MARGIN} ${COURT_W + 2 * MARGIN} ${COURT_L + 2 * MARGIN}`}
      className={className}
      role="img"
      aria-label={label}
    >
      <rect
        x={-HW - MARGIN}
        y={-HL - MARGIN}
        width={COURT_W + 2 * MARGIN}
        height={COURT_L + 2 * MARGIN}
        rx={0.4}
        fill="#24446c"
      />
      <rect x={-HW} y={-HL} width={COURT_W} height={COURT_L} fill="var(--color-court)" />
      {children}
      {COURT_LINES.map(([[x0, y0], [x1, y1]], i) => (
        <line key={i} x1={x0} y1={-y0} x2={x1} y2={-y1} stroke="#f1f5f9" strokeWidth={0.07} />
      ))}
      <line
        x1={NET[0][0] - 0.3}
        y1={0}
        x2={NET[1][0] + 0.3}
        y2={0}
        stroke="#e2e8f0"
        strokeWidth={0.14}
        strokeDasharray="0.25 0.12"
      />
    </svg>
  );
}

export const pt = (x: number, y: number) => ({ x, y: -y });

// --- heatmap ---------------------------------------------------------------------------------
const CELL = 0.25;
const RAMP: [number, [number, number, number]][] = [
  [0, [253, 230, 138]],
  [0.5, [249, 115, 22]],
  [1, [190, 18, 60]],
];

function rampRgb(v: number): [number, number, number] {
  for (let i = 1; i < RAMP.length; i++) {
    const [t1, c1] = RAMP[i];
    const [t0, c0] = RAMP[i - 1];
    if (v <= t1) {
      const f = (v - t0) / (t1 - t0);
      return c0.map((c, k) => Math.round(c + (c1[k] - c) * f)) as [number, number, number];
    }
  }
  return RAMP[RAMP.length - 1][1];
}

function blur(grid: Float32Array, w: number, h: number, r: number) {
  const tmp = new Float32Array(grid.length);
  for (let pass = 0; pass < 2; pass++) {
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let k = -r; k <= r; k++) s += grid[y * w + Math.min(w - 1, Math.max(0, x + k))];
        tmp[y * w + x] = s / (2 * r + 1);
      }
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let k = -r; k <= r; k++) s += tmp[Math.min(h - 1, Math.max(0, y + k)) * w + x];
        grid[y * w + x] = s / (2 * r + 1);
      }
  }
}

/** Where a player spent their time, from smoothed court positions (metres). Rendered as one
 * small image that the browser scales smoothly. */
export function HeatmapLayer({ x, y }: { x: number[]; y: number[] }) {
  const x0 = -HW - MARGIN;
  const y0 = -HL - MARGIN;
  const w = Math.round((COURT_W + 2 * MARGIN) / CELL);
  const h = Math.round((COURT_L + 2 * MARGIN) / CELL);

  // Tracks arrive from the API in the browser, so this never runs during server rendering.
  const href = useMemo(() => {
    if (typeof document === "undefined") return null;
    const grid = new Float32Array(w * h);
    for (let i = 0; i < x.length; i++) {
      const cx = Math.floor((x[i] - x0) / CELL);
      const cy = Math.floor((y[i] - y0) / CELL);
      if (cx >= 0 && cx < w && cy >= 0 && cy < h) grid[cy * w + cx] += 1;
    }
    blur(grid, w, h, 2);
    let max = 0;
    for (const v of grid) max = Math.max(max, v);
    if (!max) return null;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const img = ctx.createImageData(w, h);
    for (let cy = 0; cy < h; cy++)
      for (let cx = 0; cx < w; cx++) {
        const v = grid[cy * w + cx] / max;
        const row = h - 1 - cy; // court y grows upwards, image rows downwards
        const o = (row * w + cx) * 4;
        const [r, g, b] = rampRgb(v);
        img.data[o] = r;
        img.data[o + 1] = g;
        img.data[o + 2] = b;
        img.data[o + 3] = v < 0.04 ? 0 : Math.round(255 * Math.min(0.92, 0.2 + v));
      }
    ctx.putImageData(img, 0, 0);
    return canvas.toDataURL();
  }, [x, y, w, h, x0, y0]);

  if (!href) return null;
  return (
    <image
      href={href}
      x={x0}
      y={-(y0 + h * CELL)}
      width={w * CELL}
      height={h * CELL}
      preserveAspectRatio="none"
    />
  );
}

// --- shots -----------------------------------------------------------------------------------
/** Each shot: numbered hitter position; arrow to the landing bounce when one was seen;
 * hollow when it was probably volleyed back before bouncing. */
export function ShotLayer({
  shots,
  highlight,
  onSelect,
}: {
  shots: Shot[];
  highlight?: number | null;
  onSelect?: (shot: Shot) => void;
}) {
  return (
    <g>
      <defs>
        {(["near", "far"] as const).map((team) => (
          <marker
            key={team}
            id={`arrow-${team}`}
            viewBox="0 0 10 10"
            refX={9}
            refY={5}
            markerWidth={3}
            markerHeight={3}
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill={TEAM_COLOR[team]} />
          </marker>
        ))}
      </defs>
      {shots.map((s) => {
        if (!s.hitter_xy || !s.bounce) return null;
        const a = pt(s.hitter_xy[0], s.hitter_xy[1]);
        const b = pt(s.bounce.x_m, s.bounce.y_m);
        const dim = highlight != null && highlight !== s.index;
        return (
          <g key={`l${s.index}`} opacity={dim ? 0.25 : 1}>
            <line
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={TEAM_COLOR[s.team]}
              strokeWidth={0.12}
              markerEnd={`url(#arrow-${s.team})`}
            />
            <circle cx={b.x} cy={b.y} r={0.18} fill="#fff" stroke={TEAM_COLOR[s.team]} strokeWidth={0.08} />
          </g>
        );
      })}
      {shots.map((s) => {
        if (!s.hitter_xy) return null;
        const a = pt(s.hitter_xy[0], s.hitter_xy[1]);
        const volley = s.no_landing === "probably_volley";
        const dim = highlight != null && highlight !== s.index;
        const color = TEAM_COLOR[s.team];
        return (
          <g
            key={`p${s.index}`}
            opacity={dim ? 0.3 : 1}
            className={onSelect ? "cursor-pointer" : undefined}
            onClick={() => onSelect?.(s)}
          >
            <circle
              cx={a.x}
              cy={a.y}
              r={0.42}
              fill={volley ? "#24446c" : color}
              stroke={volley ? color : "#fff"}
              strokeWidth={volley ? 0.12 : 0.06}
            />
            <text
              x={a.x}
              y={a.y + 0.17}
              textAnchor="middle"
              fontSize={0.48}
              fontWeight={600}
              fill={volley ? color : "#fff"}
            >
              {s.index}
            </text>
          </g>
        );
      })}
    </g>
  );
}

/** Landing points of a team's shots. */
export function LandingLayer({ shots, team }: { shots: Shot[]; team: "near" | "far" }) {
  return (
    <g>
      {shots
        .filter((s) => s.team === team && s.bounce)
        .map((s) => {
          const b = pt(s.bounce!.x_m, s.bounce!.y_m);
          return (
            <circle
              key={`${s.rally_id}-${s.index}`}
              cx={b.x}
              cy={b.y}
              r={0.26}
              fill={TEAM_COLOR[team]}
              stroke="#fff"
              strokeWidth={0.07}
              opacity={0.9}
            />
          );
        })}
    </g>
  );
}
