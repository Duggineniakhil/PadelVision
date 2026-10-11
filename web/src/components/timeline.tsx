"use client";

import { useState } from "react";
import type { Rally, MatchEvent } from "@/lib/api";
import { clock } from "@/lib/format";
import { cx } from "./ui";

/** The whole match as a bar: rallies as segments, optional hit/bounce ticks and a playhead.
 * Click a rally to select it; click elsewhere to seek (when onSeek is given). */
export function RallyTimeline({
  duration,
  rallies,
  events,
  current,
  selected,
  onSelect,
  onSeek,
  height = "h-10",
}: {
  duration: number;
  rallies: Rally[];
  events?: MatchEvent[];
  current?: number;
  selected?: number | null;
  onSelect?: (rally: Rally) => void;
  onSeek?: (seconds: number) => void;
  height?: string;
}) {
  const [hover, setHover] = useState<{ x: number; t: number; rally?: Rally } | null>(null);
  const pos = (t: number) => `${(100 * t) / Math.max(duration, 1)}%`;
  const ticks = tickStep(duration);

  return (
    <div className="select-none">
      <div
        className={cx("relative rounded-md border border-zinc-200 bg-zinc-50", height, onSeek && "cursor-pointer")}
        onMouseMove={(e) => {
          const box = e.currentTarget.getBoundingClientRect();
          const t = ((e.clientX - box.left) / box.width) * duration;
          setHover({ x: e.clientX - box.left, t, rally: rallies.find((r) => t >= r.start_s && t <= r.end_s) });
        }}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const box = e.currentTarget.getBoundingClientRect();
          const t = ((e.clientX - box.left) / box.width) * duration;
          const r = rallies.find((x) => t >= x.start_s - 1 && t <= x.end_s + 1);
          if (r && onSelect) onSelect(r);
          else onSeek?.(t);
        }}
      >
        {rallies.map((r) => (
          <div
            key={r.id}
            className={cx(
              "absolute inset-y-1.5 rounded-sm",
              selected === r.id ? "bg-accent" : "bg-zinc-400/70 hover:bg-zinc-500",
            )}
            style={{ left: pos(r.start_s), width: `max(2px, ${pos(r.end_s - r.start_s)})` }}
          />
        ))}
        {events
          ?.filter((e) => e.kind === "hit" || e.kind === "bounce")
          .map((e, i) => (
            <div
              key={i}
              className={cx("absolute bottom-0 h-1.5 w-px", e.kind === "hit" ? "bg-red-500" : "bg-emerald-500")}
              style={{ left: pos(e.t) }}
            />
          ))}
        {current != null && (
          <div className="absolute inset-y-0 w-0.5 bg-zinc-900" style={{ left: pos(current) }} />
        )}
        {hover && (
          <div
            className="pointer-events-none absolute bottom-full z-10 mb-1.5 -translate-x-1/2 whitespace-nowrap rounded bg-zinc-900 px-2 py-1 text-xs text-white"
            style={{ left: hover.x }}
          >
            <span className="tabular">{clock(hover.t)}</span>
            {hover.rally && (
              <span className="text-zinc-300">
                {" "}
                · Rally {hover.rally.id} · {hover.rally.hits} hits
              </span>
            )}
          </div>
        )}
      </div>
      <div className="tabular relative mt-1 h-4 text-[11px] text-zinc-400">
        {Array.from({ length: Math.floor(duration / ticks) + 1 }, (_, i) => i * ticks).map((t) => (
          <span key={t} className="absolute -translate-x-1/2" style={{ left: pos(t) }}>
            {clock(t)}
          </span>
        ))}
      </div>
    </div>
  );
}

function tickStep(duration: number) {
  for (const s of [10, 30, 60, 120, 300, 600, 900]) if (duration / s <= 10) return s;
  return 1800;
}
