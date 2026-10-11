"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Video as VideoIcon } from "lucide-react";
import {
  fetchOverlay,
  useEvents,
  useHighlights,
  useRallies,
  videoUrl,
  type CourtInfo,
  type MatchEvent,
  type OverlayResponse,
  type Rally,
  type Shot,
} from "@/lib/api";
import { clock, num } from "@/lib/format";
import { PLAYER_TEAM, TEAM_COLOR } from "@/lib/court";
import { AttachVideoDialog, useMatch } from "@/components/match";
import { RallyTimeline } from "@/components/timeline";
import { Button, ButtonLink, EmptyState, Kbd, Skeleton, TeamDot, cx } from "@/components/ui";

const WINDOW_S = 60; // overlay data is fetched in windows of this length
const BALL_TRAIL = 8; // frames
const EVENT_SHOW_S = 0.5;
const LAYERS = ["players", "ball", "court", "events"] as const;
type Layer = (typeof LAYERS)[number];
const LAYER_LABEL: Record<Layer, string> = { players: "Players", ball: "Ball", court: "Court lines", events: "Hits and bounces" };

export function VideoPage() {
  return (
    <Suspense fallback={<Skeleton className="aspect-video w-full" />}>
      <VideoView />
    </Suspense>
  );
}

function VideoView() {
  const { id, data: run, ready } = useMatch();
  const { data: hl } = useHighlights(id, ready);
  const [attachOpen, setAttachOpen] = useState(false);
  if (!run) return null;
  if (!run.has_video)
    return (
      <>
        <EmptyState
          icon={<VideoIcon className="size-8" />}
          title="Add the match video to watch it with overlays"
          action={
            <div className="flex gap-2">
              <Button variant="primary" onClick={() => setAttachOpen(true)}>
                Add match video
              </Button>
              {hl?.condensed && (
                <ButtonLink href={`/matches/${id}/highlights?video=condensed`}>Watch condensed match</ButtonLink>
              )}
            </div>
          }
        >
          Players, the ball, court lines, hits and bounces are drawn over the original video, with every rally
          one click away.
        </EmptyState>
        <AttachVideoDialog open={attachOpen} onClose={() => setAttachOpen(false)} />
      </>
    );
  return <Viewer />;
}

function Viewer() {
  const { id, data: run, ready } = useMatch();
  const { data: rallies } = useRallies(id, ready);
  const { data: events } = useEvents(id, ready);
  const params = useSearchParams();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [time, setTime] = useState(0);
  const [layers, setLayers] = useState<Record<Layer, boolean>>({ players: true, ball: true, court: false, events: true });
  const windows = useRef(new Map<number, OverlayIndex | "loading">());
  const [, forceDraw] = useState(0);

  const fps = run!.video!.fps;
  const W = run!.video!.width;
  const H = run!.video!.height;
  const matchS = run!.video!.duration_s;
  const court = run?.court;

  const rallyList = useMemo(() => rallies?.rallies ?? [], [rallies]);
  const currentRally = rallyList.find((r) => time >= r.start_s - 0.5 && time <= r.end_s + 0.5) ?? null;

  const seek = useCallback((t: number, play = false) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, Math.min(t, matchS));
    if (play) void v.play();
  }, [matchS]);

  // ?t= from links (e.g. "Watch rally")
  useEffect(() => {
    const t = Number(params.get("t"));
    if (!t) return;
    const v = videoRef.current;
    if (!v) return;
    const go = () => (v.currentTime = t);
    if (v.readyState >= 1) go();
    else v.addEventListener("loadedmetadata", go, { once: true });
  }, [params]);

  const ensureWindow = useCallback(
    (frame: number) => {
      const w = Math.floor(frame / (WINDOW_S * fps));
      for (const k of [w, w + 1]) {
        if (windows.current.has(k)) continue;
        windows.current.set(k, "loading");
        const start = Math.round(k * WINDOW_S * fps);
        fetchOverlay(id, start, Math.round((k + 1) * WINDOW_S * fps) - 1)
          .then((data) => {
            windows.current.set(k, indexOverlay(data));
            forceDraw((n) => n + 1);
          })
          .catch(() => windows.current.delete(k));
      }
      const got = windows.current.get(w);
      return got && got !== "loading" ? got : null;
    },
    [id, fps],
  );

  const draw = useCallback(
    (mediaTime: number) => {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (!canvas || !ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const cw = canvas.clientWidth;
      const ch = canvas.clientHeight;
      if (canvas.width !== Math.round(cw * dpr)) {
        canvas.width = Math.round(cw * dpr);
        canvas.height = Math.round(ch * dpr);
      }
      ctx.setTransform((dpr * cw) / W, 0, 0, (dpr * ch) / H, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const frame = Math.round(mediaTime * fps);
      const ov = ensureWindow(frame);
      const k = W / 1280; // line widths scale with the analysis resolution

      if (layers.court && court) drawCourt(ctx, court, k);
      if (layers.players && ov) {
        for (const b of ov.boxes.get(frame) ?? []) {
          const color = TEAM_COLOR[PLAYER_TEAM[b.player] ?? "near"];
          ctx.strokeStyle = color;
          ctx.lineWidth = 2 * k;
          ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
          const label = `P${b.player}`;
          ctx.font = `600 ${13 * k}px ui-sans-serif, system-ui, sans-serif`;
          const tw = ctx.measureText(label).width + 8 * k;
          ctx.fillStyle = color;
          ctx.fillRect(b.x1, b.y1 - 18 * k, tw, 18 * k);
          ctx.fillStyle = "#fff";
          ctx.fillText(label, b.x1 + 4 * k, b.y1 - 5 * k);
        }
      }
      if (layers.ball && ov) {
        const trail: [number, number][] = [];
        for (let f = frame - BALL_TRAIL; f <= frame; f++) {
          const p = ov.ball.get(f);
          if (p) trail.push([p.u, p.v]);
        }
        if (trail.length > 1) {
          ctx.strokeStyle = "rgba(250, 204, 21, 0.75)";
          ctx.lineWidth = 2 * k;
          ctx.beginPath();
          trail.forEach(([u, v], i) => (i ? ctx.lineTo(u, v) : ctx.moveTo(u, v)));
          ctx.stroke();
        }
        const p = ov.ball.get(frame);
        if (p) {
          ctx.beginPath();
          ctx.arc(p.u, p.v, 6 * k, 0, Math.PI * 2);
          ctx.fillStyle = p.detected ? "#facc15" : "rgba(250, 204, 21, 0.35)";
          ctx.fill();
          ctx.lineWidth = 1.5 * k;
          ctx.strokeStyle = "#1c1917";
          ctx.stroke();
        }
      }
      if (layers.events && events) {
        for (const e of events) {
          if ((e.kind !== "hit" && e.kind !== "bounce") || mediaTime < e.t || mediaTime > e.t + EVENT_SHOW_S) continue;
          const color = e.kind === "hit" ? "#ef4444" : "#22c55e";
          ctx.strokeStyle = color;
          ctx.lineWidth = 3 * k;
          ctx.beginPath();
          ctx.arc(e.u, e.v, 16 * k, 0, Math.PI * 2);
          ctx.stroke();
          ctx.font = `600 ${14 * k}px ui-sans-serif, system-ui, sans-serif`;
          ctx.fillStyle = color;
          ctx.fillText(e.kind === "hit" ? `HIT P${e.player ?? ""}` : "BOUNCE", e.u + 20 * k, e.v - 10 * k);
        }
      }
    },
    [W, H, fps, ensureWindow, layers, court, events],
  );

  // Draw on every presented video frame (requestVideoFrameCallback), else on animation frames.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    let handle = 0;
    let stopped = false;
    const hasRvfc = "requestVideoFrameCallback" in v;
    const loop = (_now: number, meta?: { mediaTime: number }) => {
      if (stopped) return;
      draw(meta?.mediaTime ?? v.currentTime);
      handle = hasRvfc ? v.requestVideoFrameCallback(loop) : requestAnimationFrame((t) => loop(t));
    };
    loop(0);
    const redraw = () => draw(v.currentTime);
    v.addEventListener("seeked", redraw);
    window.addEventListener("resize", redraw);
    return () => {
      stopped = true;
      if (hasRvfc) v.cancelVideoFrameCallback(handle);
      else cancelAnimationFrame(handle);
      v.removeEventListener("seeked", redraw);
      window.removeEventListener("resize", redraw);
    };
  }, [draw]);

  // Keyboard: Space play/pause, ←/→ 5 s, , . one frame, J/K previous/next rally.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.target instanceof Element && e.target.closest("input, select, textarea, dialog")) return;
      const v = videoRef.current;
      if (!v) return;
      const t = v.currentTime;
      const handled = () => (e.preventDefault(), e.stopPropagation());
      if (e.key === " ") {
        handled();
        if (v.paused) void v.play();
        else v.pause();
      } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        handled();
        seek(t + (e.key === "ArrowLeft" ? -5 : 5));
      } else if (e.key === "," || e.key === ".") {
        handled();
        v.pause();
        seek(t + (e.key === "," ? -1 : 1) / fps);
      } else if (e.key.toLowerCase() === "j" || e.key.toLowerCase() === "k") {
        handled();
        const r = e.key.toLowerCase() === "k"
          ? rallyList.find((x) => x.start_s > t + 0.5)
          : [...rallyList].reverse().find((x) => x.start_s < t - 1.5);
        if (r) seek(r.start_s, true);
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [seek, fps, rallyList]);

  const prev = [...rallyList].reverse().find((x) => x.start_s < time - 1.5);
  const next = rallyList.find((x) => x.start_s > time + 0.5);

  return (
    <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_340px]">
      <div className="min-w-0 space-y-3">
        <div className="relative overflow-hidden rounded-lg border border-zinc-200 bg-black" style={{ aspectRatio: `${W} / ${H}` }}>
          <video
            ref={videoRef}
            src={videoUrl(id)}
            controls
            preload="metadata"
            className="absolute inset-0 size-full"
            onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
            onSeeked={(e) => setTime(e.currentTarget.currentTime)}
          />
          <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 size-full" />
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {LAYERS.map((l) => (
            <button
              key={l}
              type="button"
              aria-pressed={layers[l]}
              onClick={() => setLayers((s) => ({ ...s, [l]: !s[l] }))}
              className={cx(
                "h-7 rounded-md border px-2.5 text-xs font-medium",
                layers[l] ? "border-zinc-900 bg-zinc-900 text-white" : "border-zinc-300 bg-white text-zinc-600 hover:bg-zinc-50",
              )}
            >
              {LAYER_LABEL[l]}
            </button>
          ))}
          <span className="ml-auto hidden items-center gap-1.5 text-xs text-zinc-500 lg:flex">
            <Kbd>Space</Kbd> play <Kbd>←</Kbd>
            <Kbd>→</Kbd> 5 s <Kbd>,</Kbd>
            <Kbd>.</Kbd> frame <Kbd>J</Kbd>
            <Kbd>K</Kbd> rally
          </span>
        </div>
        <div className="rounded-lg border border-zinc-200 bg-white p-4">
          {rallies ? (
            <RallyTimeline
              duration={matchS}
              rallies={rallyList}
              events={events}
              current={time}
              selected={currentRally?.id ?? null}
              onSelect={(r) => seek(r.start_s, true)}
              onSeek={(t) => seek(t)}
              height="h-12"
            />
          ) : (
            <Skeleton className="h-16 w-full" />
          )}
        </div>
      </div>

      <section className="rounded-lg border border-zinc-200 bg-white">
        <header className="flex items-center justify-between border-b border-zinc-100 px-4 py-3">
          <div>
            <h2 className="text-sm font-semibold text-zinc-900">{currentRally ? `Rally ${currentRally.id}` : "Between points"}</h2>
            <p className="tabular text-xs text-zinc-500">{clock(time)} / {clock(matchS)}</p>
          </div>
          <div className="flex gap-1">
            <Button className="px-2" disabled={!prev} onClick={() => prev && seek(prev.start_s, true)} aria-label="Previous rally" title="Previous rally (J)">
              <ChevronLeft className="size-4" />
            </Button>
            <Button className="px-2" disabled={!next} onClick={() => next && seek(next.start_s, true)} aria-label="Next rally" title="Next rally (K)">
              <ChevronRight className="size-4" />
            </Button>
          </div>
        </header>
        {currentRally ? (
          <RallyShots rally={currentRally} shots={(rallies?.shots ?? []).filter((s) => s.rally_id === currentRally.id)} time={time} onSeek={seek} matchId={id} />
        ) : (
          <div className="px-4 py-6 text-sm text-zinc-500">
            {next ? (
              <>
                Next: <button type="button" className="font-medium text-accent hover:underline" onClick={() => seek(next.start_s, true)}>Rally {next.id}</button> at{" "}
                <span className="tabular">{clock(next.start_s)}</span>
              </>
            ) : (
              "No more rallies after this point."
            )}
          </div>
        )}
      </section>
    </div>
  );
}

function RallyShots({ rally, shots, time, onSeek, matchId }: { rally: Rally; shots: Shot[]; time: number; onSeek: (t: number, play?: boolean) => void; matchId: string }) {
  return (
    <div>
      <ol className="py-1">
        {shots.map((s) => {
          const past = time >= s.t - 0.05;
          return (
            <li key={s.index}>
              <button
                type="button"
                onClick={() => onSeek(s.t - 0.6, true)}
                className={cx("flex w-full items-center gap-2 px-4 py-1.5 text-left text-sm hover:bg-zinc-50", !past && "opacity-50")}
              >
                <span className="tabular w-4 text-right text-xs text-zinc-400">{s.index}</span>
                <TeamDot team={s.team} />
                <span className="text-zinc-900">Player {s.player}</span>
                <span className="ml-auto text-xs text-zinc-500">
                  {s.speed_kmh != null ? `~${num(s.speed_kmh)} km/h` : s.no_landing === "probably_volley" ? "volleyed" : s.bounce ? s.bounce.zone : ""}
                </span>
                <span className="tabular w-10 text-right text-xs text-zinc-400">{clock(s.t)}</span>
              </button>
            </li>
          );
        })}
      </ol>
      <div className="border-t border-zinc-100 px-4 py-2.5">
        <Link href={`/matches/${matchId}/rallies?rally=${rally.id}`} className="text-sm font-medium text-accent hover:underline">
          Rally details and shot map
        </Link>
      </div>
    </div>
  );
}

// --- overlay data ----------------------------------------------------------------------------
type Box = { player: number; x1: number; y1: number; x2: number; y2: number };
type OverlayIndex = { boxes: Map<number, Box[]>; ball: Map<number, { u: number; v: number; detected: boolean }> };

function indexOverlay(d: OverlayResponse): OverlayIndex {
  const boxes = new Map<number, Box[]>();
  d.box_frame.forEach((f, i) => {
    const list = boxes.get(f) ?? [];
    list.push({ player: d.box_player[i], x1: d.box_x1[i], y1: d.box_y1[i], x2: d.box_x2[i], y2: d.box_y2[i] });
    boxes.set(f, list);
  });
  const ball = new Map<number, { u: number; v: number; detected: boolean }>();
  d.ball_frame.forEach((f, i) => ball.set(f, { u: d.ball_u[i], v: d.ball_v[i], detected: d.ball_detected[i] }));
  return { boxes, ball };
}

function drawCourt(ctx: CanvasRenderingContext2D, court: CourtInfo, k: number) {
  ctx.strokeStyle = "rgba(56, 189, 248, 0.8)";
  ctx.lineWidth = 1.5 * k;
  for (const line of court.lines_image) {
    ctx.beginPath();
    line.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();
  }
}

export type { MatchEvent };
