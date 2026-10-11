"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Download, Film, RefreshCw } from "lucide-react";
import { fileUrl, useHighlights, type HighlightsVideo } from "@/lib/api";
import { clock, duration, num } from "@/lib/format";
import { AddHighlightsDialog, useMatch } from "@/components/match";
import { Button, EmptyState, Skeleton, cx } from "@/components/ui";

const CARD_S = 3; // summary card after every clip (pipeline/src/padelvision/highlights.py)

const VIDEOS = [
  ["highlights", "Highlights", "The best rallies"],
  ["condensed", "Condensed match", "Every rally, no dead time"],
] as const;

export function HighlightsPage() {
  return (
    <Suspense fallback={<Skeleton className="aspect-video w-full" />}>
      <Highlights />
    </Suspense>
  );
}

function Highlights() {
  const { id, data: run, ready } = useMatch();
  const { data } = useHighlights(id, ready);
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const [addOpen, setAddOpen] = useState(false);
  const dialog = <AddHighlightsDialog open={addOpen} onClose={() => setAddOpen(false)} />;
  if (!run) return null;
  if (!data) return <Skeleton className="aspect-video w-full" />;

  const available = VIDEOS.filter(([key]) => data[key]);
  if (!available.length)
    return (
      <>
        <EmptyState
          icon={<Film className="size-8" />}
          title="No highlight videos for this match"
          action={
            <Button variant="primary" onClick={() => setAddOpen(true)}>
              Add highlight videos
            </Button>
          }
        >
          Highlight videos are rendered by the analysis notebooks: <b>highlights.zip</b> (highlights notebook) or{" "}
          <b>results.zip</b> (full-match notebook).
        </EmptyState>
        {dialog}
      </>
    );

  const key = (params.get("video") as "highlights" | "condensed" | null) ?? available[0][0];
  const current = data[key] ?? data[available[0][0]]!;
  const clip = Number(params.get("clip") ?? -1);

  return (
    <div>
      <div className="mb-4 flex items-center justify-between gap-3">
      <div className="inline-flex rounded-md border border-zinc-200 bg-white p-0.5">
        {available.map(([k, label, sub]) => (
          <button
            key={k}
            type="button"
            onClick={() => router.replace(`${pathname}?video=${k}`, { scroll: false })}
            className={cx(
              "rounded px-3 py-1.5 text-left text-sm",
              k === key ? "bg-zinc-900 text-white" : "text-zinc-600 hover:bg-zinc-50",
            )}
          >
            <span className="font-medium">{label}</span>
            <span className={cx("ml-2 text-xs", k === key ? "text-zinc-300" : "text-zinc-400")}>{sub}</span>
          </button>
        ))}
      </div>
        <Button onClick={() => setAddOpen(true)}>
          <RefreshCw className="size-4" /> Update videos
        </Button>
      </div>
      <Player key={`${key}-${current.version}`} runId={id} video={current} initialClip={clip} />
      {dialog}
    </div>
  );
}

function Player({ runId, video, initialClip }: { runId: string; video: HighlightsVideo; initialClip: number }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [time, setTime] = useState(0);
  const starts = useMemo(
    () =>
      video.clips.reduce<number[]>(
        (acc, c, i) => [...acc, i ? acc[i - 1] + (video.clips[i - 1].end_s - video.clips[i - 1].start_s) + CARD_S : 0],
        [],
      ),
    [video],
  );
  const activeClip = starts.findLastIndex((s) => time >= s - 0.05);

  const seekClip = (i: number) => {
    const v = ref.current;
    if (!v) return;
    v.currentTime = starts[i] + 0.01;
    void v.play();
  };

  useEffect(() => {
    if (initialClip >= 0 && initialClip < starts.length) {
      const v = ref.current;
      if (!v) return;
      const go = () => (v.currentTime = starts[initialClip] + 0.01);
      if (v.readyState >= 1) go();
      else v.addEventListener("loadedmetadata", go, { once: true });
    }
  }, [initialClip, starts]);

  const src = `${fileUrl(runId, video.file)}?v=${video.version}`;
  return (
    <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_340px]">
      <div className="overflow-hidden rounded-lg border border-zinc-200 bg-black">
        <video
          ref={ref}
          src={src}
          controls
          preload="metadata"
          className="aspect-video w-full"
          onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
        />
      </div>
      <section className="rounded-lg border border-zinc-200 bg-white">
        <header className="flex items-center justify-between border-b border-zinc-100 px-4 py-3">
          <div>
            <h2 className="text-sm font-semibold text-zinc-900">{video.clips.length} clips</h2>
            <p className="text-xs text-zinc-500">{duration(video.total_s)} in total</p>
          </div>
          <a
            href={src}
            download
            className="inline-flex items-center gap-1.5 rounded-md border border-zinc-300 px-2.5 py-1.5 text-sm font-medium text-zinc-700 hover:bg-zinc-50"
          >
            <Download className="size-4" /> Download
          </a>
        </header>
        <ol className="max-h-[60vh] overflow-y-auto py-1">
          {video.clips.map((c, i) => (
            <li key={i}>
              <button
                type="button"
                onClick={() => seekClip(i)}
                className={cx(
                  "flex w-full items-center gap-3 px-4 py-2 text-left text-sm",
                  i === activeClip ? "bg-accent-soft" : "hover:bg-zinc-50",
                )}
              >
                <span className="tabular w-10 text-xs text-zinc-500">{clock(starts[i])}</span>
                <span className="font-medium text-zinc-900">Rally {c.rally_id}</span>
                <span className="tabular ml-auto text-xs text-zinc-500">
                  {c.hits} shots · {num(c.duration_s, 1)} s
                  {c.max_shot_kmh != null && ` · ~${num(c.max_shot_kmh)} km/h`}
                </span>
              </button>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
