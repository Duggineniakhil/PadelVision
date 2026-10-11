"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { AlertTriangle, ChevronLeft, Film, MoreHorizontal, Pencil, RotateCw, Trash2, Video } from "lucide-react";
import { deleteRun, renameRun, reprocessRun, useRallies } from "@/lib/api";
import { date, duration } from "@/lib/format";
import { AddHighlightsDialog, AttachVideoDialog, useMatch } from "./match";
import { Button, Dialog, EmptyState, Skeleton, Spinner, cx, inputClass } from "./ui";

const TABS = [
  ["", "Overview"],
  ["video", "Video"],
  ["players", "Players"],
  ["rallies", "Rallies"],
  ["shots", "Shots"],
  ["highlights", "Highlights"],
] as const;

export function MatchLayout({ children }: { children: ReactNode }) {
  const { id, data: run, error, ready } = useMatch();
  const { data: rallies } = useRallies(id, ready);
  const pathname = usePathname();
  const [videoOpen, setVideoOpen] = useState(false);
  const [highlightsOpen, setHighlightsOpen] = useState(false);

  useEffect(() => {
    if (run) document.title = `${run.name} · PadelVision`;
  }, [run]);

  if (error)
    return (
      <EmptyState title={error.status === 404 ? "Match not found" : "Couldn't load this match"}>
        {error.status === 404 ? "It may have been deleted." : error.message}
        <div className="mt-4">
          <Link href="/" className="text-accent hover:underline">
            Back to matches
          </Link>
        </div>
      </EmptyState>
    );
  if (!run)
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-72" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );

  const base = `/matches/${id}`;
  const counts: Record<string, number | undefined> = {
    rallies: rallies?.rallies.length,
    shots: rallies?.shots.length,
  };

  return (
    <div>
      <Link href="/" className="inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-900">
        <ChevronLeft className="size-4" />
        Matches
      </Link>
      <div className="mt-2 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <Title id={id} name={run.name} />
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-zinc-500">
            <span>Imported {date(run.created)}</span>
            {run.video && <span>{duration(run.video.duration_s)}</span>}
            {run.video && (
              <span className="tabular">
                {run.video.width}×{run.video.height} · {run.video.fps.toFixed(2)} fps
              </span>
            )}
            {run.has_video ? (
              <span className="inline-flex items-center gap-1 text-zinc-600">
                <Video className="size-3.5" /> {run.video_name}
              </span>
            ) : null}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {!run.has_video && run.status === "ready" && (
            <Button onClick={() => setVideoOpen(true)}>
              <Video className="size-4" />
              Add match video
            </Button>
          )}
          <MatchMenu
            id={id}
            disabled={run.status === "processing"}
            onRename={() => document.getElementById("match-title")?.click()}
            onAddVideo={() => setVideoOpen(true)}
            onUpdateHighlights={() => setHighlightsOpen(true)}
          />
        </div>
      </div>

      <nav className="mt-5 flex gap-1 border-b border-zinc-200" aria-label="Match sections">
        {TABS.map(([path, label]) => {
          const href = path ? `${base}/${path}` : base;
          const active = path ? pathname.startsWith(href) : pathname === base;
          const count = counts[path];
          return (
            <Link
              key={path}
              href={href}
              aria-current={active ? "page" : undefined}
              className={cx(
                "-mb-px flex items-center gap-1.5 border-b-2 px-3 py-2.5 text-sm font-medium",
                active
                  ? "border-accent text-zinc-900"
                  : "border-transparent text-zinc-500 hover:border-zinc-300 hover:text-zinc-800",
              )}
            >
              {label}
              {count != null && (
                <span className="tabular rounded bg-zinc-100 px-1.5 text-xs font-medium text-zinc-600">{count}</span>
              )}
            </Link>
          );
        })}
      </nav>

      <div className="mt-6">
        {run.status === "processing" ? (
          <div className="flex flex-col items-center rounded-lg border border-zinc-200 bg-white px-6 py-16 text-center">
            <Spinner className="size-5" />
            <div className="mt-3 text-sm font-medium text-zinc-900">{run.status_message ?? "Processing"}</div>
            <div className="mt-1 text-sm text-zinc-500">
              Preparing players, ball tracking and rallies. This usually takes one to three minutes.
            </div>
          </div>
        ) : run.status === "error" ? (
          <div className="rounded-lg border border-red-200 bg-red-50/50 px-4 py-4 text-sm">
            <div className="flex items-center gap-2 font-medium text-red-800">
              <AlertTriangle className="size-4" /> Processing failed
            </div>
            <p className="mt-1 text-red-700">{run.status_message}</p>
            <Button className="mt-3" onClick={() => reprocessRun(id)}>
              <RotateCw className="size-4" /> Try again
            </Button>
          </div>
        ) : (
          children
        )}
      </div>
      <AttachVideoDialog open={videoOpen} onClose={() => setVideoOpen(false)} />
      <AddHighlightsDialog open={highlightsOpen} onClose={() => setHighlightsOpen(false)} />
    </div>
  );
}

function Title({ id, name }: { id: string; name: string }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(name);
  const save = async () => {
    setEditing(false);
    if (value.trim() && value.trim() !== name) await renameRun(id, value.trim());
    else setValue(name);
  };
  if (editing)
    return (
      <input
        autoFocus
        className={cx(inputClass, "h-9 w-96 text-lg font-semibold")}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onBlur={save}
        onKeyDown={(e) => {
          if (e.key === "Enter") void save();
          if (e.key === "Escape") {
            setValue(name);
            setEditing(false);
          }
        }}
      />
    );
  return (
    <button
      id="match-title"
      type="button"
      onClick={() => {
        setValue(name);
        setEditing(true);
      }}
      className="group flex items-center gap-2 text-left"
      title="Rename"
    >
      <h1 className="truncate text-xl font-semibold tracking-tight text-zinc-900">{name}</h1>
      <Pencil className="size-3.5 text-zinc-300 group-hover:text-zinc-500" />
    </button>
  );
}

function MatchMenu({
  id,
  disabled,
  onRename,
  onAddVideo,
  onUpdateHighlights,
}: {
  id: string;
  disabled: boolean;
  onRename: () => void;
  onAddVideo: () => void;
  onUpdateHighlights: () => void;
}) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onDown = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, []);
  const item = "flex w-full items-center gap-2 px-3 py-2 text-left text-sm hover:bg-zinc-50";
  return (
    <div ref={ref} className="relative">
      <Button variant="secondary" className="px-2" aria-label="More actions" onClick={() => setOpen((o) => !o)}>
        <MoreHorizontal className="size-4" />
      </Button>
      {open && (
        <div className="absolute right-0 z-20 mt-1 w-56 overflow-hidden rounded-md border border-zinc-200 bg-white py-1 shadow-lg">
          <button type="button" className={item} onClick={() => (setOpen(false), onRename())}>
            <Pencil className="size-4 text-zinc-500" /> Rename
          </button>
          <button
            type="button"
            className={cx(item, disabled && "pointer-events-none opacity-50")}
            onClick={() => (setOpen(false), onAddVideo())}
          >
            <Video className="size-4 text-zinc-500" /> Add or replace match video
          </button>
          <button
            type="button"
            className={cx(item, disabled && "pointer-events-none opacity-50")}
            onClick={() => (setOpen(false), onUpdateHighlights())}
          >
            <Film className="size-4 text-zinc-500" /> Update highlight videos
          </button>
          <button
            type="button"
            className={cx(item, disabled && "pointer-events-none opacity-50")}
            onClick={() => (setOpen(false), void reprocessRun(id))}
          >
            <RotateCw className="size-4 text-zinc-500" /> Re-run analysis
          </button>
          <div className="my-1 border-t border-zinc-100" />
          <button type="button" className={cx(item, "text-red-700")} onClick={() => (setOpen(false), setConfirmDelete(true))}>
            <Trash2 className="size-4" /> Delete match
          </button>
        </div>
      )}
      <Dialog
        open={confirmDelete}
        onClose={() => setConfirmDelete(false)}
        title="Delete this match?"
        width="max-w-md"
        footer={
          <>
            <Button onClick={() => setConfirmDelete(false)}>Cancel</Button>
            <Button
              variant="danger"
              onClick={async () => {
                await deleteRun(id);
                router.push("/");
              }}
            >
              Delete
            </Button>
          </>
        }
      >
        <p className="text-sm text-zinc-600">
          The analysis results are removed from this computer. A video added from elsewhere on disk is not deleted.
        </p>
      </Dialog>
    </div>
  );
}
