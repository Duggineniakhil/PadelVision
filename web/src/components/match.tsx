"use client";

import { useParams } from "next/navigation";
import { useRef, useState } from "react";
import { HardDrive, Upload } from "lucide-react";
import { addFiles, attachVideoPath, uploadVideo, useRun } from "@/lib/api";
import { ZipDropzone } from "./dropzone";
import { Button, Dialog, Field, cx, inputClass } from "./ui";

/** The current match id from the URL and its data (shared by all match pages through SWR). */
export function useMatch() {
  const { id } = useParams<{ id: string }>();
  const run = useRun(id);
  return { id, ...run, ready: run.data?.status === "ready" };
}

export function AttachVideoDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { id, data: run } = useMatch();
  const [mode, setMode] = useState<"path" | "upload">("path");
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);

  const done = () => {
    setBusy(false);
    setProgress(null);
    setPath("");
    setFile(null);
    setError(null);
    onClose();
  };
  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      if (mode === "path") await attachVideoPath(id, path);
      else if (file) await uploadVideo(id, file, setProgress);
      done();
    } catch (e) {
      setError((e as Error).message);
      setBusy(false);
      setProgress(null);
    }
  };
  const v = run?.video;

  return (
    <Dialog
      open={open}
      onClose={() => !busy && done()}
      title="Add the match video"
      footer={
        <>
          <Button onClick={done} disabled={busy}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={submit}
            disabled={busy || (mode === "path" ? !path.trim() : !file)}
          >
            {busy ? (progress != null && progress < 1 ? `Uploading ${Math.round(progress * 100)}%` : "Checking…") : "Add video"}
          </Button>
        </>
      }
    >
      <p className="text-sm text-zinc-600">
        Overlays, the rally timeline and shot replays are drawn on the video. Use the video the analysis was
        made from{v ? ` (${v.frame_count.toLocaleString("en-GB")} frames)` : ""}; it is checked before it is
        added.
      </p>
      <div className="mt-4 grid grid-cols-2 gap-2">
        {(
          [
            ["path", HardDrive, "File on this computer", "Played in place, nothing is copied"],
            ["upload", Upload, "Upload a copy", "Stored with the match"],
          ] as const
        ).map(([m, Icon, title, sub]) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className={cx(
              "rounded-md border px-3 py-2.5 text-left",
              mode === m ? "border-accent bg-accent-soft" : "border-zinc-200 hover:bg-zinc-50",
            )}
          >
            <Icon className={cx("size-4", mode === m ? "text-accent" : "text-zinc-500")} />
            <div className="mt-1.5 text-sm font-medium text-zinc-900">{title}</div>
            <div className="text-xs text-zinc-500">{sub}</div>
          </button>
        ))}
      </div>
      <div className="mt-4">
        {mode === "path" ? (
          <Field label="Video file path" hint="Tip: in Explorer, Shift + right-click the file → Copy as path.">
            <input
              className={inputClass}
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="C:\Videos\GX010012_1280.mp4"
            />
          </Field>
        ) : (
          <div className="flex items-center gap-3">
            <Button onClick={() => fileInput.current?.click()}>Choose file</Button>
            <span className="truncate text-sm text-zinc-600">{file?.name ?? "No file chosen"}</span>
            <input
              ref={fileInput}
              type="file"
              accept="video/*"
              hidden
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </div>
        )}
      </div>
      {error && <p className="mt-3 text-sm text-red-700">{error}</p>}
    </Dialog>
  );
}

const FILE_LABEL: Record<string, string> = {
  "highlights.mp4": "Highlights video",
  "highlights.json": "Highlights clip list",
  "condensed.mp4": "Condensed match",
  "condensed.json": "Condensed match clip list",
  "heatmaps.png": "Heatmaps image",
  "placement.png": "Placement image",
  "court_overlay.png": "Court overlay image",
  "appearance.parquet": "Player colours (the player stats are re-run with them)",
};

/** Replace the match's highlight videos (and images) with newer ones from the notebooks. */
export function AddHighlightsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { id } = useMatch();
  const [files, setFiles] = useState<File[]>([]);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updated, setUpdated] = useState<string[] | null>(null);
  const close = () => {
    if (progress !== null) return;
    setFiles([]);
    setError(null);
    setUpdated(null);
    onClose();
  };
  const submit = async () => {
    setError(null);
    setProgress(0);
    try {
      const r = await addFiles(id, files, setProgress);
      setUpdated(r.updated);
      setFiles([]);
    } catch (e) {
      setError((e as Error).message);
    }
    setProgress(null);
  };
  return (
    <Dialog
      open={open}
      onClose={close}
      title="Update highlight videos"
      footer={
        updated ? (
          <Button variant="primary" onClick={close}>
            Done
          </Button>
        ) : (
          <>
            <Button onClick={close} disabled={progress !== null}>
              Cancel
            </Button>
            <Button variant="primary" onClick={submit} disabled={!files.length || progress !== null}>
              {progress === null ? "Update" : progress < 1 ? `Uploading ${Math.round(progress * 100)}%` : "Saving…"}
            </Button>
          </>
        )
      }
    >
      {updated ? (
        <div className="text-sm">
          <p className="font-medium text-zinc-900">Updated</p>
          <ul className="mt-2 space-y-1 text-zinc-600">
            {updated.map((u) => (
              <li key={u}>{FILE_LABEL[u] ?? u}</li>
            ))}
          </ul>
        </div>
      ) : (
        <>
          <p className="text-sm text-zinc-600">
            Add <span className="font-medium text-zinc-800">highlights.zip</span> from the highlights notebook or{" "}
            <span className="font-medium text-zinc-800">results.zip</span> from the full-match notebook. The
            highlight videos and images are replaced; the analysis itself stays as it is.
          </p>
          <div className="mt-4">
            <ZipDropzone
              files={files}
              disabled={progress !== null}
              onChange={(f, err) => {
                setFiles(f);
                setError(err);
              }}
            />
          </div>
          {error && <p className="mt-3 text-sm text-red-700">{error}</p>}
        </>
      )}
    </Dialog>
  );
}
