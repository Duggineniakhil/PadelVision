"use client";

import { useEffect, useState } from "react";
import { Download, History, ImagePlus, Save } from "lucide-react";
import {
  restoreRunCourt,
  runCourtBackground,
  saveRunCourt,
  useRunCourt,
  type CalibrationImage,
  type CalibrationInput,
  type RunCourt,
} from "@/lib/api";
import { num } from "@/lib/format";
import { useMatch } from "@/components/match";
import { AutoDetect, CalibrationEditor, CourtImageDrop, downloadJson, type EditorState } from "@/components/calibration";
import { Button, Callout, Dialog, Skeleton, Spinner } from "@/components/ui";

export function CourtPage() {
  const { id, data: run, ready } = useMatch();
  const { data: court, error } = useRunCourt(id, ready);
  if (!run) return null;
  if (run.status === "processing")
    return (
      <Callout level="info" title="Updating the analysis">
        {run.status_message ?? "Processing"}… The court calibration can be edited again when it&apos;s done.
      </Callout>
    );
  if (error)
    return (
      <Callout level="warning" title="Couldn't load the calibration">
        {error.message}
      </Callout>
    );
  if (!court) return <Skeleton className="aspect-video w-full" />;
  return <CourtEditor court={court} hasVideo={run.has_video} />;
}

function CourtEditor({ court, hasVideo }: { court: RunCourt; hasVideo: boolean }) {
  const { id } = useMatch();
  const [image, setImage] = useState<CalibrationImage | null>(null);
  const [loading, setLoading] = useState(hasVideo);
  const [bgError, setBgError] = useState<string | null>(null);
  const [version, setVersion] = useState(0); // bumped to discard edits
  const [detected, setDetected] = useState<CalibrationInput | null>(null);
  const [w, h] = court.video_size;

  useEffect(() => {
    if (!hasVideo) return;
    let live = true;
    runCourtBackground(id)
      .then((img) => live && setImage(img))
      .catch((e: Error) => live && setBgError(e.message))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [id, hasVideo]);

  const check = (img: CalibrationImage) =>
    img.width === w && img.height === h
      ? null
      : `This image is ${img.width}×${img.height}, but the match was analysed at ${w}×${h}. Use a frame at that resolution (the full-match notebook's calibration.zip has one).`;

  return (
    <div className="space-y-4">
      {!court.can_recompute && (
        <Callout level="warning" title="This match can't be recomputed with a new calibration">
          It was imported without the raw detections. Import the notebook&apos;s run_cache.zip to save here; you can
          still adjust the calibration and download it.
        </Callout>
      )}
      {!image ? (
        loading ? (
          <div className="flex items-center gap-3 rounded-lg border border-zinc-200 bg-white px-4 py-10 text-sm text-zinc-600">
            <Spinner />
            Building a clean image of the court from the video (players removed). The first time takes a few
            seconds.
          </div>
        ) : (
          <div className="space-y-3">
            {bgError && (
              <Callout level="warning" title="Couldn't read the match video">
                {bgError}
              </Callout>
            )}
            <CourtImageDrop
              onImage={setImage}
              check={check}
              hint={
                <>
                  {hasVideo ? "Or use" : "Add the match video to use it directly, or use"} an image at {w}×{h}: a
                  frame of the video, or the full-match notebook&apos;s calibration.zip.
                </>
              }
            />
          </div>
        )
      ) : (
        <>
        <AutoDetect
          image={image}
          onResult={(cal) => {
            setDetected(cal);
            setVersion((v) => v + 1);
          }}
        />
        <CalibrationEditor key={`${image.id}-${version}`} image={image} initial={detected ?? court.calibration ?? null}>
          {(state) => (
            <Actions
              state={state}
              court={court}
              onDiscard={() => {
                setDetected(null);
                setVersion((v) => v + 1);
              }}
              onOtherImage={() => setImage(null)}
            />
          )}
        </CalibrationEditor>
        </>
      )}
    </div>
  );
}

function cm(m: number | null | undefined) {
  return m == null ? "n/a" : `${num(m * 100, 0)} cm`;
}

function Actions({
  state,
  court,
  onDiscard,
  onOtherImage,
}: {
  state: EditorState;
  court: RunCourt;
  onDiscard: () => void;
  onOtherImage: () => void;
}) {
  const { id } = useMatch();
  const [confirm, setConfirm] = useState<"save" | "restore" | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const saved = court.accuracy;
  const { body, fit, fitError, dirty } = state;
  const canSave = court.can_recompute && dirty && !!body && !!fit && !fitError;

  const apply = async (what: "save" | "restore") => {
    setBusy(true);
    setError(null);
    try {
      if (what === "save" && body) await saveRunCourt(id, body);
      else if (what === "restore") await restoreRunCourt(id);
      setConfirm(null);
    } catch (e) {
      setError((e as Error).message);
    }
    setBusy(false);
  };

  return (
    <section className="rounded-lg border border-zinc-200 bg-white px-4 py-3">
      <h2 className="text-sm font-semibold text-zinc-900">This match&apos;s calibration</h2>
      <p className="mt-0.5 text-xs text-zinc-500">
        {saved
          ? `In use: fit ${num(saved.rms_px, 1)} px, near half ${cm(saved.zones.near_half?.expected_error_m)}, far half ${cm(saved.zones.far_half?.expected_error_m)}.`
          : "No calibration saved yet."}{" "}
        {dirty ? "You have unsaved changes." : "No changes yet."}
      </p>
      <div className="mt-3 flex flex-wrap gap-2">
        <Button variant="primary" disabled={!canSave} onClick={() => setConfirm("save")}>
          <Save className="size-4" /> Save and re-run analysis
        </Button>
        {dirty && <Button onClick={onDiscard}>Discard changes</Button>}
        <Button
          variant="ghost"
          disabled={!fit || !!fitError}
          onClick={() => fit && downloadJson(fit.court_json, "court.json")}
        >
          <Download className="size-4" /> court.json
        </Button>
        <Button variant="ghost" onClick={onOtherImage}>
          <ImagePlus className="size-4" /> Other image
        </Button>
        {court.has_previous && !dirty && (
          <Button variant="ghost" onClick={() => setConfirm("restore")}>
            <History className="size-4" /> Restore previous
          </Button>
        )}
      </div>
      <Dialog
        open={confirm !== null}
        onClose={() => !busy && setConfirm(null)}
        title={confirm === "restore" ? "Restore the previous calibration?" : "Save the new calibration?"}
        footer={
          <>
            <Button onClick={() => setConfirm(null)} disabled={busy}>
              Cancel
            </Button>
            <Button variant="primary" onClick={() => confirm && apply(confirm)} disabled={busy}>
              {busy ? "Saving…" : confirm === "restore" ? "Restore and re-run" : "Save and re-run"}
            </Button>
          </>
        }
      >
        <p className="text-sm text-zinc-600">
          Player court positions, distances, speeds, heatmaps, ball tracking, hits, bounces, rallies and shot speeds
          are recomputed. This takes a minute or two. The calibration it replaces is kept, so you can switch back.
        </p>
        <p className="mt-2 text-sm text-zinc-600">
          Highlight videos and images were rendered by the notebook and stay as they are until you update them.
        </p>
        {error && <p className="mt-3 text-sm text-red-700">{error}</p>}
      </Dialog>
    </section>
  );
}
