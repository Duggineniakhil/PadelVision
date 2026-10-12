"use client";

import { useEffect, useRef, useState } from "react";
import { Check, Copy, Download, FileJson, RotateCcw } from "lucide-react";
import type { CalibrationImage, CalibrationInput } from "@/lib/api";
import { AutoDetect, CalibrationEditor, CourtImageDrop, downloadJson, parseCourtJson } from "@/components/calibration";
import { Button, Callout } from "@/components/ui";

export function CalibratePage() {
  const [image, setImage] = useState<CalibrationImage | null>(null);
  const [initial, setInitial] = useState<CalibrationInput | null>(null);
  const [rev, setRev] = useState(0); // bumped to load a new starting calibration into the editor
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const jsonInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    document.title = "Calibrate a court · PadelVision";
  }, []);

  const loadJson = async (file: File | undefined) => {
    if (!file) return;
    try {
      const cal = parseCourtJson(await file.text());
      if (image && (cal.image_size[0] !== image.width || cal.image_size[1] !== image.height))
        throw new Error(
          `That calibration was made on a ${cal.image_size[0]}×${cal.image_size[1]} image; this one is ${image.width}×${image.height}.`,
        );
      setInitial(cal);
      setRev((r) => r + 1);
      setJsonError(null);
    } catch (e) {
      setJsonError((e as Error).message);
    }
  };

  return (
    <div>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-zinc-900">Calibrate a court</h1>
          <p className="mt-1 max-w-2xl text-sm text-zinc-500">
            Maps the video image to court positions in metres. Needed once per camera placement, before a new video
            is analysed. To fix the calibration of a match you have already imported, use its Court tab.
          </p>
        </div>
        {image && (
          <div className="flex gap-2">
            <Button onClick={() => jsonInput.current?.click()}>
              <FileJson className="size-4" /> Open court.json
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                setImage(null);
                setInitial(null);
                setJsonError(null);
              }}
            >
              <RotateCcw className="size-4" /> Start over
            </Button>
          </div>
        )}
        <input
          ref={jsonInput}
          type="file"
          accept=".json,application/json"
          hidden
          onChange={(e) => {
            void loadJson(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>

      {jsonError && (
        <div className="mt-4">
          <Callout level="warning" title="Couldn't open that calibration">
            {jsonError}
          </Callout>
        </div>
      )}

      {!image ? (
        <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
          <CourtImageDrop onImage={setImage} />
          <ol className="space-y-3 text-sm text-zinc-600">
            <li>
              <span className="font-medium text-zinc-900">1. Get an image of the court.</span> Run the full-match
              notebook: without a calibration it stops and saves <b>calibration.zip</b> (a frame and a background with
              the players removed) at the resolution it analyses.
            </li>
            <li>
              <span className="font-medium text-zinc-900">2. Place the court lines</span> you can see; clicks snap onto
              the painted lines and wall edges.
            </li>
            <li>
              <span className="font-medium text-zinc-900">3. Copy the result</span> into the notebook&apos;s{" "}
              <code className="rounded bg-zinc-100 px-1 text-xs">COURT_JSON_TEXT</code> and run it again, or save it as{" "}
              <code className="rounded bg-zinc-100 px-1 text-xs">ml/calibrations/&lt;video name&gt;.json</code>.
            </li>
          </ol>
        </div>
      ) : (
        <div className="mt-6">
          <div className="mb-4">
            <AutoDetect
              image={image}
              onResult={(cal) => {
                setInitial(cal);
                setRev((r) => r + 1);
              }}
            />
          </div>
          <CalibrationEditor key={`${image.id}-${rev}`} image={image} initial={initial}>
            {({ fit, fitError }) => (
              <section className="rounded-lg border border-zinc-200 bg-white px-4 py-3">
                <h2 className="text-sm font-semibold text-zinc-900">Use it</h2>
                <p className="mt-0.5 text-xs text-zinc-500">
                  Paste into <code>COURT_JSON_TEXT</code> in the full-match notebook, or save as{" "}
                  <code>ml/calibrations/&lt;video name&gt;.json</code>.
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <Button
                    variant="primary"
                    disabled={!fit || !!fitError}
                    onClick={() => {
                      if (!fit) return;
                      void navigator.clipboard.writeText(JSON.stringify(fit.court_json, null, 2)).then(() => {
                        setCopied(true);
                        setTimeout(() => setCopied(false), 1500);
                      });
                    }}
                  >
                    {copied ? <Check className="size-4" /> : <Copy className="size-4" />}
                    {copied ? "Copied" : "Copy JSON"}
                  </Button>
                  <Button disabled={!fit || !!fitError} onClick={() => fit && downloadJson(fit.court_json, "court.json")}>
                    <Download className="size-4" /> Download court.json
                  </Button>
                </div>
              </section>
            )}
          </CalibrationEditor>
        </div>
      )}
    </div>
  );
}
