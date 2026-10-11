"use client";

import { useRef, useState, type ReactNode } from "react";
import { FileArchive, X } from "lucide-react";
import { cx } from "./ui";

/** Drop or choose .zip files; shows the chosen files with a remove button. */
export function ZipDropzone({
  files,
  onChange,
  disabled,
  hint,
}: {
  files: File[];
  onChange: (files: File[], error: string | null) => void;
  disabled?: boolean;
  hint?: ReactNode;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const add = (list: FileList | null) => {
    if (!list) return;
    const zips = Array.from(list).filter((f) => f.name.toLowerCase().endsWith(".zip"));
    onChange(
      [...files, ...zips.filter((z) => !files.some((p) => p.name === z.name))],
      zips.length ? null : "Choose the .zip files downloaded from the analysis notebook.",
    );
  };
  return (
    <>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) add(e.dataTransfer.files);
        }}
        onClick={() => !disabled && input.current?.click()}
        className={cx(
          "flex cursor-pointer flex-col items-center rounded-md border border-dashed px-4 py-8 text-center",
          dragging ? "border-accent bg-accent-soft" : "border-zinc-300 hover:bg-zinc-50",
        )}
      >
        <FileArchive className="size-6 text-zinc-400" />
        <div className="mt-2 text-sm font-medium text-zinc-800">Drop zip files here, or click to choose</div>
        {hint && <div className="mt-0.5 text-xs text-zinc-500">{hint}</div>}
        <input
          ref={input}
          type="file"
          accept=".zip"
          multiple
          hidden
          onChange={(e) => {
            add(e.target.files);
            e.target.value = "";
          }}
        />
      </div>
      {files.length > 0 && (
        <ul className="mt-3 divide-y divide-zinc-100 rounded-md border border-zinc-200">
          {files.map((f) => (
            <li key={f.name} className="flex items-center gap-2 px-3 py-2 text-sm">
              <FileArchive className="size-4 text-zinc-400" />
              <span className="truncate text-zinc-800">{f.name}</span>
              <span className="tabular ml-auto text-xs text-zinc-500">{(f.size / 1e6).toFixed(1)} MB</span>
              <button
                type="button"
                className="rounded p-0.5 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700"
                onClick={() => onChange(files.filter((x) => x !== f), null)}
                aria-label={`Remove ${f.name}`}
                disabled={disabled}
              >
                <X className="size-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
