"use client";

import Link from "next/link";
import { useParams, usePathname, useRouter } from "next/navigation";
import {
  Suspense,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { FileArchive, Search, Upload, X } from "lucide-react";
import { importRun, useRallies, useRun, useRuns, type RunListItem } from "@/lib/api";
import { clock, duration } from "@/lib/format";
import { Button, Dialog, Field, Kbd, cx, inputClass } from "./ui";

type Shell = { openImport: () => void; openSearch: () => void };
const ShellContext = createContext<Shell>({ openImport: () => {}, openSearch: () => {} });
export const useShell = () => useContext(ShellContext);

export function AppShell({ children }: { children: ReactNode }) {
  const [importOpen, setImportOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const shell = useMemo(
    () => ({ openImport: () => setImportOpen(true), openSearch: () => setSearchOpen(true) }),
    [],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSearchOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <ShellContext.Provider value={shell}>
      <header className="sticky top-0 z-30 border-b border-zinc-200 bg-white/95 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-6 px-6">
          <Link href="/" className="flex items-center gap-2.5 text-zinc-900">
            <Logo />
            <span className="text-[15px] font-semibold tracking-tight">PadelVision</span>
          </Link>
          <nav className="flex items-center gap-1 text-sm">
            <Suspense fallback={<span className="px-2.5 py-1.5 font-medium text-zinc-900">Matches</span>}>
              <HeaderLink href="/">Matches</HeaderLink>
            </Suspense>
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={() => setSearchOpen(true)}
              className="flex h-8 w-60 items-center gap-2 whitespace-nowrap rounded-md border border-zinc-300 bg-white px-2.5 text-sm text-zinc-500 hover:border-zinc-400"
            >
              <Search className="size-4 shrink-0" />
              <span>Search</span>
              <span className="ml-auto flex gap-0.5">
                <Kbd>Ctrl</Kbd>
                <Kbd>K</Kbd>
              </span>
            </button>
            <Button variant="primary" onClick={() => setImportOpen(true)}>
              <Upload className="size-4" />
              Import match
            </Button>
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-[1400px] flex-1 px-6 py-6">{children}</main>
      <ImportDialog open={importOpen} onClose={() => setImportOpen(false)} />
      {searchOpen && (
        <Suspense>
          <CommandPalette onClose={() => setSearchOpen(false)} />
        </Suspense>
      )}
    </ShellContext.Provider>
  );
}

function HeaderLink({ href, children }: { href: string; children: ReactNode }) {
  const pathname = usePathname();
  const active = href === "/" ? pathname === "/" || pathname.startsWith("/matches") : pathname.startsWith(href);
  return (
    <Link
      href={href}
      className={cx(
        "rounded-md px-2.5 py-1.5 font-medium",
        active ? "text-zinc-900" : "text-zinc-500 hover:text-zinc-900",
      )}
    >
      {children}
    </Link>
  );
}

export function Logo({ className = "size-6" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden>
      <rect x="1" y="1" width="22" height="22" rx="5" fill="var(--color-court)" />
      <rect x="6" y="4" width="12" height="16" rx="0.5" fill="none" stroke="#fff" strokeWidth="1.4" />
      <line x1="6" y1="12" x2="18" y2="12" stroke="#fff" strokeWidth="1.4" />
      <line x1="12" y1="7.5" x2="12" y2="16.5" stroke="#fff" strokeWidth="1.1" />
      <line x1="6" y1="7.5" x2="18" y2="7.5" stroke="#fff" strokeWidth="1.1" />
      <line x1="6" y1="16.5" x2="18" y2="16.5" stroke="#fff" strokeWidth="1.1" />
    </svg>
  );
}

// --- import ----------------------------------------------------------------------------------
function ImportDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const router = useRouter();
  const [files, setFiles] = useState<File[]>([]);
  const [name, setName] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const reset = () => {
    setFiles([]);
    setName("");
    setProgress(null);
    setError(null);
  };
  const close = () => {
    if (progress !== null) return; // uploading
    reset();
    onClose();
  };
  const add = (list: FileList | null) => {
    if (!list) return;
    const zips = Array.from(list).filter((f) => f.name.toLowerCase().endsWith(".zip"));
    setFiles((prev) => [...prev, ...zips.filter((z) => !prev.some((p) => p.name === z.name))]);
    setError(zips.length ? null : "Choose the .zip files downloaded from the analysis notebook.");
  };
  const submit = async () => {
    setError(null);
    setProgress(0);
    try {
      const run = await importRun(files, name, setProgress);
      reset();
      onClose();
      router.push(`/matches/${run.id}`);
    } catch (e) {
      setError((e as Error).message);
      setProgress(null);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={close}
      title="Import a match"
      footer={
        <>
          <Button onClick={close} disabled={progress !== null}>
            Cancel
          </Button>
          <Button variant="primary" onClick={submit} disabled={!files.length || progress !== null}>
            {progress === null ? "Import" : progress < 1 ? `Uploading ${Math.round(progress * 100)}%` : "Processing…"}
          </Button>
        </>
      }
    >
      <p className="text-sm text-zinc-600">
        Add the result files from the analysis notebook: <span className="font-medium text-zinc-800">run_cache.zip</span>{" "}
        and, for highlight videos, <span className="font-medium text-zinc-800">results.zip</span>.
      </p>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          add(e.dataTransfer.files);
        }}
        onClick={() => input.current?.click()}
        className={cx(
          "mt-4 flex cursor-pointer flex-col items-center rounded-md border border-dashed px-4 py-8 text-center",
          dragging ? "border-accent bg-accent-soft" : "border-zinc-300 hover:bg-zinc-50",
        )}
      >
        <FileArchive className="size-6 text-zinc-400" />
        <div className="mt-2 text-sm font-medium text-zinc-800">Drop zip files here, or click to choose</div>
        <div className="mt-0.5 text-xs text-zinc-500">The match video itself is added later, on the match page.</div>
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
                onClick={() => setFiles(files.filter((x) => x !== f))}
                aria-label={`Remove ${f.name}`}
                disabled={progress !== null}
              >
                <X className="size-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="mt-4">
        <Field label="Match name" hint="Optional. Defaults to the video's file name.">
          <input className={inputClass} value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Club league, round 3" />
        </Field>
      </div>
      {error && <p className="mt-3 text-sm text-red-700">{error}</p>}
    </Dialog>
  );
}

// --- command palette -------------------------------------------------------------------------
type Item = { id: string; group: string; label: string; hint?: string; href: string; keywords?: string };

const SECTIONS = [
  ["", "Overview"],
  ["video", "Video"],
  ["players", "Players"],
  ["rallies", "Rallies"],
  ["shots", "Shots"],
  ["highlights", "Highlights"],
] as const;

function CommandPalette({ onClose }: { onClose: () => void }) {
  const router = useRouter();
  const params = useParams<{ id?: string }>();
  const runId = params.id ?? "";
  const { data: runs } = useRuns();
  const { data: run } = useRun(runId || null);
  const { data: rallies } = useRallies(runId, Boolean(runId) && run?.status === "ready");
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const listRef = useRef<HTMLUListElement>(null);

  const items = useMemo<Item[]>(() => {
    const out: Item[] = [];
    if (runId && run) {
      for (const [path, label] of SECTIONS)
        out.push({ id: `s-${path}`, group: run.name, label, href: `/matches/${runId}${path ? `/${path}` : ""}` });
      for (const r of rallies?.rallies ?? [])
        out.push({
          id: `r-${r.id}`,
          group: "Rallies",
          label: `Rally ${r.id}`,
          hint: `${clock(r.start_s)} · ${r.hits} hits · ${r.duration_s.toFixed(1)} s`,
          href: `/matches/${runId}/rallies?rally=${r.id}`,
          keywords: `rally ${r.id}`,
        });
    }
    for (const m of runs ?? [])
      out.push({
        id: `m-${m.id}`,
        group: "Matches",
        label: m.name,
        hint: m.duration_s ? duration(m.duration_s) : undefined,
        href: `/matches/${m.id}`,
      });
    return out;
  }, [runId, run, rallies, runs]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items.slice(0, 40);
    return items.filter((i) => `${i.label} ${i.keywords ?? ""} ${i.group}`.toLowerCase().includes(q)).slice(0, 60);
  }, [items, query]);

  const go = useCallback(
    (item: Item | undefined) => {
      if (!item) return;
      onClose();
      router.push(item.href);
    },
    [onClose, router],
  );

  useEffect(() => {
    listRef.current?.querySelector(`[data-index="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [active]);

  let lastGroup = "";
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-zinc-900/30 pt-[12vh]" onClick={onClose}>
      <div
        className="w-full max-w-xl overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 border-b border-zinc-100 px-4">
          <Search className="size-4 text-zinc-400" />
          <input
            autoFocus
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActive(0);
            }}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") {
                e.preventDefault();
                setActive((a) => Math.min(a + 1, filtered.length - 1));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setActive((a) => Math.max(a - 1, 0));
              } else if (e.key === "Enter") go(filtered[active]);
              else if (e.key === "Escape") onClose();
            }}
            placeholder={runId ? "Jump to a section, rally or match…" : "Search matches…"}
            className="h-12 w-full bg-transparent text-sm outline-none placeholder:text-zinc-400 focus-visible:outline-none"
          />
          <Kbd>Esc</Kbd>
        </div>
        <ul ref={listRef} className="max-h-[50vh] overflow-y-auto py-1.5">
          {filtered.length === 0 && <li className="px-4 py-6 text-center text-sm text-zinc-500">No results</li>}
          {filtered.map((item, i) => {
            const header = item.group !== lastGroup ? item.group : null;
            lastGroup = item.group;
            return (
              <li key={item.id}>
                {header && <div className="px-4 pb-1 pt-2.5 text-xs font-medium text-zinc-400">{header}</div>}
                <button
                  type="button"
                  data-index={i}
                  onMouseMove={() => setActive(i)}
                  onClick={() => go(item)}
                  className={cx(
                    "flex w-full items-center gap-3 px-4 py-2 text-left text-sm",
                    i === active ? "bg-zinc-100 text-zinc-900" : "text-zinc-700",
                  )}
                >
                  <span className="truncate">{item.label}</span>
                  {item.hint && <span className="tabular ml-auto shrink-0 text-xs text-zinc-500">{item.hint}</span>}
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}

export type { RunListItem };
