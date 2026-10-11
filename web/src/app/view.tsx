"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { ChevronRight, Film, Search, Upload, Video } from "lucide-react";
import { useRuns, type RunListItem } from "@/lib/api";
import { date, duration } from "@/lib/format";
import { useShell } from "@/components/shell";
import { Badge, Button, EmptyState, Skeleton, Spinner, inputClass } from "@/components/ui";

export function MatchesPage() {
  const { data: runs, error, isLoading } = useRuns();
  const { openImport } = useShell();
  const [query, setQuery] = useState("");
  const filtered = useMemo(
    () => (runs ?? []).filter((r) => r.name.toLowerCase().includes(query.trim().toLowerCase())),
    [runs, query],
  );

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-zinc-900">Matches</h1>
          <p className="mt-1 text-sm text-zinc-500">Analysed matches on this computer.</p>
        </div>
        {runs && runs.length > 0 && (
          <div className="relative w-72">
            <Search className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-zinc-400" />
            <input
              className={`${inputClass} pl-8`}
              placeholder="Filter by name"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
        )}
      </div>

      <div className="mt-6">
        {error ? (
          <EmptyState title="The analysis service isn't running">
            {error.message}
          </EmptyState>
        ) : isLoading ? (
          <div className="space-y-2">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-14 w-full" />
            ))}
          </div>
        ) : runs && runs.length === 0 ? (
          <EmptyState
            icon={<Film className="size-8" />}
            title="No matches yet"
            action={
              <Button variant="primary" onClick={openImport}>
                <Upload className="size-4" />
                Import match
              </Button>
            }
          >
            Analyse a match in the full-match notebook, download its <b>run_cache.zip</b> and{" "}
            <b>results.zip</b>, and import them here.
          </EmptyState>
        ) : (
          <div className="overflow-hidden rounded-lg border border-zinc-200 bg-white">
            <table className="w-full text-sm">
              <thead className="border-b border-zinc-200 bg-zinc-50 text-left text-xs font-medium text-zinc-500">
                <tr>
                  <th className="px-4 py-2.5 font-medium">Match</th>
                  <th className="px-4 py-2.5 font-medium">Imported</th>
                  <th className="px-4 py-2.5 text-right font-medium">Length</th>
                  <th className="px-4 py-2.5 text-right font-medium">Rallies</th>
                  <th className="px-4 py-2.5 text-right font-medium">Shots</th>
                  <th className="px-4 py-2.5 font-medium">Media</th>
                  <th className="w-8" />
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-100">
                {filtered.map((r) => (
                  <MatchRow key={r.id} run={r} />
                ))}
                {filtered.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-4 py-8 text-center text-zinc-500">
                      No match matches “{query}”.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function MatchRow({ run }: { run: RunListItem }) {
  const href = `/matches/${run.id}`;
  return (
    <tr className="group relative hover:bg-zinc-50">
      <td className="px-4 py-3">
        <Link href={href} className="font-medium text-zinc-900 after:absolute after:inset-0">
          {run.name}
        </Link>
        <div className="mt-0.5">
          <Status run={run} />
        </div>
      </td>
      <td className="px-4 py-3 text-zinc-600">{date(run.created)}</td>
      <td className="tabular px-4 py-3 text-right text-zinc-700">{duration(run.duration_s)}</td>
      <td className="tabular px-4 py-3 text-right text-zinc-700">{run.rallies ?? "–"}</td>
      <td className="tabular px-4 py-3 text-right text-zinc-700">{run.hits ?? "–"}</td>
      <td className="px-4 py-3">
        <div className="flex gap-1.5">
          {run.has_video && (
            <Badge>
              <Video className="size-3" /> Video
            </Badge>
          )}
          {run.has_highlights && (
            <Badge>
              <Film className="size-3" /> Highlights
            </Badge>
          )}
          {!run.has_video && !run.has_highlights && <span className="text-zinc-400">–</span>}
        </div>
      </td>
      <td className="pr-3 text-zinc-300 group-hover:text-zinc-500">
        <ChevronRight className="size-4" />
      </td>
    </tr>
  );
}

function Status({ run }: { run: RunListItem }) {
  if (run.status === "processing")
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-zinc-500">
        <Spinner className="size-3" /> {run.status_message ?? "Processing"}
      </span>
    );
  if (run.status === "error") return <span className="text-xs text-red-700">Processing failed</span>;
  return null;
}
