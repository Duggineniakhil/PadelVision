"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Play } from "lucide-react";
import { useHighlights, useRallies, type Rally, type Shot } from "@/lib/api";
import { clock, num, ZONE_LABEL, shotTypeLabel } from "@/lib/format";
import { PLAYER_TEAM } from "@/lib/court";
import { CourtSvg, ShotLayer } from "@/components/court";
import { useMatch } from "@/components/match";
import { ButtonLink, EmptyState, Kbd, Skeleton, TeamDot, cx } from "@/components/ui";

type SortKey = "id" | "duration_s" | "hits" | "max_shot_kmh";

export function RalliesPage() {
  return (
    <Suspense fallback={<Skeleton className="h-96 w-full" />}>
      <Rallies />
    </Suspense>
  );
}

function Rallies() {
  const { id, data: run, ready } = useMatch();
  const { data } = useRallies(id, ready);
  const { data: hl } = useHighlights(id, ready);
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "id", desc: false });
  const [minShots, setMinShots] = useState(0);
  const [player, setPlayer] = useState(0);

  const rows = useMemo(() => {
    const list = (data?.rallies ?? []).filter(
      (r) => r.hits >= minShots && (!player || (r.hits_by_player[String(player)] ?? 0) > 0),
    );
    const v = (r: Rally) => r[sort.key] ?? -1;
    return [...list].sort((a, b) => (sort.desc ? v(b) - v(a) : v(a) - v(b)));
  }, [data, sort, minShots, player]);

  const selectedId = Number(params.get("rally")) || rows[0]?.id || null;
  const selected = data?.rallies.find((r) => r.id === selectedId) ?? null;
  const select = (r: Rally) => router.replace(`${pathname}?rally=${r.id}`, { scroll: false });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.target instanceof Element && e.target.closest("input, select, textarea, dialog")) return;
      if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
      e.preventDefault();
      const i = rows.findIndex((r) => r.id === selectedId);
      const next = rows[Math.min(rows.length - 1, Math.max(0, i + (e.key === "ArrowDown" ? 1 : -1)))];
      if (next) select(next);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  useEffect(() => {
    document.querySelector(`[data-rally="${selectedId}"]`)?.scrollIntoView({ block: "nearest" });
  }, [selectedId]);

  if (!run) return null;
  if (!data) return <Skeleton className="h-96 w-full" />;
  if (data.rallies.length === 0)
    return (
      <EmptyState title="No rallies detected">
        A rally needs an exchange of shots between the two teams. If the ball was rarely found (see Overview),
        play may not have been recognised.
      </EmptyState>
    );

  const condensedClip = hl?.condensed?.clips.findIndex((c) => c.rally_id === selected?.id) ?? -1;
  const watchHref = selected
    ? run.has_video
      ? `/matches/${id}/video?t=${selected.start_s.toFixed(2)}`
      : condensedClip >= 0
        ? `/matches/${id}/highlights?video=condensed&clip=${condensedClip}`
        : null
    : null;

  const header = (key: SortKey, label: string, right = true) => (
    <th className={cx("px-3 py-2 font-medium", right && "text-right")}>
      <button
        type="button"
        className="inline-flex items-center gap-1 hover:text-zinc-900"
        onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : key !== "id" }))}
      >
        {label}
        {sort.key === key && (sort.desc ? <ArrowDown className="size-3" /> : <ArrowUp className="size-3" />)}
      </button>
    </th>
  );

  return (
    <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_480px]">
      <section className="rounded-lg border border-zinc-200 bg-white">
        <div className="flex flex-wrap items-center gap-3 border-b border-zinc-100 px-4 py-3 text-sm">
          <label className="flex items-center gap-2 text-zinc-600">
            Shots
            <select
              className="h-8 rounded-md border border-zinc-300 bg-white px-2 text-sm text-zinc-900"
              value={minShots}
              onChange={(e) => setMinShots(Number(e.target.value))}
            >
              <option value={0}>Any</option>
              <option value={2}>2 or more</option>
              <option value={3}>3 or more</option>
              <option value={5}>5 or more</option>
            </select>
          </label>
          <label className="flex items-center gap-2 text-zinc-600">
            Player
            <select
              className="h-8 rounded-md border border-zinc-300 bg-white px-2 text-sm text-zinc-900"
              value={player}
              onChange={(e) => setPlayer(Number(e.target.value))}
            >
              <option value={0}>Any</option>
              {[1, 2, 3, 4].map((p) => (
                <option key={p} value={p}>
                  Player {p}
                </option>
              ))}
            </select>
          </label>
          <span className="tabular text-zinc-500">
            {rows.length} of {data.rallies.length}
          </span>
          <span className="ml-auto hidden items-center gap-1 text-xs text-zinc-400 md:flex">
            <Kbd>↑</Kbd>
            <Kbd>↓</Kbd> to move
          </span>
        </div>
        <div className="max-h-[calc(100vh-260px)] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 z-10 bg-white text-left text-xs text-zinc-500 shadow-[0_1px_0_#f4f4f5]">
              <tr>
                {header("id", "Rally", false)}
                <th className="px-3 py-2 font-medium">Start</th>
                {header("duration_s", "Length")}
                {header("hits", "Shots")}
                <th className="px-3 py-2 text-right font-medium">Landed</th>
                <th className="px-3 py-2 text-right font-medium">Volleys</th>
                {header("max_shot_kmh", "Fastest (est.)")}
              </tr>
            </thead>
            <tbody className="tabular">
              {rows.map((r) => (
                <tr
                  key={r.id}
                  data-rally={r.id}
                  onClick={() => select(r)}
                  aria-selected={r.id === selectedId}
                  className={cx(
                    "cursor-pointer border-b border-zinc-50",
                    r.id === selectedId ? "bg-accent-soft" : "hover:bg-zinc-50",
                  )}
                >
                  <td className="px-3 py-2 font-medium text-zinc-900">
                    <span className={cx("mr-2 inline-block h-4 w-0.5 align-middle", r.id === selectedId ? "bg-accent" : "bg-transparent")} />
                    {r.id}
                  </td>
                  <td className="px-3 py-2 text-zinc-600">{clock(r.start_s)}</td>
                  <td className="px-3 py-2 text-right text-zinc-700">{num(r.duration_s, 1)} s</td>
                  <td className="px-3 py-2 text-right text-zinc-700">{r.hits}</td>
                  <td className="px-3 py-2 text-right text-zinc-700">{r.landed}</td>
                  <td className="px-3 py-2 text-right text-zinc-700">{r.probable_volleys}</td>
                  <td className="px-3 py-2 text-right text-zinc-700">{r.max_shot_kmh != null ? `${num(r.max_shot_kmh)} km/h` : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length === 0 && <p className="px-4 py-8 text-center text-sm text-zinc-500">No rally matches these filters.</p>}
        </div>
      </section>

      {selected && (
        <RallyDetail
          rally={selected}
          shots={data.shots.filter((s) => s.rally_id === selected.id)}
          watchHref={watchHref}
          watchLabel={run.has_video ? "Watch rally" : "Watch in condensed match"}
        />
      )}
    </div>
  );
}

function RallyDetail({
  rally,
  shots,
  watchHref,
  watchLabel,
}: {
  rally: Rally;
  shots: Shot[];
  watchHref: string | null;
  watchLabel: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  return (
    <section className="rounded-lg border border-zinc-200 bg-white xl:sticky xl:top-20">
      <header className="flex items-start justify-between gap-3 border-b border-zinc-100 px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold text-zinc-900">Rally {rally.id}</h2>
          <p className="tabular mt-0.5 text-xs text-zinc-500">
            {clock(rally.start_s)} – {clock(rally.end_s)} · {num(rally.duration_s, 1)} s · {rally.hits} shots
          </p>
        </div>
        {watchHref && (
          <ButtonLink href={watchHref} variant="primary">
            <Play className="size-3.5" />
            {watchLabel}
          </ButtonLink>
        )}
      </header>
      <div className="grid grid-cols-[200px_minmax(0,1fr)] gap-4 p-4">
        <div>
          <CourtSvg label={`Shot map of rally ${rally.id}`} className="w-full">
            <ShotLayer shots={shots} highlight={hover} />
          </CourtSvg>
          <div className="mt-2 space-y-1 text-[11px] text-zinc-500">
            <div className="flex items-center gap-1.5">
              <span className="inline-block size-2.5 rounded-full bg-zinc-500" /> Hitter, numbered in order
            </div>
            <div className="flex items-center gap-1.5">
              <span className="inline-block size-2.5 rounded-full border-2 border-zinc-500" /> Probably volleyed back
            </div>
          </div>
        </div>
        <ol className="min-w-0 space-y-1">
          {shots.map((s) => (
            <li
              key={s.index}
              onMouseEnter={() => setHover(s.index)}
              onMouseLeave={() => setHover(null)}
              className="rounded-md px-2 py-1.5 hover:bg-zinc-50"
            >
              <div className="flex items-center gap-2 text-sm">
                <span className="tabular w-4 text-right text-xs font-medium text-zinc-400">{s.index}</span>
                <TeamDot team={PLAYER_TEAM[s.player] ?? s.team} />
                <span className="font-medium text-zinc-900">Player {s.player}</span>
                {shotTypeLabel(s) && (
                  <span className="rounded bg-zinc-100 px-1.5 py-0.5 text-xs text-zinc-700" title={s.type_basis ?? undefined}>
                    {shotTypeLabel(s)}
                  </span>
                )}
                <span className="tabular ml-auto text-xs text-zinc-500">{clock(s.t)}</span>
              </div>
              <div className="ml-6 mt-0.5 text-xs text-zinc-600">{shotOutcome(s)}</div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

function shotOutcome(s: Shot): string {
  if (s.bounce) {
    const where = `Landed in the ${(ZONE_LABEL[s.bounce.zone] ?? s.bounce.zone).toLowerCase()} zone`;
    if (s.speed_kmh != null && s.speed_range_kmh)
      return `${where} · ~${num(s.speed_kmh)} km/h (${s.speed_range_kmh[0]}–${s.speed_range_kmh[1]})`;
    return where;
  }
  if (s.no_landing === "probably_volley") return "Probably volleyed back before bouncing";
  return "Landing not seen";
}
