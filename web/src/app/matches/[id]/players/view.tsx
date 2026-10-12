"use client";

import { useTracks, type PlayerStats, type TeamStats } from "@/lib/api";
import { num, pct, teamName, ZONE_LABEL } from "@/lib/format";
import { CourtSvg, HeatmapLayer } from "@/components/court";
import { useMatch } from "@/components/match";
import { Card, Skeleton, TeamDot, cx } from "@/components/ui";

export function PlayersPage() {
  const { id, data: run, ready } = useMatch();
  const { data: tracks } = useTracks(id, ready);
  if (!run) return null;
  const pairs = run.changeovers.length > 0;

  return (
    <div className="space-y-5">
      <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
        {run.players.map((p) => {
          const tr = tracks?.players[String(p.player)];
          return (
            <PlayerCard key={p.player} player={p} pairs={pairs} track={tracks ? tr ?? null : undefined} />
          );
        })}
      </div>
      <div className="grid gap-5 lg:grid-cols-2">
        {run.teams.map((t) => (
          <TeamCard key={t.team} team={t} pairs={pairs} />
        ))}
      </div>
      <p className="text-xs text-zinc-500">
        {pairs
          ? `The teams changed ends ${run.changeovers.length} times. Each player is followed across the changeovers by the side of the court they play (forehand or backhand side); team A (players 1 and 2) started near the camera.`
          : "Players 1–2 are near the camera, 3–4 far."}
      </p>
    </div>
  );
}

function PlayerCard({
  player: p,
  pairs,
  track,
}: {
  player: PlayerStats;
  pairs: boolean;
  track: { x: number[]; y: number[] } | null | undefined;
}) {
  const ok = !p.insufficient_data;
  return (
    <section className="flex flex-col rounded-lg border border-zinc-200 bg-white">
      <header className="flex items-center justify-between border-b border-zinc-100 px-4 py-3">
        <div className="flex items-center gap-2">
          <TeamDot team={p.team} />
          <h2 className="text-sm font-semibold text-zinc-900">Player {p.player}</h2>
          <span className="text-xs text-zinc-500">{teamName(p.team, pairs)}</span>
        </div>
        <span className="tabular text-xs text-zinc-500">tracked {pct(p.tracked_fraction)}</span>
      </header>
      <div className="px-4 pt-4">
        <div className="mx-auto max-w-[220px]">
          {track === undefined ? (
            <Skeleton className="aspect-[12/22] w-full" />
          ) : (
            <CourtSvg label={`Court coverage of player ${p.player}`} className="w-full">
              {ok && track && <HeatmapLayer x={track.x} y={track.y} />}
              {(!ok || !track) && (
                <text x={0} y={0.4} textAnchor="middle" fontSize={0.9} fill="#cbd5e1">
                  Not enough data
                </text>
              )}
            </CourtSvg>
          )}
        </div>
        <p className="mt-1.5 text-center text-[11px] text-zinc-400">Camera side at the bottom</p>
      </div>
      <div className="flex-1 px-4 pb-4 pt-3">
        {ok ? (
          <>
            <dl className="tabular grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
              <Metric label="Distance" value={num(p.distance_m, 0, "m")} />
              <Metric label="Sprints" value={num(p.sprints)} />
              <Metric label="Avg speed" value={num(p.avg_speed_kmh, 1, "km/h")} />
              <Metric label="Peak speed" value={num(p.peak_speed_kmh, 1, "km/h")} />
              <Metric label="Rally shots" value={String(p.rally_hits)} />
              <Metric
                label="Shot speed (est.)"
                value={p.shot_speed_median_kmh != null ? `${num(p.shot_speed_median_kmh)} km/h` : "–"}
                hint={
                  p.shot_speed_samples
                    ? `median of ${p.shot_speed_samples}, max ${num(p.shot_speed_max_kmh)}`
                    : "no measured shots"
                }
              />
            </dl>
            {p.zones && <ZoneBar zones={p.zones} team={p.team} />}
          </>
        ) : (
          <p className="rounded-md bg-zinc-50 px-3 py-2.5 text-sm text-zinc-600">
            No movement stats: this player was {p.insufficient_data}.
          </p>
        )}
      </div>
    </section>
  );
}

function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs text-zinc-500">{label}</dt>
      <dd className="truncate font-medium text-zinc-900">{value}</dd>
      {hint && <dd className="truncate text-[11px] text-zinc-500">{hint}</dd>}
    </div>
  );
}

const ZONE_ORDER = ["net", "transition", "back"];

function ZoneBar({ zones, team }: { zones: Record<string, number>; team: "near" | "far" }) {
  const shades = team === "near" ? ["bg-near", "bg-near/60", "bg-near/30"] : ["bg-far", "bg-far/60", "bg-far/30"];
  return (
    <div className="mt-4">
      <div className="text-xs text-zinc-500">Time by court zone</div>
      <div className="mt-1.5 flex h-2 overflow-hidden rounded-full bg-zinc-100">
        {ZONE_ORDER.map((z, i) => (
          <div key={z} className={shades[i]} style={{ width: `${(zones[z] ?? 0) * 100}%` }} />
        ))}
      </div>
      <div className="tabular mt-1.5 flex justify-between text-[11px] text-zinc-600">
        {ZONE_ORDER.map((z, i) => (
          <span key={z} className="flex items-center gap-1">
            <span className={cx("size-2 rounded-sm", shades[i])} />
            {ZONE_LABEL[z]} {pct(zones[z])}
          </span>
        ))}
      </div>
    </div>
  );
}

const FORMATION_LABEL: Record<string, string> = {
  both_net: "Both at the net",
  both_back: "Both at the back",
  split: "One up, one back",
  other: "One in transition",
};

function TeamCard({ team: t, pairs }: { team: TeamStats; pairs: boolean }) {
  return (
    <Card
      title={
        <span className="flex items-center gap-2">
          <TeamDot team={t.team} /> {teamName(t.team, pairs)} · players {t.team === "near" ? "1 and 2" : "3 and 4"}
        </span>
      }
      description="Where the partners stand relative to each other"
    >
      {t.insufficient_data ? (
        <p className="text-sm text-zinc-600">Not enough data: {t.insufficient_data}.</p>
      ) : (
        <div className="grid gap-6 sm:grid-cols-[1fr_auto]">
          <div className="space-y-2">
            {Object.entries(t.formation ?? {}).map(([k, v]) => (
              <div key={k} className="grid grid-cols-[140px_1fr_44px] items-center gap-3 text-sm">
                <span className="text-zinc-600">{FORMATION_LABEL[k] ?? k}</span>
                <div className="h-1.5 overflow-hidden rounded-full bg-zinc-100">
                  <div className={t.team === "near" ? "h-full bg-near" : "h-full bg-far"} style={{ width: `${v * 100}%` }} />
                </div>
                <span className="tabular text-right text-zinc-900">{pct(v)}</span>
              </div>
            ))}
          </div>
          <div className="border-zinc-100 sm:border-l sm:pl-6">
            <div className="text-xs text-zinc-500">Partner spacing</div>
            <div className="tabular mt-1 text-xl font-semibold text-zinc-900">{num(t.partner_spacing_m, 1, "m")}</div>
            <div className="tabular text-xs text-zinc-500">both tracked {pct(t.both_tracked_fraction)}</div>
          </div>
        </div>
      )}
    </Card>
  );
}
