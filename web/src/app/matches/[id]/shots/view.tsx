"use client";

import Link from "next/link";
import { useMemo } from "react";
import { useRallies, type Shot } from "@/lib/api";
import { clock, num, pct, teamName, ZONE_LABEL } from "@/lib/format";
import { CourtSvg, LandingLayer } from "@/components/court";
import { useMatch } from "@/components/match";
import { Card, EmptyState, Skeleton, Stat, TeamDot } from "@/components/ui";

const ZONES = ["net", "transition", "back"];

export function ShotsPage() {
  const { id, data: run, ready } = useMatch();
  const { data } = useRallies(id, ready);
  if (!run) return null;
  if (!data) return <Skeleton className="h-96 w-full" />;
  if (data.shots.length === 0)
    return <EmptyState title="No shots">Shots are counted inside rallies, and none were detected.</EmptyState>;

  const shots = data.shots;
  const measured = shots.filter((s) => s.speed_kmh != null).sort((a, b) => b.speed_kmh! - a.speed_kmh!);
  const sp = run.speed_summary;

  return (
    <div className="space-y-5">
      <Card bodyClassName="grid grid-cols-2 gap-x-6 gap-y-5 p-5 md:grid-cols-4">
        <Stat label="Shots in rallies" value={num(shots.length)} />
        <Stat
          label="Landing seen"
          value={num(shots.filter((s) => s.bounce).length)}
          hint={pct(shots.filter((s) => s.bounce).length / shots.length)}
        />
        <Stat
          label="Probably volleys"
          value={num(shots.filter((s) => s.no_landing === "probably_volley").length)}
          hint="hit back before bouncing"
        />
        <Stat label="Speed measured" value={num(measured.length)} hint={sp?.median_kmh != null ? `median ${num(sp.median_kmh)} km/h` : "too few to summarise"} />
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        {(["near", "far"] as const).map((team) => (
          <Placement key={team} team={team} shots={shots} />
        ))}
      </div>

      <Card
        title="Shot speed"
        description="Average ground speed from the hitter to where the ball landed. The ball leaves the racket faster, so read these as lower bounds. Only shots with a clearly tracked landing are measured; bars show the uncertainty."
        bodyClassName="p-0"
      >
        {measured.length === 0 ? (
          <p className="px-4 py-6 text-sm text-zinc-500">No shot had a clearly tracked landing.</p>
        ) : (
          <SpeedTable id={id} shots={measured} />
        )}
      </Card>
    </div>
  );
}

function Placement({ team, shots }: { team: "near" | "far"; shots: Shot[] }) {
  const mine = shots.filter((s) => s.team === team);
  const landed = mine.filter((s) => s.bounce);
  const counts = Object.fromEntries(ZONES.map((z) => [z, landed.filter((s) => s.bounce!.zone === z).length]));
  return (
    <Card
      title={
        <span className="flex items-center gap-2">
          <TeamDot team={team} /> Where the {teamName(team).toLowerCase()}&apos;s shots landed
        </span>
      }
      description={`${landed.length} of ${mine.length} landings seen, on the other side of the net`}
    >
      <div className="grid grid-cols-[180px_1fr] items-center gap-6">
        <CourtSvg label={`Landing points of the ${teamName(team).toLowerCase()}'s shots`} className="w-full">
          <LandingLayer shots={shots} team={team} />
        </CourtSvg>
        <div className="space-y-3">
          {ZONES.map((z) => (
            <div key={z}>
              <div className="flex justify-between text-sm">
                <span className="text-zinc-600">{ZONE_LABEL[z]}</span>
                <span className="tabular font-medium text-zinc-900">
                  {counts[z]}
                  <span className="ml-1 font-normal text-zinc-500">{landed.length ? pct(counts[z] / landed.length) : ""}</span>
                </span>
              </div>
              <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-zinc-100">
                <div
                  className={team === "near" ? "h-full bg-near" : "h-full bg-far"}
                  style={{ width: `${landed.length ? (100 * counts[z]) / landed.length : 0}%` }}
                />
              </div>
            </div>
          ))}
          <p className="pt-1 text-xs text-zinc-500">Zones by distance from the net: net up to 4 m, back court behind the service line.</p>
        </div>
      </div>
    </Card>
  );
}

function SpeedTable({ id, shots }: { id: string; shots: Shot[] }) {
  const max = useMemo(() => Math.max(...shots.map((s) => s.speed_range_kmh?.[1] ?? s.speed_kmh!)) * 1.05, [shots]);
  const x = (v: number) => `${(100 * v) / max}%`;
  return (
    <table className="w-full text-sm">
      <thead className="text-left text-xs text-zinc-500">
        <tr className="border-b border-zinc-100">
          <th className="px-4 py-2 font-medium">Player</th>
          <th className="px-4 py-2 font-medium">Rally</th>
          <th className="px-4 py-2 font-medium">Time</th>
          <th className="px-4 py-2 text-right font-medium">Speed</th>
          <th className="w-[34%] px-4 py-2 font-medium">Uncertainty</th>
        </tr>
      </thead>
      <tbody className="tabular divide-y divide-zinc-50">
        {shots.map((s) => (
          <tr key={`${s.rally_id}-${s.index}`} className="hover:bg-zinc-50">
            <td className="px-4 py-2">
              <span className="flex items-center gap-2">
                <TeamDot team={s.team} /> Player {s.player}
              </span>
            </td>
            <td className="px-4 py-2">
              <Link href={`/matches/${id}/rallies?rally=${s.rally_id}`} className="text-accent hover:underline">
                Rally {s.rally_id}
              </Link>
              <span className="text-zinc-400"> · shot {s.index}</span>
            </td>
            <td className="px-4 py-2 text-zinc-600">{clock(s.t)}</td>
            <td className="px-4 py-2 text-right">
              <span className="font-medium text-zinc-900">~{num(s.speed_kmh)} km/h</span>
              {s.speed_range_kmh && (
                <span className="ml-2 text-xs text-zinc-400">
                  {s.speed_range_kmh[0]}–{s.speed_range_kmh[1]}
                </span>
              )}
            </td>
            <td className="px-4 py-2">
              {s.speed_range_kmh && (
                <div className="relative h-4" title={`${s.speed_range_kmh[0]}–${s.speed_range_kmh[1]} km/h`}>
                  <div className="absolute inset-x-0 top-1/2 h-px bg-zinc-100" />
                  <div
                    className="absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full bg-zinc-300"
                    style={{ left: x(s.speed_range_kmh[0]), width: x(s.speed_range_kmh[1] - s.speed_range_kmh[0]) }}
                  />
                  <div className="absolute top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-zinc-800" style={{ left: x(s.speed_kmh!) }} />
                </div>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
