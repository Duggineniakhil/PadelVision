"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { ArrowRight } from "lucide-react";
import { useRallies } from "@/lib/api";
import { duration, num, pct, teamName, ZONE_LABEL } from "@/lib/format";
import { useMatch } from "@/components/match";
import { RallyTimeline } from "@/components/timeline";
import { Callout, Card, Skeleton, Stat, TeamDot } from "@/components/ui";

export function OverviewPage() {
  const { id, data: run, ready } = useMatch();
  const { data: rallies } = useRallies(id, ready);
  const router = useRouter();
  if (!run) return null;
  const rs = run.rally_summary;
  const sp = run.speed_summary;
  const pts = run.points_summary;
  const matchS = run.video?.duration_s ?? 0;

  return (
    <div className="space-y-5">
      <Card bodyClassName="grid grid-cols-2 gap-x-6 gap-y-5 p-5 md:grid-cols-4">
        <Stat label="Match length" value={duration(matchS)} hint={run.video ? `${run.video.frame_count.toLocaleString("en-GB")} frames` : undefined} />
        <Stat label="Rallies" value={num(rs?.count)} hint={rs ? `${duration(rs.total_s)} of play` : undefined} />
        <Stat label="Shots in rallies" value={num(rs?.hits_total)} hint={rs?.hits_per_rally != null ? `${num(rs.hits_per_rally, 1)} per rally` : undefined} />
        <Stat label="Longest rally" value={rs?.max_s != null ? `${num(rs.max_s, 1)} s` : "–"} hint={rs?.median_s != null ? `median ${num(rs.median_s, 1)} s` : undefined} />
        <Stat
          label="Fastest shot (est.)"
          value={sp?.max_kmh != null ? `${num(sp.max_kmh)} km/h` : "–"}
          hint={sp?.median_kmh != null ? `median ${num(sp.median_kmh)} km/h · ${sp.samples} shots` : "not enough measured shots"}
        />
        <Stat label="Ball found" value={pct(run.ball_coverage)} hint="share of frames" />
        <Stat
          label="Points"
          value={num(pts?.points)}
          hint={pts ? `from ${pts.serves} serves found, ${pts.confirmed_serves} confirmed` : undefined}
        />
        <Stat
          label="Games"
          value={pts?.games != null ? num(pts.games) : "–"}
          hint={pts ? (pts.games != null ? "estimate from serves" : "not countable yet: serves missed") : undefined}
        />
      </Card>

      {run.quality.length > 0 && (
        <div className="space-y-2">
          {run.quality.map((q, i) => (
            <Callout key={i} level={q.level} title={q.title}>
              {q.detail}
            </Callout>
          ))}
        </div>
      )}

      <Card
        title="Rallies over the match"
        description="Each bar is a rally. Select one to see its shots."
        action={
          <Link href={`/matches/${id}/rallies`} className="inline-flex items-center gap-1 text-sm font-medium text-accent hover:underline">
            All rallies <ArrowRight className="size-3.5" />
          </Link>
        }
      >
        {rallies ? (
          <RallyTimeline
            duration={matchS}
            rallies={rallies.rallies}
            onSelect={(r) => router.push(`/matches/${id}/rallies?rally=${r.id}`)}
          />
        ) : (
          <Skeleton className="h-14 w-full" />
        )}
      </Card>

      <div className="grid gap-5 lg:grid-cols-3">
        <Card
          className="lg:col-span-2"
          title="Players"
          action={
            <Link href={`/matches/${id}/players`} className="inline-flex items-center gap-1 text-sm font-medium text-accent hover:underline">
              Heatmaps and details <ArrowRight className="size-3.5" />
            </Link>
          }
          bodyClassName="p-0"
        >
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-zinc-500">
              <tr className="border-b border-zinc-100">
                <th className="px-4 py-2 font-medium">Player</th>
                <th className="px-4 py-2 text-right font-medium">Tracked</th>
                <th className="px-4 py-2 text-right font-medium">Distance</th>
                <th className="px-4 py-2 text-right font-medium">Avg speed</th>
                <th className="px-4 py-2 text-right font-medium">Peak speed</th>
                <th className="px-4 py-2 text-right font-medium">Rally shots</th>
              </tr>
            </thead>
            <tbody className="tabular divide-y divide-zinc-100">
              {run.players.map((p) => (
                <tr key={p.player}>
                  <td className="px-4 py-2.5">
                    <div className="flex items-center gap-2">
                      <TeamDot team={p.team} />
                      <span className="font-medium text-zinc-900">Player {p.player}</span>
                      <span className="text-xs text-zinc-500">{teamName(p.team, run.changeovers.length > 0)}</span>
                    </div>
                  </td>
                  <td className="px-4 py-2.5 text-right text-zinc-700">{pct(p.tracked_fraction)}</td>
                  {p.insufficient_data ? (
                    <td colSpan={3} className="px-4 py-2.5 text-right text-xs text-zinc-500">
                      Not enough tracking for movement stats
                    </td>
                  ) : (
                    <>
                      <td className="px-4 py-2.5 text-right text-zinc-700">{num(p.distance_m, 0, "m")}</td>
                      <td className="px-4 py-2.5 text-right text-zinc-700">{num(p.avg_speed_kmh, 1, "km/h")}</td>
                      <td className="px-4 py-2.5 text-right text-zinc-700">{num(p.peak_speed_kmh, 1, "km/h")}</td>
                    </>
                  )}
                  <td className="px-4 py-2.5 text-right text-zinc-700">{p.rally_hits}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Measurement accuracy" description="How precise court positions are, by area of the court.">
          {run.court ? (
            <dl className="space-y-2.5 text-sm">
              {Object.entries(run.court.zones).map(([zone, z]) => (
                <div key={zone} className="flex items-baseline justify-between gap-4">
                  <dt className="text-zinc-600">{ZONE_LABEL[zone] ?? zone}</dt>
                  <dd className="tabular font-medium text-zinc-900">
                    {z.expected_error_m != null ? `± ${num(z.expected_error_m, 2)} m` : "–"}
                  </dd>
                </div>
              ))}
              <div className="flex items-baseline justify-between gap-4 border-t border-zinc-100 pt-2.5">
                <dt className="text-zinc-600">Calibration fit</dt>
                <dd className="tabular font-medium text-zinc-900">{num(run.court.rms_px, 1)} px</dd>
              </div>
            </dl>
          ) : (
            <p className="text-sm text-zinc-500">No court calibration in this run.</p>
          )}
        </Card>
      </div>
    </div>
  );
}
