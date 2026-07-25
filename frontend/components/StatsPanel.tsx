"use client";

import { Activity, Gauge, Zap, Footprints, ShieldAlert, Users } from "lucide-react";

interface PlayerMetrics {
  total_shots: number;
  avg_shot_speed: number;
  max_shot_speed: number;
  avg_player_speed: number;
  distance_covered: number;
}

interface TeamMetrics {
  total_shots: number;
  avg_shot_speed: number;
  max_shot_speed: number;
  avg_player_speed: number;
  distance_covered: number;
}

type MetricKey = keyof TeamMetrics;

interface StatsProps {
  player1?: PlayerMetrics;
  player2?: PlayerMetrics;
  player3?: PlayerMetrics;
  player4?: PlayerMetrics;
  player1Name?: string;
  player2Name?: string;
  player3Name?: string;
  player4Name?: string;
}

const EMPTY_METRICS: PlayerMetrics = {
  total_shots: 0,
  avg_shot_speed: 0,
  max_shot_speed: 0,
  avg_player_speed: 0,
  distance_covered: 0,
};

function combineTeam(p1: PlayerMetrics | undefined, p2: PlayerMetrics | undefined): TeamMetrics {
  const p1Data = p1 || EMPTY_METRICS;
  const p2Data = p2 || EMPTY_METRICS;
  return {
    total_shots: p1Data.total_shots + p2Data.total_shots,
    avg_shot_speed: Math.round(((p1Data.avg_shot_speed || 0) + (p2Data.avg_shot_speed || 0)) / 2),
    max_shot_speed: Math.max(p1Data.max_shot_speed || 0, p2Data.max_shot_speed || 0),
    avg_player_speed: Math.round(((p1Data.avg_player_speed || 0) + (p2Data.avg_player_speed || 0)) / 2),
    distance_covered: Math.round((p1Data.distance_covered || 0) + (p2Data.distance_covered || 0)),
  };
}

export default function StatsPanel({
  player1,
  player2,
  player3,
  player4,
  player1Name = "Player 1",
  player2Name = "Player 2",
  player3Name = "Player 3",
  player4Name = "Player 4",
}: StatsProps) {
  const players = [
    { p: player1 || EMPTY_METRICS, name: player1Name, color: "bg-[#0250B0]", label: "P1" },
    { p: player2 || EMPTY_METRICS, name: player2Name, color: "bg-blue-500", label: "P2" },
    { p: player3 || EMPTY_METRICS, name: player3Name, color: "bg-pink-600", label: "P3" },
    { p: player4 || EMPTY_METRICS, name: player4Name, color: "bg-rose-500", label: "P4" },
  ];
  const teamA = combineTeam(player1, player2);
  const teamB = combineTeam(player3, player4);

  const metrics = [
    { label: "Total Shots", key: "total_shots", unit: "", icon: Activity },
    { label: "Avg Shot Speed", key: "avg_shot_speed", unit: "km/h", icon: Gauge },
    { label: "Max Shot Speed", key: "max_shot_speed", unit: "km/h", icon: Zap },
    { label: "Avg Movement", key: "avg_player_speed", unit: "km/h", icon: Footprints },
    { label: "Distance Run", key: "distance_covered", unit: "m", icon: ShieldAlert },
  ] satisfies { label: string; key: MetricKey; unit: string; icon: typeof Activity }[];

  return (
    <div className="p-6 rounded-2xl bg-[#131B2E] border border-[#1E2A40] shadow-xl space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-[#1E2A40]">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-[#0250B0] flex items-center justify-center font-bold text-xs text-white">
            <Users className="w-4 h-4" />
          </div>
          <h3 className="text-base font-bold text-white">Padel Match Stats</h3>
        </div>

        <div className="px-3 py-1 rounded-full bg-[#0E1626] border border-[#1E2A40] text-xs font-bold text-[#8E9BAE] uppercase tracking-wider">
          2 Teams / 4 Players
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-[#8E9BAE] uppercase">Team B</span>
          <div className="w-7 h-7 rounded-lg bg-pink-600 flex items-center justify-center font-bold text-xs text-white">
            P3+P4
          </div>
        </div>
      </div>

      {/* Team Header Row */}
      <div className="grid grid-cols-3 gap-4 text-center">
        <div className="col-span-1" />
        <div className="flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-[#0250B0]/20 border border-[#0250B0]/30">
          <div className="w-6 h-6 rounded-lg bg-[#0250B0] flex items-center justify-center font-bold text-xs text-white">
            P1+P2
          </div>
          <span className="text-sm font-semibold text-white">Team A</span>
        </div>
        <div className="flex items-center justify-center gap-2 px-3 py-2 rounded-xl bg-pink-600/20 border border-pink-600/30">
          <span className="text-sm font-semibold text-white">Team B</span>
          <div className="w-6 h-6 rounded-lg bg-pink-600 flex items-center justify-center font-bold text-xs text-white">
            P3+P4
          </div>
        </div>
      </div>

      {/* Metrics List */}
      <div className="space-y-5">
        {metrics.map((m) => {
          const valA = Number(teamA[m.key] || 0);
          const valB = Number(teamB[m.key] || 0);
          const total = Math.max(valA + valB, 1);
          const pctA = Math.round((valA / total) * 100);
          const pctB = 100 - pctA;
          const aWins = valA >= valB;

          return (
            <div key={m.key} className="space-y-1.5">
              <div className="flex items-center justify-between text-xs font-semibold">
                <span className={`font-mono text-sm ${aWins ? "text-[#D0FF41] font-bold" : "text-[#C6D0DD]"}`}>
                  {valA} {m.unit}
                </span>

                <span className="text-[#8E9BAE] uppercase tracking-wider flex items-center gap-1">
                  <m.icon className="w-3 h-3" />
                  {m.label}
                </span>

                <span className={`font-mono text-sm ${!aWins ? "text-[#D0FF41] font-bold" : "text-[#C6D0DD]"}`}>
                  {valB} {m.unit}
                </span>
              </div>

              {/* Progress Comparison Bar */}
              <div className="h-2.5 w-full bg-[#0E1626] rounded-full overflow-hidden flex gap-0.5 p-0.5 border border-[#1E2A40]">
                <div
                  style={{ width: `${pctA}%` }}
                  className={`h-full rounded-l-full transition-all duration-500 ${
                    aWins ? "bg-[#0250B0]" : "bg-[#1E2A40]"
                  }`}
                />
                <div
                  style={{ width: `${pctB}%` }}
                  className={`h-full rounded-r-full transition-all duration-500 ${
                    !aWins ? "bg-pink-500" : "bg-[#1E2A40]"
                  }`}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Individual Player Breakdown */}
      <div className="pt-4 border-t border-[#1E2A40]">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-[#8E9BAE] mb-3 flex items-center gap-2">
          <Users className="w-4 h-4" />
          Individual Breakdown
        </h4>
        <div className="grid grid-cols-2 gap-3">
          {players.map(({ p, name, color, label }) => (
            <div key={label} className="p-3 rounded-xl bg-[#0E1626] border border-[#1E2A40]">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className={`w-6 h-6 rounded-lg ${color} flex items-center justify-center font-bold text-xs text-white`}>
                    {label}
                  </div>
                  <span className="text-sm font-semibold text-white">{name}</span>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-[#8E9BAE]">Shots:</span>
                  <span className="font-mono text-white ml-1">{p.total_shots}</span>
                </div>
                <div>
                  <span className="text-[#8E9BAE]">Avg Speed:</span>
                  <span className="font-mono text-white ml-1">{p.avg_shot_speed} km/h</span>
                </div>
                <div>
                  <span className="text-[#8E9BAE]">Max Speed:</span>
                  <span className="font-mono text-white ml-1">{p.max_shot_speed} km/h</span>
                </div>
                <div>
                  <span className="text-[#8E9BAE]">Dist:</span>
                  <span className="font-mono text-white ml-1">{p.distance_covered}m</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
