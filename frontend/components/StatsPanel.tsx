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
    <div className="p-6 md:p-8 rounded-3xl bg-[#0A0F1D]/80 backdrop-blur-2xl border border-white/5 shadow-2xl space-y-8 hover:shadow-cyan-500/10 transition-all duration-500">
      {/* Header */}
      <div className="flex items-center justify-between pb-5 border-b border-white/5">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20">
            <Users className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg font-black text-white tracking-tight">Match Statistics</h3>
            <p className="text-xs font-medium text-white/50 uppercase tracking-widest mt-0.5">2 Teams • 4 Players</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-white/50 uppercase tracking-wider hidden sm:inline-block">Team B</span>
          <div className="px-3 py-1.5 rounded-lg bg-pink-500/20 border border-pink-500/30 text-pink-400 font-bold text-xs tracking-wider">
            P3 + P4
          </div>
        </div>
      </div>

      {/* Team Header Row */}
      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-6">
        <div className="flex flex-col items-start gap-1">
          <div className="px-3 py-1.5 rounded-lg bg-cyan-500/20 border border-cyan-500/30 text-cyan-400 font-bold text-xs tracking-wider mb-2">
            P1 + P2
          </div>
          <span className="text-sm font-black text-white uppercase tracking-widest">Team A</span>
        </div>
        <div className="w-[1px] h-12 bg-white/10" />
        <div className="flex flex-col items-end gap-1">
          <div className="px-3 py-1.5 rounded-lg bg-pink-500/20 border border-pink-500/30 text-pink-400 font-bold text-xs tracking-wider mb-2">
            P3 + P4
          </div>
          <span className="text-sm font-black text-white uppercase tracking-widest">Team B</span>
        </div>
      </div>

      {/* Metrics List */}
      <div className="space-y-6">
        {metrics.map((m) => {
          const valA = Number(teamA[m.key] || 0);
          const valB = Number(teamB[m.key] || 0);
          const total = Math.max(valA + valB, 1);
          const pctA = Math.round((valA / total) * 100);
          const pctB = 100 - pctA;
          const aWins = valA >= valB;

          return (
            <div key={m.key} className="space-y-2 group">
              <div className="flex items-center justify-between text-sm font-semibold">
                <span className={`font-mono text-base ${aWins ? "text-cyan-400 font-black drop-shadow-[0_0_8px_rgba(34,211,238,0.5)]" : "text-white/60"}`}>
                  {valA} <span className="text-[10px] text-white/40 font-sans ml-0.5">{m.unit}</span>
                </span>

                <span className="text-white/50 uppercase tracking-widest flex items-center gap-1.5 text-[11px] font-bold">
                  <m.icon className="w-3.5 h-3.5 group-hover:text-white transition-colors duration-300" />
                  {m.label}
                </span>

                <span className={`font-mono text-base ${!aWins ? "text-pink-400 font-black drop-shadow-[0_0_8px_rgba(244,114,182,0.5)]" : "text-white/60"}`}>
                  {valB} <span className="text-[10px] text-white/40 font-sans ml-0.5">{m.unit}</span>
                </span>
              </div>

              {/* Progress Comparison Bar */}
              <div className="h-3 w-full bg-black/40 rounded-full overflow-hidden flex gap-1 p-0.5 border border-white/5 shadow-inner">
                <div
                  style={{ width: `${pctA}%` }}
                  className={`h-full rounded-l-full transition-all duration-1000 ease-out ${
                    aWins ? "bg-gradient-to-r from-blue-600 to-cyan-400" : "bg-white/10"
                  }`}
                />
                <div
                  style={{ width: `${pctB}%` }}
                  className={`h-full rounded-r-full transition-all duration-1000 ease-out ${
                    !aWins ? "bg-gradient-to-l from-rose-600 to-pink-400" : "bg-white/10"
                  }`}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Individual Player Breakdown */}
      <div className="pt-6 border-t border-white/5">
        <h4 className="text-[11px] font-bold uppercase tracking-widest text-white/50 mb-4 flex items-center gap-2">
          <Activity className="w-4 h-4" />
          Individual Breakdown
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {players.map(({ p, name, color, label }) => (
            <div key={label} className="p-4 rounded-2xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.04] transition-colors duration-300">
              <div className="flex items-center gap-3 mb-4">
                <div className={`w-8 h-8 rounded-lg ${color} flex items-center justify-center font-black text-xs text-white shadow-md`}>
                  {label}
                </div>
                <span className="text-sm font-bold text-white tracking-wide">{name}</span>
              </div>
              <div className="grid grid-cols-2 gap-y-3 gap-x-2 text-xs">
                <div className="flex flex-col">
                  <span className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">Shots</span>
                  <span className="font-mono text-white text-sm mt-0.5">{p.total_shots}</span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">Avg Spd</span>
                  <span className="font-mono text-white text-sm mt-0.5">{p.avg_shot_speed} <span className="text-[9px] text-white/30">km/h</span></span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">Max Spd</span>
                  <span className="font-mono text-white text-sm mt-0.5">{p.max_shot_speed} <span className="text-[9px] text-white/30">km/h</span></span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] uppercase tracking-wider text-white/40 font-semibold">Distance</span>
                  <span className="font-mono text-white text-sm mt-0.5">{p.distance_covered} <span className="text-[9px] text-white/30">m</span></span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
