"use client";

import { API_BASE } from "@/lib/api";
import { Layers, Maximize2 } from "lucide-react";

interface HeatmapViewProps {
  p1Url?: string;
  p2Url?: string;
  p3Url?: string;
  p4Url?: string;
}

const playerConfigs = [
  { id: 1, name: "PLAYER 1", color: "#0250B0", hoverColor: "#0250B0" },
  { id: 2, name: "PLAYER 2", color: "#EC4899", hoverColor: "#D0FF41" },
  { id: 3, name: "PLAYER 3", color: "#0250B0", hoverColor: "#0250B0" },
  { id: 4, name: "PLAYER 4", color: "#EC4899", hoverColor: "#D0FF41" },
] as const;

export default function HeatmapView({ p1Url, p2Url, p3Url, p4Url }: HeatmapViewProps) {
  const urls = [p1Url, p2Url, p3Url, p4Url];
  if (urls.every((u) => !u)) return null;

  const getFullUrl = (url: string) =>
    url.startsWith("http") ? url : `${API_BASE.replace("/api", "")}${url}`;

  return (
    <div className="p-6 md:p-8 rounded-3xl bg-[#0A0F1D]/80 backdrop-blur-2xl border border-white/5 shadow-2xl space-y-6 hover:shadow-cyan-500/10 transition-all duration-500">
      <div className="flex items-center justify-between pb-5 border-b border-white/5">
        <h3 className="text-lg font-black text-white tracking-tight flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20">
            <Layers className="w-5 h-5" />
          </div>
          Court Coverage Heatmaps
        </h3>
        <span className="text-xs font-bold px-3 py-1.5 rounded-lg bg-white/5 text-white/50 border border-white/10 uppercase tracking-widest">
          4 Players
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        {urls.map((url, idx) => {
          if (!url) return null;
          const config = playerConfigs[idx];
          return (
            <div key={idx} className="flex flex-col items-center gap-3 group">
              <div className="relative w-full rounded-2xl bg-white/[0.02] border border-white/10 overflow-hidden hover:border-cyan-500/50 hover:shadow-[0_0_20px_rgba(34,211,238,0.15)] transition-all duration-500 aspect-[4/5]">
                <span className="absolute top-3 left-3 z-10 px-2.5 py-1 rounded-lg text-[10px] font-black text-white uppercase tracking-widest shadow-lg" style={{ backgroundColor: config.color }}>
                  {config.name}
                </span>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={getFullUrl(url)}
                  alt={`${config.name} Heatmap`}
                  className="w-full h-full object-contain group-hover:scale-[1.03] group-hover:rotate-1 transition-transform duration-700 ease-out"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[#0A0F1D] via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 flex items-end justify-center pb-4">
                  <Maximize2 className="w-5 h-5 text-white/70" />
                </div>
              </div>
              <span className="text-[11px] font-bold uppercase tracking-widest text-white/50 group-hover:text-white/80 transition-colors duration-300">
                {config.name} Coverage
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
